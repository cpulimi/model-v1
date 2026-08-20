#!/usr/bin/env python3
"""
Standalone FSL QUBO solver running on the NEC Vector Annealing (VA) engine.

This file is fully self-contained: it does NOT import
run_aligned_fsl_comparison.py. Every helper it needs -- data loading, the
aligned cost basis, batching, penalty weighting, sample evaluation,
post-processing and output schemas -- is inlined below, so the solver can be
copied to a VA node on its own. openjij is never imported here, at any scope.

PROBLEM FORMULATION: the QUBO is built natively with pyqubo. Decision variables
Z_ijk / Y_jk / X_j are pyqubo `Binary` objects, the objective and the C1-C3
constraints are written as mathematical expressions, and
`.compile().to_qubo()` produces the raw {(u, v): coeff} dictionary and the
constant offset handed to the VA sampler. The penalty weights are pyqubo
`Placeholder`s, so a batch is compiled ONCE and each adaptive-penalty iteration
only re-feeds new penalty values -- no rebuild, no recompile.

The encoding is unchanged from the hand-rolled dictionary this replaced:
variable names, constraint encodings and objective coefficients are identical,
so results stay comparable to the OpenJij arm. The one visible difference is
that the C1 "exactly one hub" constant, which the manual construction silently
dropped, now surfaces explicitly as `to_qubo()`'s offset. See VA OFFSET below.

PENALTY SCALE: objective-scale normalization is OFF by default here, so C1-C4
penalties sit flat at --min-penalty (50,000) instead of being lifted to
~scale*multiplier (~2.51M on instances_low). That makes this run comparable to
the NO-SCALE OpenJij arm (run_aligned_fsl_comparison_noscale.py, e.g.
scripts/adaptive_scale_sweep_noscale.sh), and NOT to scale-ON baselines.
Pass --enable-objective-scale to restore the scale-ON penalties.

    python run_va_fsl_solver.py --dataset-dir instances_low --dry-run
    python run_va_fsl_solver.py --dataset-dir instances_low --run-root results/va_low

VA OFFSET
---------
`to_qubo()` returns (Q, offset). The offset is the C1 constant, lam_c1 times the
number of active demand rows in the batch -- 250M-scale on instances_low. It is
recorded everywhere (batch plan CSV, batch summary, summary.json) but is NOT
handed to VectorAnnealing.model() unless --va-include-offset is passed, for two
reasons:

  * VA reports energies in single precision. Adding a ~2.5e8 constant to every
    energy costs about 5 significant digits of resolution and would swamp the
    fp32 precision audit this script exists to measure.
  * Energies stay directly comparable to the OpenJij baseline, which samples the
    same Q with no offset.

The offset is a constant, so including it shifts every energy equally and cannot
change which sample is selected. Add `offset` to a reported energy to recover the
true Hamiltonian value.

DELIBERATE SCOPE NOTES
----------------------
* No seeding. The VA PoC API exposes no seed parameter, so VA runs are not
  reproducible read-for-read. Use --va-repeats N to characterize the spread
  instead; per-repeat statistics land in va_batch_summary.csv.
* Adaptive penalty IS implemented, and is ON by default: rebuild the QUBO with
  grown multipliers for violated constraints, resample, repeat until feasible.
  The seed line the OpenJij loop carries is dropped, since VA has no seed and
  the escalation never depended on one. It is on by default because with
  objective scale OFF, C3's penalty starts at 50,000 against an S_lim of
  500,000, so stocking at a closed hub is initially 10x cheaper than opening
  one; only the escalation corrects that.
  --adaptive-penalty-iterations defaults to 8 rather than the OpenJij path's 5,
  because ceil(log(500000/50000)/log(1.5)) = 6 iterations are needed to clear
  S_lim. Pass --adaptive-penalty-mode off for a single static pass.
* No retry-reads escalation (--max-stages / --retry-reads-boost). Each batch
  gets one VA sampling call per repeat, plus constraint-failure retries.

Outputs land in <run-root>/va, a sibling of the existing "qubo" and "gurobi"
directories, using the same filenames and column schemas the OpenJij runner
writes, so run_aligned_fsl_comparison.build_combined_outputs and the existing
comparison tooling can read it unchanged.

EXECUTION MODEL: LOCAL VE CARD ONLY
-----------------------------------
pyqubo is the modeling layer; the locally installed `VectorAnnealing` module is
the hardware execution layer, driving the physical NEC Vector Engine card in the
node this process runs on. This is exactly the pattern the PoC manual documents:

    qubo, offset = model.to_qubo(feed_dict={...})   # pyqubo
    import VectorAnnealing                          # local on-prem module
    va_model = VectorAnnealing.model(qubo, offset)
    result = VectorAnnealing.sampler().sample(va_model, num_reads=...)

No cloud or service-client path exists anywhere in this file: there is no
SACServiceClient, no REST/HTTP call, no endpoint, no credential. The solver
imports the on-disk module and talks to the card through it. `import_vector_
annealing()` reports the resolved module path and the visible /dev/ve* devices
at startup so each run's log proves it executed on local hardware.

On ASU SOL that card lives on node `sfpga01n`; see va_solve.sh, which pins
`#SBATCH -w sfpga01n`. Run va_probe.py there first to confirm the install.

Environment:
    Python 3.8+, numpy >= 1.22.3, pyqubo >= 1.0.13, pandas, and the VA python
    directory on PYTHONPATH, e.g.
        export PYTHONPATH=/opt/va/V3.0.0/libexec/VectorAnnealing/python:${PYTHONPATH}
    The 2022 PoC manual documents release VApoc_0201; SOL carries V3.0.0, so
    the path differs per install. VA_CANDIDATE_GLOB below is used only to build
    a helpful error message -- it never changes what gets imported. The VA
    module version is printed at startup when it exposes one.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import socket
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from pyqubo import Binary, Placeholder
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "ERROR: pyqubo is required to build the QUBO.\n"
        f"  underlying error: {exc}\n"
        "  install it with:  pip install 'pyqubo>=1.0.13'"
    )

try:
    import psutil  # type: ignore
except ModuleNotFoundError:
    psutil = None


# VA hard specification, from the PoC manual.
VA_HARD_MAX_VARS = 100_000          # "Binary data size: up to 100 thousand bit"
VA_DENSE_BYTES_PER_ENTRY = 4        # 32-bit resolution, dense full-connection matrix
VA_MANUAL_REF = "NEC Vector Annealing PoC Manual, 2nd Edition (Nov 2022)"

# Where on-prem VA installs live. Used ONLY to build actionable error text and
# to report which install was picked up -- never to import anything implicitly.
VA_CANDIDATE_GLOB = "/opt/va/*/libexec/VectorAnnealing/python"
# The physical VE cards appear as these device nodes on the executing host.
VE_DEVICE_GLOBS = ("/dev/veslot*", "/dev/ve[0-9]*")

# Coefficients at or below this magnitude are dropped from the compiled QUBO.
# The hand-rolled construction this replaced applied the same tolerance when
# accumulating terms, so pruning keeps the interaction counts in batch_summary.csv
# comparable with the historical OpenJij runs.
QUBO_ZERO_TOLERANCE = 1e-12

SCALAR_COLUMNS = (
    "C",
    "h_s",
    "h_d",
    "d_s",
    "L",
    "S_lim",
    "S_var",
    "lambda_1",
    "lambda_2",
    "lambda_3",
    "base_miles",
    "penalty_start_miles",
    "max_service_miles",
)

REQUIRED_FILES = (
    "demand.csv",
    "distances.csv",
    "hubs.csv",
    "parameters.csv",
    "parts.csv",
    "zips.csv",
)

ID_COLUMNS = {"hub_id", "zip_id", "part_id", "anchor_id", "region_code"}


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    row_indices: list[int]
    estimated_z_vars: int
    note: str = ""


@dataclass
class BatchResult:
    batch_id: int
    num_rows: int
    num_parts: int
    num_z: int
    num_y: int
    num_x: int
    interactions: int
    build_seconds: float
    sample_seconds: float
    eval_seconds: float
    total_seconds: float
    stage_used: int
    suggested_reads: int
    energy: float
    reconstructed_cost: float
    c1_violations: int
    c2_violations: int
    c3_violations: int
    c4_violations: int
    penalty_c1: float
    penalty_c2: float
    penalty_c3: float
    open_hubs: list[str]
    stocked_pairs: list[tuple[str, str]]
    assignments: list[tuple[str, str, str]]
    adaptive_iterations_used: int = 0
    adaptive_was_feasible: bool = False
    final_penalty_multipliers: dict[str, float] = field(
        default_factory=lambda: {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    )
    adaptive_iteration_log: list[dict] = field(default_factory=list)
    adaptive_exit_reason: str = ""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def money(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    return f"${float(x):,.2f}"


def current_rss_mb() -> float:
    """Current resident set size in MB, or 0.0 if it cannot be read."""
    if psutil is not None:
        try:
            return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass
    return 0.0


def peak_rss_mb() -> float:
    """TRUE high-water-mark RSS in MB for this process.

    getrusage(RUSAGE_SELF).ru_maxrss is a monotonic peak the kernel maintains,
    so it survives the frees and gc between batches. psutil's memory_info().rss
    is only the CURRENT value: because memory_report_mb() runs at the end of a
    run, after the batch QUBOs have been freed, using it silently understated
    peak memory by the size of the largest batch. On this workload memory is the
    binding constraint (VA stores the problem densely), so that is the one
    measurement that must not read low.

    ru_maxrss is bytes on macOS and kilobytes on Linux. peak_wset is used on
    Windows, where getrusage does not exist.
    """
    try:
        import resource  # type: ignore

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if usage > 0:
            return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0
    except Exception:
        pass

    if psutil is not None:
        try:
            proc = psutil.Process()
            info = proc.memory_full_info() if hasattr(proc, "memory_full_info") else None
            peak = getattr(info, "peak_wset", None) if info is not None else None
            if peak:
                return float(peak) / (1024.0 * 1024.0)
        except Exception:
            pass

    # No peak source available; current RSS is the best remaining estimate.
    return current_rss_mb()


def peak_or_current_rss_mb() -> float:
    """Backwards-compatible alias. Returns the true peak where available."""
    return peak_rss_mb()


def memory_report_mb() -> tuple[float, float]:
    """(peak_mb, current_mb). Peak is the true process high-water mark.

    tracemalloc only sees Python allocations, so it misses numpy/pandas buffers
    and anything the VA extension allocates; RSS covers all of it. The peak
    therefore takes the max of the two, and current stays current.
    """
    tracemalloc_current_mb = 0.0
    tracemalloc_peak_mb = 0.0
    try:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc_current_mb = current / (1024.0 * 1024.0)
        tracemalloc_peak_mb = peak / (1024.0 * 1024.0)
    except Exception:
        pass

    current_mb = max(tracemalloc_current_mb, current_rss_mb())
    peak_mb = max(tracemalloc_peak_mb, peak_rss_mb(), current_mb)
    return peak_mb, current_mb


def memory_breakdown_mb() -> dict[str, float]:
    """Every memory figure separately, so a run can be diagnosed after the fact."""
    tm_current = tm_peak = 0.0
    try:
        current, peak = tracemalloc.get_traced_memory()
        tm_current = current / (1024.0 * 1024.0)
        tm_peak = peak / (1024.0 * 1024.0)
    except Exception:
        pass
    return {
        "rss_current_mb": float(current_rss_mb()),
        "rss_peak_mb": float(peak_rss_mb()),
        "tracemalloc_current_mb": float(tm_current),
        "tracemalloc_peak_mb": float(tm_peak),
    }


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:,.1f} {unit}"
        x /= 1024.0
    return f"{x:,.1f} TiB"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in ID_COLUMNS or col.endswith("_id"):
            out[col] = out[col].astype(str)
    return out


def read_csv_required(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return normalize_id_columns(pd.read_csv(path))


def require_columns(df: pd.DataFrame, filename: str, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")


def load_problem_data(
    dataset_dir: Path | str,
    *,
    max_service_miles_override: float | None,
    penalty_start_miles_override: float | None,
    top_hubs_per_zip: int | None,
    max_parts_total: int | None,
) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    missing_files = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Dataset directory is missing required files: {missing_files}")

    params = read_csv_required(root, "parameters.csv")
    if params.empty:
        raise ValueError(f"Empty parameters.csv in {root}")
    row = params.iloc[0].to_dict()
    missing_scalars = [c for c in SCALAR_COLUMNS if c not in row]
    if missing_scalars:
        raise ValueError(f"parameters.csv missing columns: {missing_scalars}")

    scalar = {c: float(row[c]) for c in SCALAR_COLUMNS}
    scalar["L"] = int(float(scalar["L"]))
    if max_service_miles_override is not None and float(max_service_miles_override) > 0:
        scalar["max_service_miles"] = float(max_service_miles_override)
    if penalty_start_miles_override is not None and float(penalty_start_miles_override) >= 0:
        scalar["penalty_start_miles"] = float(penalty_start_miles_override)

    hubs = read_csv_required(root, "hubs.csv")
    parts = read_csv_required(root, "parts.csv")
    zips = read_csv_required(root, "zips.csv")
    demand = read_csv_required(root, "demand.csv")
    distances = read_csv_required(root, "distances.csv")

    require_columns(hubs, "hubs.csv", ["hub_id", "T_j"])
    require_columns(parts, "parts.csv", ["part_id"])
    require_columns(zips, "zips.csv", ["zip_id"])
    require_columns(demand, "demand.csv", ["zip_id", "part_id", "Q_ik", "b_ik"])
    require_columns(distances, "distances.csv", ["zip_id", "hub_id", "d_ij"])

    price_col = "P_k" if "P_k" in parts.columns else "price"
    if price_col not in parts.columns:
        raise ValueError("parts.csv must contain P_k or price")
    if price_col != "P_k":
        parts = parts.rename(columns={price_col: "P_k"})

    hubs = hubs.copy()
    parts = parts.copy()
    zips = zips.copy()
    demand = demand.copy()
    distances = distances.copy()

    hubs["T_j"] = pd.to_numeric(hubs["T_j"], errors="raise").astype(int)
    if "B_j" in hubs.columns:
        hubs["B_j"] = pd.to_numeric(hubs["B_j"], errors="coerce").fillna(0.0)
    else:
        hubs["B_j"] = 0.0
    parts["P_k"] = pd.to_numeric(parts["P_k"], errors="raise")
    demand["Q_ik"] = pd.to_numeric(demand["Q_ik"], errors="raise")
    demand["b_ik"] = pd.to_numeric(demand["b_ik"], errors="raise")
    distances["d_ij"] = pd.to_numeric(distances["d_ij"], errors="raise")

    if hubs.duplicated("hub_id").any():
        raise ValueError("hubs.csv has duplicate hub_id values")
    if parts.duplicated("part_id").any():
        raise ValueError("parts.csv has duplicate part_id values")

    selected_parts = sorted(parts["part_id"].unique().tolist())
    if max_parts_total is not None and int(max_parts_total) > 0:
        selected_parts = selected_parts[: int(max_parts_total)]
    selected_parts_set = set(selected_parts)
    parts = parts[parts["part_id"].isin(selected_parts_set)].copy()
    demand = demand[demand["part_id"].isin(selected_parts_set)].copy()

    missing_parts = sorted(set(demand.loc[demand["Q_ik"] > 0, "part_id"]) - set(parts["part_id"]))
    if missing_parts:
        raise ValueError(f"parts.csv is missing demanded parts. Examples: {missing_parts[:10]}")
    missing_zips = sorted(set(demand.loc[demand["Q_ik"] > 0, "zip_id"]) - set(zips["zip_id"]))
    if missing_zips:
        raise ValueError(f"zips.csv is missing demanded ZIPs. Examples: {missing_zips[:10]}")

    active = demand[demand["Q_ik"] > 0][["zip_id", "part_id", "b_ik"]].copy()
    active = (
        active.groupby(["zip_id", "part_id"], as_index=False, sort=False)["b_ik"]
        .sum()
        .reset_index(drop=True)
    )
    if active.empty:
        raise ValueError("No active demand rows after filtering")

    max_service_miles = float(scalar["max_service_miles"])
    dist_f = distances[distances["d_ij"] <= max_service_miles].copy()
    if dist_f.empty:
        raise ValueError(f"No distance rows with d_ij <= max_service_miles={max_service_miles}")

    dist_f = dist_f.sort_values(["zip_id", "d_ij", "hub_id"])
    if top_hubs_per_zip is not None and int(top_hubs_per_zip) > 0:
        dist_f = dist_f.groupby("zip_id", sort=False).head(int(top_hubs_per_zip)).copy()

    zip_to_hubs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    d_map: dict[tuple[str, str], float] = {}
    for r in dist_f[["zip_id", "hub_id", "d_ij"]].itertuples(index=False):
        i, j, dij = str(r.zip_id), str(r.hub_id), float(r.d_ij)
        zip_to_hubs[i].append((j, dij))
        d_map[(i, j)] = dij
    for i in list(zip_to_hubs):
        zip_to_hubs[i].sort(key=lambda t: (t[1], t[0]))

    no_candidate = sorted(set(active["zip_id"]) - set(zip_to_hubs))
    if no_candidate:
        examples = active[active["zip_id"].isin(no_candidate)][["zip_id", "part_id"]].head(10)
        raise ValueError(
            "Some active ZIP-part pairs have no eligible hub within max_service_miles. Examples:\n"
            + examples.to_string(index=False)
        )

    active_parts = set(active["part_id"].unique().tolist())
    part_order = [p for p in selected_parts if p in active_parts]

    p_map = {str(r.part_id): float(r.P_k) for r in parts[["part_id", "P_k"]].itertuples(index=False)}
    t_map = {str(r.hub_id): int(r.T_j) for r in hubs[["hub_id", "T_j"]].itertuples(index=False)}
    b_cap_map = {str(r.hub_id): float(r.B_j) for r in hubs[["hub_id", "B_j"]].itertuples(index=False)}
    b_demand = {(str(r.zip_id), str(r.part_id)): float(r.b_ik) for r in active.itertuples(index=False)}

    baseline_path = root / "optional_baseline_part_homes.csv"
    parameter_key_path = root / "optional_parameter_key.csv"

    return {
        "dataset_dir": str(root),
        "dataset_name": root.name,
        "scalar": scalar,
        "hubs": hubs,
        "parts": parts,
        "zips": zips,
        "demand": demand,
        "distances_filtered": dist_f,
        "active": active,
        "part_order": part_order,
        "P": p_map,
        "T": t_map,
        "B_j": b_cap_map,
        "B": b_demand,
        "D": d_map,
        "zip_to_hubs": dict(zip_to_hubs),
        "J": sorted(hubs["hub_id"].unique().tolist()),
        "K": selected_parts,
        "max_service_miles": max_service_miles,
        "top_hubs_per_zip": top_hubs_per_zip,
        "max_parts_total": max_parts_total,
        "baseline_part_homes_rows": len(pd.read_csv(baseline_path)) if baseline_path.is_file() else 0,
        "parameter_key_rows": len(pd.read_csv(parameter_key_path)) if parameter_key_path.is_file() else 0,
    }


# ---------------------------------------------------------------------------
# Aligned cost basis
# ---------------------------------------------------------------------------


def assignment_cost_by_values(b: float, dij: float, scalar: dict[str, Any]) -> dict[str, float]:
    miles_after_base = max(0.0, float(dij) - float(scalar["base_miles"]))
    miles_after_penalty_start = max(0.0, float(dij) - float(scalar["penalty_start_miles"]))
    linehaul = float(scalar["lambda_1"]) * float(scalar["h_s"]) * float(b)
    distance = float(scalar["lambda_2"]) * float(scalar["h_d"]) * float(b) * miles_after_base
    penalty = float(scalar["lambda_3"]) * float(b) * miles_after_penalty_start
    return {
        "b_ik": float(b),
        "d_ij": float(dij),
        "miles_after_base": float(miles_after_base),
        "miles_after_penalty_start": float(miles_after_penalty_start),
        "linehaul_cost": float(linehaul),
        "distance_cost": float(distance),
        "distance_penalty_cost": float(penalty),
        "assignment_cost": float(linehaul + distance + penalty),
        "sla_violation": bool(float(dij) > float(scalar["penalty_start_miles"])),
    }


def assignment_cost(zip_id: str, hub_id: str, part_id: str, data: dict[str, Any]) -> dict[str, float]:
    b = float(data["B"].get((str(zip_id), str(part_id)), 0.0))
    dij = float(data["D"].get((str(zip_id), str(hub_id)), 0.0))
    return assignment_cost_by_values(b, dij, data["scalar"])


def compute_solution_cost(
    assignments: Iterable[tuple[str, str, str]],
    stocked_pairs: Iterable[tuple[str, str]],
    open_hubs: Iterable[str],
    data: dict[str, Any],
) -> dict[str, float]:
    scalar = data["scalar"]
    assignment_set = set((str(i), str(j), str(k)) for i, j, k in assignments)
    stocked_set = set((str(j), str(k)) for j, k in stocked_pairs)
    open_set = set(str(j) for j in open_hubs)

    inventory_cost = sum(float(data["P"].get(k, 0.0)) for _, k in stocked_set)
    fixed_open_cost = float(scalar["S_lim"]) * float(len(open_set))
    transfer_cost = sum((1 - int(data["T"].get(j, 0))) * float(scalar["C"]) for j, _ in stocked_set)

    by_hub: dict[str, int] = defaultdict(int)
    for j, _ in stocked_set:
        by_hub[j] += 1
    overflow_units = sum(max(0, cnt - int(scalar["L"])) for cnt in by_hub.values())
    overflow_cost = float(scalar["S_var"]) * float(overflow_units)

    transport_cost = 0.0
    for i, j, k in assignment_set:
        transport_cost += assignment_cost(i, j, k, data)["assignment_cost"]

    total = inventory_cost + fixed_open_cost + overflow_cost + transfer_cost + transport_cost
    return {
        "total_cost": float(total),
        "inventory_cost": float(inventory_cost),
        "fixed_open_hub_cost": float(fixed_open_cost),
        "overflow_storage_cost": float(overflow_cost),
        "new_hub_transfer_cost": float(transfer_cost),
        "assignment_transport_cost": float(transport_cost),
        "overflow_units": float(overflow_units),
    }


def global_audit(
    assignments: Iterable[tuple[str, str, str]],
    stocked_pairs: Iterable[tuple[str, str]],
    open_hubs: Iterable[str],
    data: dict[str, Any],
) -> dict[str, int]:
    assignment_list = [(str(i), str(j), str(k)) for i, j, k in assignments]
    stocked_set = set((str(j), str(k)) for j, k in stocked_pairs)
    open_set = set(str(j) for j in open_hubs)

    active_pairs = set(
        (str(r.zip_id), str(r.part_id))
        for r in data["active"][["zip_id", "part_id"]].itertuples(index=False)
    )
    candidate_by_zip = {i: {j for j, _ in hubs} for i, hubs in data["zip_to_hubs"].items()}

    assignment_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    invalid_hub = 0
    max_service_violations = 0
    sla_distance_violations = 0
    for i, j, k in assignment_list:
        assignment_map[(i, k)].append(j)
        if j not in candidate_by_zip.get(i, set()):
            invalid_hub += 1
        dij = float(data["D"].get((i, j), math.inf))
        if not math.isfinite(dij) or dij > float(data["scalar"]["max_service_miles"]):
            max_service_violations += 1
        if math.isfinite(dij) and dij > float(data["scalar"]["penalty_start_miles"]):
            sla_distance_violations += 1

    missing = 0
    multiple = 0
    for pair in active_pairs:
        count = len(assignment_map.get(pair, []))
        if count == 0:
            missing += 1
        elif count != 1:
            multiple += 1
    extra = sum(1 for pair in assignment_map if pair not in active_pairs)

    c1 = missing + multiple + invalid_hub + extra
    c2 = sum(1 for _, j, k in assignment_list if (j, k) not in stocked_set)
    c3 = sum(1 for j, _ in stocked_set if j not in open_set)

    by_hub: dict[str, int] = defaultdict(int)
    for j, _ in stocked_set:
        by_hub[j] += 1
    l_cap = int(data["scalar"]["L"])
    c4_hubs_over_l = sum(1 for c in by_hub.values() if c > l_cap)
    c4_overflow_units = sum(max(0, c - l_cap) for c in by_hub.values())

    structural = c1 + c2 + c3
    return {
        "c1_assignment_violations": int(c1),
        "c1_missing_assignments": int(missing),
        "c1_multiple_assignments": int(multiple),
        "c1_invalid_hub_assignments": int(invalid_hub),
        "c1_extra_assignments": int(extra),
        "c2_assignment_without_stock": int(c2),
        "c3_stock_without_open_hub": int(c3),
        "c4_hubs_over_L": int(c4_hubs_over_l),
        "c4_total_overflow_units": int(c4_overflow_units),
        "max_service_distance_violations": int(max_service_violations),
        "sla_distance_violations": int(sla_distance_violations),
        "total_structural_violations": int(structural),
    }


# ---------------------------------------------------------------------------
# Solution output schemas (identical to the OpenJij runner's)
# ---------------------------------------------------------------------------


def assignment_rows_dataframe(
    assignments: Iterable[tuple[str, str, str]],
    data: dict[str, Any],
    assignment_sources: dict[tuple[str, str, str], str] | None = None,
) -> pd.DataFrame:
    rows = []
    sources = assignment_sources or {}
    for i, j, k in sorted((str(i), str(j), str(k)) for i, j, k in assignments):
        c = assignment_cost(i, j, k, data)
        rows.append(
            {
                "zip_id": i,
                "hub_id": j,
                "part_id": k,
                "b_ik": c["b_ik"],
                "d_ij": c["d_ij"],
                "miles_after_base": c["miles_after_base"],
                "miles_after_penalty_start": c["miles_after_penalty_start"],
                "sla_violation": int(bool(c["sla_violation"])),
                "linehaul_cost": c["linehaul_cost"],
                "distance_cost": c["distance_cost"],
                "distance_penalty_cost": c["distance_penalty_cost"],
                "assignment_cost": c["assignment_cost"],
                "source": sources.get((i, j, k), "raw_or_solver"),
            }
        )
    return pd.DataFrame(rows)


def stocked_pairs_dataframe(stocked_pairs: Iterable[tuple[str, str]], data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for j, k in sorted((str(j), str(k)) for j, k in stocked_pairs):
        rows.append(
            {
                "hub_id": j,
                "part_id": k,
                "part_cost_P_k": float(data["P"].get(k, 0.0)),
                "hub_existing_T_j": int(data["T"].get(j, 0)),
                "new_hub_transfer_cost": (1 - int(data["T"].get(j, 0))) * float(data["scalar"]["C"]),
            }
        )
    return pd.DataFrame(rows)


def hub_status_dataframe(
    open_hubs: Iterable[str],
    stocked_pairs: Iterable[tuple[str, str]],
    assignments: Iterable[tuple[str, str, str]],
    data: dict[str, Any],
) -> pd.DataFrame:
    open_set = set(str(j) for j in open_hubs)
    stock_count: dict[str, int] = defaultdict(int)
    assign_count: dict[str, int] = defaultdict(int)
    for j, _ in stocked_pairs:
        stock_count[str(j)] += 1
    for _, j, _ in assignments:
        assign_count[str(j)] += 1

    rows = []
    for j in data["J"]:
        opened = j in open_set
        existing = int(data["T"].get(j, 0))
        rows.append(
            {
                "hub_id": j,
                "status": "OPEN" if opened else "CLOSED",
                "opened": int(opened),
                "closed": int(not opened),
                "existing_T_j": existing,
                "opened_new_hub": int(opened and existing == 0),
                "stocked_part_count": int(stock_count.get(j, 0)),
                "assignment_count": int(assign_count.get(j, 0)),
                "fixed_open_cost_if_open": float(data["scalar"]["S_lim"]) if opened else 0.0,
            }
        )
    return pd.DataFrame(rows)


def final_results_block(summary: dict[str, Any], title: str) -> str:
    sol = summary["final_solution"]
    cost = sol["cost"]
    audit = sol["audit"]
    rt = summary["runtime"]
    lines = [
        "=" * 76,
        title,
        "=" * 76,
        f"  total cost:                 {money(cost['total_cost'])}",
        f"  inventory cost:             {money(cost['inventory_cost'])}",
        f"  fixed open-hub cost:        {money(cost['fixed_open_hub_cost'])}",
        f"  overflow storage cost:      {money(cost['overflow_storage_cost'])}",
        f"  new-hub transfer cost:      {money(cost['new_hub_transfer_cost'])}",
        f"  assignment transport cost:  {money(cost['assignment_transport_cost'])}",
        f"  open hubs:                  {sol['open_hubs_count']:,}",
        f"  closed hubs:                {sol['closed_hubs_count']:,}",
        f"  stocked hub-part pairs:     {sol['stocked_pairs_count']:,}",
        f"  hub-zip-part pairings:      {sol['assignments_count']:,}",
        f"  structural violations:      {audit['total_structural_violations']:,}",
        f"  SLA distance violations:    {audit['sla_distance_violations']:,}",
        f"  wall time:                  {rt['wall_seconds']:,.2f}s",
        f"  peak/current memory:        {rt['peak_memory_mb']:,.1f}/{rt['current_memory_mb']:,.1f} MB",
        f"  output folder:              {summary['output_dir']}",
        "=" * 76,
    ]
    return "\n".join(lines)


def write_solution_outputs(
    run_dir: Path,
    *,
    solver_name: str,
    data: dict[str, Any],
    assignments: list[tuple[str, str, str]],
    stocked_pairs: list[tuple[str, str]],
    open_hubs: list[str],
    runtime: dict[str, Any],
    extra: dict[str, Any] | None = None,
    assignment_sources: dict[tuple[str, str, str], str] | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    closed_count = int(len(data["J"]) - len(set(open_hubs)))
    cost = compute_solution_cost(assignments, stocked_pairs, open_hubs, data)
    audit = global_audit(assignments, stocked_pairs, open_hubs, data)

    assignment_rows_dataframe(assignments, data, assignment_sources).to_csv(
        run_dir / "hub_zip_part_pairings.csv", index=False
    )
    stocked_pairs_dataframe(stocked_pairs, data).to_csv(run_dir / "stocked_hub_part_pairs.csv", index=False)
    stocked_pairs_dataframe(stocked_pairs, data).to_csv(run_dir / "stocked_pairs.csv", index=False)
    hub_status_dataframe(open_hubs, stocked_pairs, assignments, data).to_csv(
        run_dir / "hubs_open_closed.csv", index=False
    )
    pd.DataFrame({"hub_id": sorted(set(open_hubs))}).to_csv(run_dir / "open_hubs.csv", index=False)
    pd.DataFrame({"hub_id": sorted(set(data["J"]) - set(open_hubs))}).to_csv(
        run_dir / "closed_hubs.csv", index=False
    )

    summary = {
        "solver": solver_name,
        "dataset": {
            "dataset_name": data["dataset_name"],
            "dataset_dir": data["dataset_dir"],
            "hubs": int(len(data["J"])),
            "parts": int(len(data["K"])),
            "zips": int(len(data["zips"])),
            "active_demand_pairs": int(len(data["active"])),
            "max_service_miles": float(data["scalar"]["max_service_miles"]),
            "base_miles": float(data["scalar"]["base_miles"]),
            "penalty_start_miles": float(data["scalar"]["penalty_start_miles"]),
            "top_hubs_per_zip": data["top_hubs_per_zip"],
            "max_parts_total": data["max_parts_total"],
        },
        "final_solution": {
            "open_hubs_count": int(len(set(open_hubs))),
            "closed_hubs_count": closed_count,
            "stocked_pairs_count": int(len(set(stocked_pairs))),
            "assignments_count": int(len(set(assignments))),
            "cost": cost,
            "audit": audit,
        },
        "runtime": runtime,
        "output_dir": str(run_dir),
        "extra": extra or {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.txt").write_text(
        final_results_block(summary, f"{solver_name.upper()} FINAL RESULTS") + "\n", encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def build_batches(
    active: pd.DataFrame,
    part_order: list[str],
    zip_to_hubs: dict[str, list[tuple[str, float]]],
    *,
    part_batch_size: int,
    max_z_vars_per_batch: int,
) -> list[BatchSpec]:
    if part_batch_size <= 0:
        raise ValueError("--part-batch-size must be > 0")
    if max_z_vars_per_batch <= 0:
        raise ValueError("--max-z-vars-per-batch must be > 0")

    df = active.reset_index(drop=True).copy()
    row_z = df["zip_id"].map(lambda z: len(zip_to_hubs.get(str(z), ())))
    if (row_z <= 0).any():
        bad = df.loc[row_z <= 0, ["zip_id", "part_id"]].head(5).to_dict("records")
        raise ValueError(f"Active demand rows without candidate hubs: {bad}")
    if (row_z > max_z_vars_per_batch).any():
        bad = df.loc[row_z > max_z_vars_per_batch, ["zip_id", "part_id"]].head(5).to_dict("records")
        raise ValueError(
            "A single demand row has more candidate hubs than --max-z-vars-per-batch. "
            f"Examples: {bad}. Increase the cap or reduce --top-hubs-per-zip."
        )

    rows_by_part: dict[str, list[int]] = defaultdict(list)
    for idx, part_id in enumerate(df["part_id"].tolist()):
        rows_by_part[str(part_id)].append(idx)

    batches: list[BatchSpec] = []
    current: list[int] = []
    current_parts: set[str] = set()
    current_z = 0

    def flush(note: str = "") -> None:
        nonlocal current, current_parts, current_z
        if current:
            batches.append(BatchSpec(len(batches) + 1, list(current), int(current_z), note))
        current = []
        current_parts = set()
        current_z = 0

    for part_id in part_order:
        idxs = rows_by_part.get(str(part_id), [])
        if not idxs:
            continue
        part_z = int(row_z.iloc[idxs].sum())

        if part_z > max_z_vars_per_batch:
            flush()
            chunk: list[int] = []
            chunk_z = 0
            for idx in idxs:
                rz = int(row_z.iloc[idx])
                if chunk and chunk_z + rz > max_z_vars_per_batch:
                    batches.append(
                        BatchSpec(len(batches) + 1, list(chunk), int(chunk_z), f"split_large_part:{part_id}")
                    )
                    chunk = []
                    chunk_z = 0
                chunk.append(idx)
                chunk_z += rz
            if chunk:
                batches.append(
                    BatchSpec(len(batches) + 1, list(chunk), int(chunk_z), f"split_large_part:{part_id}")
                )
            continue

        if current and (current_z + part_z > max_z_vars_per_batch or len(current_parts) >= part_batch_size):
            flush()
        current.extend(idxs)
        current_parts.add(str(part_id))
        current_z += part_z

    flush()
    return batches


# ---------------------------------------------------------------------------
# Penalty weighting
# ---------------------------------------------------------------------------


def estimate_objective_scale(batch_df: pd.DataFrame, data: dict[str, Any]) -> float:
    if batch_df.empty:
        return 1.0
    scalar = data["scalar"]
    part_ids = batch_df["part_id"].unique().tolist()
    max_part_cost = max((float(data["P"].get(k, 0.0)) for k in part_ids), default=0.0)
    max_b = float(batch_df["b_ik"].max()) if "b_ik" in batch_df.columns else 0.0
    max_d = max((float(v) for v in data["D"].values()), default=0.0)
    max_transport = (
        float(scalar["lambda_1"]) * float(scalar["h_s"]) * max_b
        + float(scalar["lambda_2"]) * float(scalar["h_d"]) * max_b * max(0.0, max_d - float(scalar["base_miles"]))
        + float(scalar["lambda_3"]) * max_b * max(0.0, max_d - float(scalar["penalty_start_miles"]))
    )
    return max(1.0, max_part_cost + float(scalar["C"]) + float(scalar["S_lim"]) + max_transport)


def penalty_weights(
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    multipliers: dict[str, float] | None = None,
    return_diagnostics: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, dict[str, Any]]]:
    scale = 1.0 if getattr(args, "disable_objective_scale", False) else estimate_objective_scale(batch_df, data)
    out: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for c in ("c1", "c2", "c3", "c4"):
        override_min = float(getattr(args, f"min_penalty_{c}", -1.0))
        override_mult = float(getattr(args, f"constraint_multiplier_{c}", -1.0))
        min_pen = override_min if override_min >= 0 else float(args.min_penalty)
        mult = override_mult if override_mult >= 0 else float(args.constraint_multiplier)
        scaled_value = float(scale) * float(mult)
        if args.penalty_mode == "fixed":
            out[c] = float(min_pen)
        else:
            out[c] = max(float(min_pen), scaled_value)

        if return_diagnostics:
            # "floor" means the min_penalty floor won; "scaled" means scale*mult won.
            # In fixed mode the floor is used unconditionally.
            if args.penalty_mode == "fixed" or float(min_pen) >= scaled_value:
                binding_branch = "floor"
            else:
                binding_branch = "scaled"
            adaptive_mult = float(multipliers[c]) if (multipliers is not None and c in multipliers) else 1.0
            diagnostics[c] = {
                "objective_scale": float(scale),
                "min_pen": float(min_pen),
                "mult": float(mult),
                "scaled_value": float(scaled_value),
                "chosen_value": float(out[c]),
                "binding_branch": binding_branch,
                "adaptive_mult": float(adaptive_mult),
                "final_value": float(out[c]) * float(adaptive_mult),
            }

    # Adaptive multipliers (within-batch adaptive penalty). When None, the
    # returned penalties are bit-identical to the static behavior.
    if multipliers is not None:
        for c in ("c1", "c2", "c3", "c4"):
            if c in multipliers:
                out[c] = out[c] * float(multipliers[c])

    if return_diagnostics:
        return out, diagnostics
    return out


# ---------------------------------------------------------------------------
# PyQUBO formulation
# ---------------------------------------------------------------------------
#
# Decision variables
#   Z_ijk  1 if ZIP i is served part k from hub j
#   Y_jk   1 if hub j stocks part k
#   X_j    1 if hub j is open
#
# Objective (minimize)
#   sum_ijk transport(i,j,k) Z_ijk
# + sum_jk  (P_k + (1 - T_j) C) Y_jk
# + sum_j   S_lim X_j
#
# Constraints, as quadratic penalties with Placeholder weights
#   C1  sum_j Z_ijk == 1     ->  lam_c1 * (sum_j Z_ijk - 1)^2
#   C2  Z_ijk <= Y_jk        ->  lam_c2 * Z_ijk (1 - Y_jk)
#   C3  Y_jk  <= X_j         ->  lam_c3 * Y_jk  (1 - X_j)
#   C4  sum_k Y_jk <= L      ->  NOT hard-encoded; see c4_note. Pure QUBO needs
#                                bounded slack variables for this inequality,
#                                and the final cost already prices overflow.
#
# The penalties are Placeholders rather than literals so a batch compiles ONCE
# and the adaptive loop re-feeds grown weights through to_qubo(feed_dict=...).
# Variable counts are invariant under the weights, so the VA ceiling check made
# in preflight stays valid for every adaptive iteration.

PLACEHOLDER_C1 = "lam_c1"
PLACEHOLDER_C2 = "lam_c2"
PLACEHOLDER_C3 = "lam_c3"


def var_x(hub_id: str) -> str:
    return f"X|{hub_id}"


def var_y(hub_id: str, part_id: str) -> str:
    return f"Y|{hub_id}|{part_id}"


def var_z(zip_id: str, hub_id: str, part_id: str) -> str:
    return f"Z|{zip_id}|{hub_id}|{part_id}"


@dataclass
class BatchModel:
    """A compiled pyqubo model for one batch, plus its variable index.

    Reused across every adaptive-penalty iteration of the batch: only the
    Placeholder feed changes, so the expensive compile happens once.
    """

    model: Any
    z_name: dict[tuple[str, str, str], str]
    y_name: dict[tuple[str, str], str]
    x_name: dict[str, str]
    demand_groups: dict[tuple[str, str], list[tuple[str, str]]]
    c4_note: str
    express_seconds: float
    compile_seconds: float

    @property
    def num_z(self) -> int:
        return len(self.z_name)

    @property
    def num_y(self) -> int:
        return len(self.y_name)

    @property
    def num_x(self) -> int:
        return len(self.x_name)

    @property
    def total_vars(self) -> int:
        return self.num_z + self.num_y + self.num_x


def build_batch_model(batch_df: pd.DataFrame, data: dict[str, Any], args: argparse.Namespace) -> BatchModel:
    """Formulate one batch natively in pyqubo and compile it."""
    scalar = data["scalar"]
    zip_to_hubs = data["zip_to_hubs"]

    lam_c1 = Placeholder(PLACEHOLDER_C1)
    lam_c2 = Placeholder(PLACEHOLDER_C2)
    lam_c3 = Placeholder(PLACEHOLDER_C3)

    z_name: dict[tuple[str, str, str], str] = {}
    y_name: dict[tuple[str, str], str] = {}
    x_name: dict[str, str] = {}
    demand_groups: dict[tuple[str, str], list[tuple[str, str]]] = {}

    y_var: dict[tuple[str, str], Any] = {}
    x_var: dict[str, Any] = {}
    terms: list[Any] = []

    t_express = time.perf_counter()

    for r in batch_df[["zip_id", "part_id", "b_ik"]].itertuples(index=False):
        i = str(r.zip_id)
        k = str(r.part_id)
        b_val = float(r.b_ik)
        group_entries: list[tuple[str, str]] = []
        group_z: list[Any] = []

        for j, dij in zip_to_hubs[i]:
            zn = var_z(i, j, k)
            yn = var_y(j, k)
            xn = var_x(j)
            z_name[(i, j, k)] = zn
            y_name[(j, k)] = yn
            x_name[j] = xn

            z = Binary(zn)
            y = y_var.get((j, k))
            if y is None:
                y = Binary(yn)
                y_var[(j, k)] = y
            if j not in x_var:
                x_var[j] = Binary(xn)

            group_entries.append((j, zn))
            group_z.append(z)

            # Objective: transport cost of serving (i, k) from j.
            transport = assignment_cost_by_values(b_val, float(dij), scalar)["assignment_cost"]
            # C2: Z_ijk <= Y_jk.
            terms.append(transport * z + lam_c2 * z * (1 - y))

        demand_groups[(i, k)] = group_entries
        # C1: exactly one hub serves this active (zip, part).
        terms.append(lam_c1 * (sum(group_z) - 1) ** 2)

    for (j, k), y in y_var.items():
        # Objective: part cost, plus the transfer charge when j is not an existing hub.
        stock_cost = float(data["P"].get(k, 0.0)) + (1 - int(data["T"].get(j, 0))) * float(scalar["C"])
        # C3: Y_jk <= X_j.
        terms.append(stock_cost * y + lam_c3 * y * (1 - x_var[j]))

    for j, x in x_var.items():
        # Objective: fixed cost of opening hub j.
        terms.append(float(scalar["S_lim"]) * x)

    # Tier 3.8: approximate X<=sum(Y) penalty. Pure QUBO can't express OR(Y),
    # so we add lam_xy on each X-diagonal and refund lam_xy/n_k per (X,Y) pair.
    # When X=1 with all parts stocked, the refund cancels the bias; with no Y
    # stocked, the full lam_xy penalizes empty hubs. Default factor is 0.
    x_empty_factor = float(getattr(args, "x_empty_penalty_factor", 0.0))
    if x_empty_factor > 0:
        lam_xy = float(scalar["S_lim"]) * x_empty_factor
        parts_per_hub: dict[str, list[str]] = defaultdict(list)
        for (j, k) in y_var:
            parts_per_hub[j].append(k)
        for j, x in x_var.items():
            parts = parts_per_hub.get(j, [])
            if not parts:
                continue
            per_pair_refund = lam_xy / float(len(parts))
            terms.append(lam_xy * x - per_pair_refund * x * sum(y_var[(j, k)] for k in parts))

    # Tier 3.9: linear S_var-style overflow penalty per stocked Y. This is a
    # rough proxy because true overflow is max(0, sum(Y) - L), which needs
    # slack vars to encode in QUBO. Default factor is 0 (off); inactive on
    # instances_low since L=50000 dwarfs the parts count.
    y_overflow_factor = float(getattr(args, "y_overflow_penalty_factor", 0.0))
    if y_overflow_factor > 0:
        s_var_coeff = float(scalar["S_var"]) * y_overflow_factor
        for y in y_var.values():
            terms.append(s_var_coeff * y)

    c4_note = "inactive_or_soft_cost_only"
    max_stock_per_hub = max((sum(1 for jj, _ in y_name if jj == j) for j in x_name), default=0)
    if max_stock_per_hub > int(scalar["L"]):
        c4_note = "capacity_can_bind; QUBO does not hard-encode C4, final cost includes overflow"
        if args.c4_mode == "on":
            raise NotImplementedError(
                "C4 can bind for this batch. This runner does not use the old incorrect equality slack "
                "encoding. Use --c4-mode auto/off or implement a bounded inequality slack formulation."
            )

    H = sum(terms)
    express_seconds = time.perf_counter() - t_express

    t_compile = time.perf_counter()
    model = H.compile()
    compile_seconds = time.perf_counter() - t_compile

    del terms, H, y_var, x_var
    return BatchModel(
        model=model,
        z_name=z_name,
        y_name=y_name,
        x_name=x_name,
        demand_groups=demand_groups,
        c4_note=c4_note,
        express_seconds=float(express_seconds),
        compile_seconds=float(compile_seconds),
    )


def qubo_from_model(
    batch_model: BatchModel, penalties: dict[str, float]
) -> tuple[dict[tuple[str, str], float], float]:
    """Feed penalty Placeholders and emit the raw QUBO dict plus its offset.

    The offset is the C1 constant (lam_c1 per active demand row in the batch).
    Coefficients that cancel to ~0 are dropped so the interaction counts match
    the hand-rolled construction this replaced.
    """
    feed = {
        PLACEHOLDER_C1: float(penalties["c1"]),
        PLACEHOLDER_C2: float(penalties["c2"]),
        PLACEHOLDER_C3: float(penalties["c3"]),
    }
    raw_q, offset = batch_model.model.to_qubo(feed_dict=feed)
    Q = {
        (str(u), str(v)): float(coeff)
        for (u, v), coeff in raw_q.items()
        if abs(float(coeff)) > QUBO_ZERO_TOLERANCE
    }
    return Q, float(offset)


def build_qubo_for_batch(
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    multipliers: dict[str, float] | None = None,
    batch_model: BatchModel | None = None,
) -> dict[str, Any]:
    """Compile (or reuse) the batch model and produce this iteration's QUBO."""
    bm = batch_model if batch_model is not None else build_batch_model(batch_df, data, args)
    penalties = penalty_weights(batch_df, data, args, multipliers=multipliers)
    Q, offset = qubo_from_model(bm, penalties)
    return {
        "Q": Q,
        "offset": float(offset),
        "batch_model": bm,
        "z_name": bm.z_name,
        "y_name": bm.y_name,
        "x_name": bm.x_name,
        "demand_groups": bm.demand_groups,
        "penalties": penalties,
        "c4_note": bm.c4_note,
    }


# ---------------------------------------------------------------------------
# Sample evaluation
# ---------------------------------------------------------------------------


def sample_is_one(value: Any) -> bool:
    try:
        return int(round(float(value))) == 1
    except Exception:
        return bool(value)


def evaluate_sample(
    sample: dict[str, int], energy: float, qubo_meta: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    demand_groups = qubo_meta["demand_groups"]
    z_name = qubo_meta["z_name"]
    y_name = qubo_meta["y_name"]
    x_name = qubo_meta["x_name"]

    c1 = 0
    for entries in demand_groups.values():
        selected = sum(1 for _, zn in entries if sample_is_one(sample.get(zn, 0)))
        if selected != 1:
            c1 += 1

    c2 = 0
    for (i, j, k), zn in z_name.items():
        if sample_is_one(sample.get(zn, 0)) and not sample_is_one(sample.get(y_name[(j, k)], 0)):
            c2 += 1

    c3 = 0
    for (j, k), yn in y_name.items():
        if sample_is_one(sample.get(yn, 0)) and not sample_is_one(sample.get(x_name[j], 0)):
            c3 += 1

    selected_assignments = sorted(
        [(i, j, k) for (i, j, k), zn in z_name.items() if sample_is_one(sample.get(zn, 0))]
    )
    stocked_pairs = sorted([(j, k) for (j, k), yn in y_name.items() if sample_is_one(sample.get(yn, 0))])
    open_hubs = sorted([j for j, xn in x_name.items() if sample_is_one(sample.get(xn, 0))])
    cost = compute_solution_cost(selected_assignments, stocked_pairs, open_hubs, data)["total_cost"]

    return {
        "sample": sample,
        "energy": float(energy),
        "cost": float(cost),
        "c1": int(c1),
        "c2": int(c2),
        "c3": int(c3),
        "c4": 0,
        "total_violations": int(c1 + c2 + c3),
        "assignments": selected_assignments,
        "stocked_pairs": stocked_pairs,
        "open_hubs": open_hubs,
    }


def suggested_num_reads(num_z: int, args: argparse.Namespace) -> int:
    if num_z <= 0:
        return max(1, int(args.num_reads))
    scale = max(1.0, math.sqrt(float(num_z) / 5000.0))
    return max(1, int(math.ceil(float(args.num_reads) * scale)))


# ---------------------------------------------------------------------------
# Result aggregation and post-processing
# ---------------------------------------------------------------------------


def batch_summary_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "batch_id": r.batch_id,
                "num_rows": r.num_rows,
                "num_parts": r.num_parts,
                "num_z": r.num_z,
                "num_y": r.num_y,
                "num_x": r.num_x,
                "interactions": r.interactions,
                "build_seconds": r.build_seconds,
                "sample_seconds": r.sample_seconds,
                "eval_seconds": r.eval_seconds,
                "total_seconds": r.total_seconds,
                "stage_used": r.stage_used,
                "suggested_reads": r.suggested_reads,
                "energy": r.energy,
                "reconstructed_cost": r.reconstructed_cost,
                "c1_violations": r.c1_violations,
                "c2_violations": r.c2_violations,
                "c3_violations": r.c3_violations,
                "c4_violations": r.c4_violations,
                "penalty_c1": r.penalty_c1,
                "penalty_c2": r.penalty_c2,
                "penalty_c3": r.penalty_c3,
                "open_hubs_count": len(r.open_hubs),
                "stocked_pairs_count": len(r.stocked_pairs),
                "assignments_count": len(r.assignments),
            }
        )
    return pd.DataFrame(rows)


def batch_adaptive_summary_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        mult = r.final_penalty_multipliers
        rows.append(
            {
                "batch_id": r.batch_id,
                "adaptive_iterations_used": int(r.adaptive_iterations_used),
                "adaptive_was_feasible": bool(r.adaptive_was_feasible),
                "adaptive_exit_reason": str(r.adaptive_exit_reason),
                "final_mult_c1": float(mult.get("c1", 1.0)),
                "final_mult_c2": float(mult.get("c2", 1.0)),
                "final_mult_c3": float(mult.get("c3", 1.0)),
                "final_mult_c4": float(mult.get("c4", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def adaptive_iteration_log_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    """Flatten every batch's per-iteration adaptive log into one DataFrame."""
    return pd.DataFrame([row for r in results for row in r.adaptive_iteration_log])


def aggregate_raw_results(results: list[BatchResult]) -> dict[str, Any]:
    return {
        "open_hubs": sorted({hub for r in results for hub in r.open_hubs}),
        "stocked_pairs": sorted({pair for r in results for pair in r.stocked_pairs}),
        "assignments": sorted({asg for r in results for asg in r.assignments}),
    }


def incremental_assignment_score(
    zip_id: str,
    hub_id: str,
    part_id: str,
    data: dict[str, Any],
    current_open: set[str],
    current_stocked: set[tuple[str, str]],
) -> float:
    scalar = data["scalar"]
    score = assignment_cost(zip_id, hub_id, part_id, data)["assignment_cost"]
    if (hub_id, part_id) not in current_stocked:
        score += float(data["P"].get(part_id, 0.0))
        score += (1 - int(data["T"].get(hub_id, 0))) * float(scalar["C"])
    if hub_id not in current_open:
        score += float(scalar["S_lim"])
    return float(score)


def hub_prune_pass(
    assignments: list[tuple[str, str, str]],
    stocked_pairs: list[tuple[str, str]],
    open_hubs: list[str],
    data: dict[str, Any],
    *,
    max_iterations: int = 10,
) -> dict[str, Any]:
    """Tier 1.2: close open hubs whose assignments can be cheaply rerouted.

    For each open hub j (lowest-traffic first), compute the marginal cost of moving
    each (zip, j, part) assignment to the cheapest already-open alternative within
    max_service_miles. If S_lim plus j's transport and stocking costs exceed the
    sum of relocation costs, close j and apply the moves. Iterate until no hub
    closes in a full pass or max_iterations is hit.
    """
    scalar = data["scalar"]
    s_lim = float(scalar["S_lim"])
    transfer_C = float(scalar["C"])
    zip_to_hubs = data["zip_to_hubs"]
    P_map = data["P"]
    T_map = data["T"]

    asg = list((str(i), str(j), str(k)) for i, j, k in assignments)
    stocked: set[tuple[str, str]] = set((str(j), str(k)) for j, k in stocked_pairs)
    opened: set[str] = set(str(j) for j in open_hubs)

    closures = 0
    relocations = 0

    for _ in range(max_iterations):
        hub_assignment_indices: dict[str, list[int]] = defaultdict(list)
        for idx, (i, j, k) in enumerate(asg):
            hub_assignment_indices[j].append(idx)

        # Try low-traffic hubs first; they're cheapest to dismantle.
        candidates = sorted(opened, key=lambda j: len(hub_assignment_indices.get(j, [])))
        closed_this_round = False

        for j in candidates:
            indices = hub_assignment_indices.get(j, [])

            # Cost we save by closing j: S_lim + transport at j + stocking at j.
            transport_at_j = sum(
                assignment_cost(asg[idx][0], j, asg[idx][2], data)["assignment_cost"]
                for idx in indices
            )
            stocked_at_j = {(jj, k) for jj, k in stocked if jj == j}
            stocking_at_j = sum(
                float(P_map.get(k, 0.0)) + (1 - int(T_map.get(j, 0))) * transfer_C
                for _, k in stocked_at_j
            )
            current_cost = s_lim + transport_at_j + stocking_at_j

            # Try to relocate every assignment at j to another already-open hub.
            relocate_plan: list[tuple[int, str]] = []
            new_pairs: set[tuple[str, str]] = set()
            relocate_cost = 0.0
            feasible = True

            for idx in indices:
                i, _, k = asg[idx]
                alternatives = [
                    (jj, dij) for jj, dij in zip_to_hubs.get(i, [])
                    if jj != j and jj in opened
                ]
                if not alternatives:
                    feasible = False
                    break

                best_jj: str | None = None
                best_marginal = math.inf
                for jj, _dij in alternatives:
                    transport = assignment_cost(i, jj, k, data)["assignment_cost"]
                    stock_pair_exists = (jj, k) in stocked or (jj, k) in new_pairs
                    stock_marginal = (
                        0.0 if stock_pair_exists
                        else float(P_map.get(k, 0.0)) + (1 - int(T_map.get(jj, 0))) * transfer_C
                    )
                    total = transport + stock_marginal
                    if total < best_marginal:
                        best_marginal = total
                        best_jj = jj

                if best_jj is None or not math.isfinite(best_marginal):
                    feasible = False
                    break

                relocate_plan.append((idx, best_jj))
                relocate_cost += best_marginal
                if (best_jj, k) not in stocked:
                    new_pairs.add((best_jj, k))

            if not feasible:
                continue

            if relocate_cost < current_cost - 1e-9:
                # Apply the closure.
                for idx, new_j in relocate_plan:
                    i, _, k = asg[idx]
                    asg[idx] = (i, new_j, k)
                    stocked.add((new_j, k))
                stocked = {(jj, k) for jj, k in stocked if jj != j}
                opened.discard(j)
                closures += 1
                relocations += len(relocate_plan)
                closed_this_round = True
                break  # restart with refreshed traffic counts

        if not closed_this_round:
            break

    return {
        "assignments": sorted(asg),
        "stocked_pairs": sorted(stocked),
        "open_hubs": sorted(opened),
        "closures": int(closures),
        "relocations": int(relocations),
    }


def postprocess_qubo_solution(
    raw_assignments: list[tuple[str, str, str]],
    raw_stocked_pairs: list[tuple[str, str]],
    raw_open_hubs: list[str],
    data: dict[str, Any],
    *,
    repair_assignments: bool,
    trim_unused: bool,
    hub_prune: bool = True,
    hub_prune_max_iterations: int = 10,
) -> dict[str, Any]:
    active = data["active"]
    zip_to_hubs = data["zip_to_hubs"]

    raw_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i, j, k in raw_assignments:
        raw_by_pair[(str(i), str(k))].append(str(j))

    current_open = set(str(j) for j in raw_open_hubs)
    current_stocked = set((str(j), str(k)) for j, k in raw_stocked_pairs)

    final_assignments: list[tuple[str, str, str]] = []
    assignment_sources: dict[tuple[str, str, str], str] = {}
    missing_unrepaired = 0

    for r in active[["zip_id", "part_id"]].itertuples(index=False):
        i = str(r.zip_id)
        k = str(r.part_id)
        candidates = zip_to_hubs.get(i, [])
        candidate_hubs = {j for j, _ in candidates}
        raw_selected = [j for j in raw_by_pair.get((i, k), []) if j in candidate_hubs]

        chosen: str | None = None
        source = ""
        if len(raw_selected) == 1:
            chosen = raw_selected[0]
            source = "raw_unique"
        elif len(raw_selected) > 1:
            chosen = min(
                raw_selected,
                key=lambda j: incremental_assignment_score(i, j, k, data, current_open, current_stocked),
            )
            source = "raw_multiple_repaired"
        elif repair_assignments:
            if not candidates:
                missing_unrepaired += 1
                continue
            chosen = min(
                [j for j, _ in candidates],
                key=lambda j: incremental_assignment_score(i, j, k, data, current_open, current_stocked),
            )
            source = "missing_repaired"
        else:
            missing_unrepaired += 1
            continue

        row = (i, chosen, k)
        final_assignments.append(row)
        assignment_sources[row] = source
        current_open.add(chosen)
        current_stocked.add((chosen, k))

    if trim_unused:
        final_stocked = sorted({(j, k) for _, j, k in final_assignments})
        final_open = sorted({j for _, j, _ in final_assignments})
    else:
        final_stocked = sorted(current_stocked)
        final_open = sorted(current_open)

    prune_stats = {"closures": 0, "relocations": 0}
    if hub_prune:
        pruned = hub_prune_pass(
            sorted(final_assignments),
            final_stocked,
            final_open,
            data,
            max_iterations=int(hub_prune_max_iterations),
        )
        # Reassignments from pruning invalidate the recorded source labels for moved rows;
        # re-tag those rows so the audit CSV stays accurate.
        new_assignments = pruned["assignments"]
        original_lookup = {(i, k): j for i, j, k in final_assignments}
        relocated_sources: dict[tuple[str, str, str], str] = {}
        for i, j, k in new_assignments:
            orig_j = original_lookup.get((i, k))
            if orig_j is not None and orig_j != j:
                relocated_sources[(i, j, k)] = "hub_prune_relocated"
            else:
                relocated_sources[(i, j, k)] = assignment_sources.get((i, orig_j or j, k), "raw_or_solver")
        assignment_sources = relocated_sources
        final_assignments = new_assignments
        final_stocked = pruned["stocked_pairs"]
        final_open = pruned["open_hubs"]
        prune_stats = {"closures": pruned["closures"], "relocations": pruned["relocations"]}

    return {
        "assignments": sorted(final_assignments),
        "stocked_pairs": final_stocked,
        "open_hubs": final_open,
        "assignment_sources": assignment_sources,
        "missing_unrepaired": int(missing_unrepaired),
        "hub_prune_stats": prune_stats,
    }


# ---------------------------------------------------------------------------
# VA environment
# ---------------------------------------------------------------------------


def discover_va_installs() -> list[str]:
    """List on-disk VA python directories, newest-looking last. Diagnostics only."""
    try:
        return sorted(glob.glob(VA_CANDIDATE_GLOB))
    except Exception:
        return []


def visible_ve_devices() -> list[str]:
    """The VE card device nodes visible to THIS host. Empty means no local card."""
    devices: list[str] = []
    for pattern in VE_DEVICE_GLOBS:
        try:
            devices.extend(glob.glob(pattern))
        except Exception:
            pass
    return sorted(set(devices))


def import_vector_annealing() -> Any:
    """Import the local VectorAnnealing module, or exit with actionable guidance.

    There is no fallback and no remote path: this imports the on-prem module
    that drives the physical VE card in this node. It never constructs a service
    client and never opens a network connection.
    """
    try:
        import VectorAnnealing  # type: ignore
    except Exception as exc:  # ImportError, but VA can also fail on VE probing
        installs = discover_va_installs()
        print("=" * 76, file=sys.stderr)
        print("ERROR: could not import VectorAnnealing.", file=sys.stderr)
        print(f"  underlying error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Source the VA environment and put the VA python dir on PYTHONPATH:", file=sys.stderr)
        print("    source /opt/nec/ve/nlc/<ver>/bin/nlcvars.sh", file=sys.stderr)
        print("    export PATH=${PATH}:/opt/nec/ve/bin", file=sys.stderr)
        if installs:
            print("", file=sys.stderr)
            print(f"  VA installs found on this host ({VA_CANDIDATE_GLOB}):", file=sys.stderr)
            for path in installs:
                print(f"    export PYTHONPATH={path}:${{PYTHONPATH}}", file=sys.stderr)
        else:
            print("    export PYTHONPATH=/opt/va/<version>/libexec/VectorAnnealing/python:${PYTHONPATH}", file=sys.stderr)
            print(f"  NOTE: nothing matched {VA_CANDIDATE_GLOB} on this host.", file=sys.stderr)
            print("        You are probably not on the VE node. On ASU SOL use sfpga01n", file=sys.stderr)
            print("        (va_solve.sh pins '#SBATCH -w sfpga01n').", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"  active python:   {sys.executable}", file=sys.stderr)
        print(f"  python version:  {sys.version.splitlines()[0]}", file=sys.stderr)
        print(f"  hostname:        {socket.gethostname()}", file=sys.stderr)
        print(f"  PYTHONPATH:      {os.environ.get('PYTHONPATH', '<unset>')}", file=sys.stderr)
        print(f"  VE devices:      {visible_ve_devices() or '<none visible from this host>'}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  This solver does not fall back to any other sampler, and has no", file=sys.stderr)
        print("  cloud/service-client path. Run va_probe.py on the VE node to inspect", file=sys.stderr)
        print("  the installed module, or --dry-run to validate the batch plan with", file=sys.stderr)
        print("  no card at all.", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        raise SystemExit(2)

    version = None
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(VectorAnnealing, attr, None)
        if value is not None and not callable(value):
            version = str(value)
            break

    module_file = getattr(VectorAnnealing, "__file__", None)
    devices = visible_ve_devices()

    print("VectorAnnealing imported (local on-prem module; no service client).", flush=True)
    print(f"  module file:     {module_file or '<built-in / no __file__>'}", flush=True)
    print(f"  module version:  {version if version else '<module exposes no version attribute>'}", flush=True)
    print(f"  python:          {sys.version.splitlines()[0]}", flush=True)
    print(f"  hostname:        {socket.gethostname()}", flush=True)
    print(f"  VE_NODE_NUMBER:  {os.environ.get('VE_NODE_NUMBER', '<unset>')}", flush=True)
    print(
        f"  VE devices:      {len(devices)} card(s) visible"
        + (f" -> {', '.join(devices)}" if devices else " -> NONE"),
        flush=True,
    )
    if not devices:
        # Not fatal: device nodes can be hidden by cgroups even when usable, and
        # the sample() call itself is the real test. Say so loudly rather than
        # guessing, so a genuinely card-less node is obvious in the log.
        print(
            "  WARNING: no /dev/ve* device is visible from this host. If sampling "
            "fails, you are not on the VE node (ASU SOL: sfpga01n).",
            flush=True,
        )
    return VectorAnnealing


# ---------------------------------------------------------------------------
# Ceiling arithmetic and the batch plan
# ---------------------------------------------------------------------------


def dense_matrix_bytes(num_vars: int) -> int:
    """VA stores the problem as a dense matrix; QUBO sparsity does not help.

    100000^2 * 4 B = 40.0 GB, consistent with the manual's 48 GB requirement for
    full coupling at 100k bits; 70000^2 * 4 B = 19.6 GB matches its 24 GB tier.
    """
    return int(num_vars) * int(num_vars) * VA_DENSE_BYTES_PER_ENTRY


def build_batch_plan(
    data: dict[str, Any],
    batches: list[BatchSpec],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Compile every batch's pyqubo model once to learn its TRUE variable count.

    build_batches caps Z-vars only (--max-z-vars-per-batch); Y and X are whatever
    the batch's hub/part footprint implies. The VA ceiling applies to
    len(z_name) + len(y_name) + len(x_name), so it can only be checked after the
    model is built. Each model and QUBO is discarded immediately; only counts are
    kept. Compiling here also means a pyqubo formulation error surfaces during
    --dry-run, before any VE time is spent.
    """
    plan: list[dict[str, Any]] = []
    active = data["active"]

    for batch in batches:
        batch_df = active.iloc[batch.row_indices].reset_index(drop=True)
        t0 = time.perf_counter()
        qubo_meta = build_qubo_for_batch(batch_df, data, args)
        build_seconds = time.perf_counter() - t0
        bm: BatchModel = qubo_meta["batch_model"]

        num_z = bm.num_z
        num_y = bm.num_y
        num_x = bm.num_x
        total_vars = num_z + num_y + num_x
        qubo_vars = len({name for key in qubo_meta["Q"] for name in key})

        plan.append(
            {
                "batch_id": int(batch.batch_id),
                "num_parts": int(batch_df["part_id"].nunique()),
                "num_rows": int(len(batch_df)),
                "num_z": int(num_z),
                "num_y": int(num_y),
                "num_x": int(num_x),
                "total_vars": int(total_vars),
                "qubo_vars_in_Q": int(qubo_vars),
                "interactions": int(len(qubo_meta["Q"])),
                "qubo_offset": float(qubo_meta["offset"]),
                "dense_matrix_bytes": int(dense_matrix_bytes(total_vars)),
                "penalty_c1": float(qubo_meta["penalties"]["c1"]),
                "penalty_c2": float(qubo_meta["penalties"]["c2"]),
                "penalty_c3": float(qubo_meta["penalties"]["c3"]),
                "c4_note": str(qubo_meta["c4_note"]),
                "build_seconds": float(build_seconds),
                "pyqubo_express_seconds": float(bm.express_seconds),
                "pyqubo_compile_seconds": float(bm.compile_seconds),
                "note": str(batch.note),
            }
        )

        del qubo_meta, bm
        gc.collect()

    return plan


def print_batch_plan(plan: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ceiling = int(args.va_max_vars_per_batch)
    print("\n" + "=" * 108, flush=True)
    print("VA BATCH PLAN", flush=True)
    print("=" * 108, flush=True)
    print(f"  configured ceiling (--va-max-vars-per-batch): {ceiling:,} vars/batch", flush=True)
    print(f"  VA hard maximum (per manual):                 {VA_HARD_MAX_VARS:,} vars", flush=True)
    print(f"  dense matrix at the hard max:                 {human_bytes(dense_matrix_bytes(VA_HARD_MAX_VARS))}", flush=True)
    print(f"  dense matrix at the ceiling:                  {human_bytes(dense_matrix_bytes(ceiling))}", flush=True)
    print("  NOTE: VA stores the problem densely; QUBO sparsity does not reduce this.", flush=True)
    print("-" * 108, flush=True)
    header = (
        f"{'batch':>6}  {'parts':>6}  {'rows':>8}  {'Z':>9}  {'Y':>8}  {'X':>6}  "
        f"{'total_vars':>11}  {'dense_bytes':>16}  {'dense':>11}  {'status':>9}"
    )
    print(header, flush=True)
    print("-" * 108, flush=True)

    for row in plan:
        over = row["total_vars"] > ceiling
        status = "OVER" if over else ("OK" if row["total_vars"] <= VA_HARD_MAX_VARS else "OVER-HARD")
        print(
            f"{row['batch_id']:>6}  {row['num_parts']:>6,}  {row['num_rows']:>8,}  "
            f"{row['num_z']:>9,}  {row['num_y']:>8,}  {row['num_x']:>6,}  "
            f"{row['total_vars']:>11,}  {row['dense_matrix_bytes']:>16,}  "
            f"{human_bytes(row['dense_matrix_bytes']):>11}  {status:>9}",
            flush=True,
        )

    print("-" * 108, flush=True)
    if plan:
        worst = max(plan, key=lambda r: r["total_vars"])
        totals = sum(r["total_vars"] for r in plan)
        print(
            f"  batches: {len(plan):,} | largest batch: #{worst['batch_id']} at "
            f"{worst['total_vars']:,} vars ({human_bytes(worst['dense_matrix_bytes'])} dense) | "
            f"summed vars across batches: {totals:,}",
            flush=True,
        )
        compile_total = sum(r["pyqubo_compile_seconds"] for r in plan)
        express_total = sum(r["pyqubo_express_seconds"] for r in plan)
        print(
            f"  pyqubo: expression build {express_total:,.2f}s + compile {compile_total:,.2f}s "
            f"across {len(plan):,} batch(es)",
            flush=True,
        )
        max_offset = max(r["qubo_offset"] for r in plan)
        print(
            f"  largest to_qubo offset (the C1 constant): {max_offset:,.2f} "
            f"({'PASSED to VA' if bool(args.va_include_offset) else 'held out of VA; see --va-include-offset'})",
            flush=True,
        )
        mismatched = [r for r in plan if r["qubo_vars_in_Q"] != r["total_vars"]]
        if mismatched:
            print(
                f"  note: {len(mismatched)} batch(es) have variables that cancelled out of Q entirely; "
                "the ceiling check uses the larger len(z)+len(y)+len(x) count.",
                flush=True,
            )
    print("=" * 108, flush=True)


def check_ceiling(plan: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Abort loudly if any batch exceeds the ceiling. Never truncate or split."""
    ceiling = int(args.va_max_vars_per_batch)
    if ceiling > VA_HARD_MAX_VARS:
        raise SystemExit(
            f"--va-max-vars-per-batch={ceiling:,} exceeds VA's hard limit of "
            f"{VA_HARD_MAX_VARS:,} binary variables ({VA_MANUAL_REF})."
        )

    offenders = [r for r in plan if r["total_vars"] > ceiling]
    if not offenders:
        return

    current_max_z = int(args.max_z_vars_per_batch)
    suggestions = []
    lines = [
        "",
        "=" * 108,
        "ABORT: batch exceeds the VA variable ceiling",
        "=" * 108,
        f"  ceiling (--va-max-vars-per-batch): {ceiling:,} total variables per batch",
        f"  VA hard maximum:                   {VA_HARD_MAX_VARS:,} ({VA_MANUAL_REF})",
        "",
        "  Offending batches (total = len(z_name) + len(y_name) + len(x_name)):",
    ]
    for r in offenders:
        ratio = float(ceiling) / float(r["total_vars"])
        suggested = max(1, int(current_max_z * ratio))
        suggestions.append(suggested)
        lines.append(
            f"    batch {r['batch_id']:>4}: {r['total_vars']:,} vars "
            f"(Z={r['num_z']:,} Y={r['num_y']:,} X={r['num_x']:,}) "
            f"-> dense matrix {human_bytes(r['dense_matrix_bytes'])}, "
            f"over by {r['total_vars'] - ceiling:,}"
        )

    suggested_max_z = min(suggestions)
    lines.extend(
        [
            "",
            f"  Current --max-z-vars-per-batch is {current_max_z:,}, which caps Z only.",
            f"  Suggested retry:  --max-z-vars-per-batch {suggested_max_z:,}",
            "  (Y and X do not shrink proportionally with Z, so re-run --dry-run to confirm.)",
            "",
            "  Not truncating, splitting, or proceeding. This is a hard memory wall.",
            "=" * 108,
        ]
    )
    raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------------------
# float64 energy recompute (precision instrumentation)
# ---------------------------------------------------------------------------


def recompute_energy_float64(
    Q: dict[tuple[str, str], float], spin: dict[str, Any], offset: float = 0.0
) -> float:
    """Recompute a sample's QUBO energy on the host in double precision.

    VA computes in single precision (manual: "VA processes using single
    precision floating point arithmetic"). This is the fp64 reference value.
    math.fsum keeps the summation exact so the reported difference is
    attributable to VA, not to our accumulation order. `offset` must be the same
    constant handed to VectorAnnealing.model(), or the audit compares two
    different Hamiltonians.
    """
    on = {name for name, value in spin.items() if sample_is_one(value)}
    terms = [float(c) for (u, v), c in Q.items() if u in on and v in on]
    return math.fsum(terms) + float(offset)


def relative_difference(va_energy: float, recomputed: float) -> float:
    denom = abs(float(recomputed))
    if denom <= 1e-12:
        return float("nan")
    return abs(float(va_energy) - float(recomputed)) / denom


# ---------------------------------------------------------------------------
# VA sampling
# ---------------------------------------------------------------------------


def parse_beta_range(text: str) -> list[float] | None:
    """Parse "start,end[,nsteps]" into VA's beta_range. Empty -> VA default."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",") if p.strip()]
    if len(parts) not in (2, 3):
        raise SystemExit(
            f"--va-beta-range must be 'start,end' or 'start,end,nsteps'; got {text!r}. "
            f"VA default is [10,100,200] ({VA_MANUAL_REF})."
        )
    values: list[float] = [float(parts[0]), float(parts[1])]
    if len(parts) == 3:
        values.append(int(float(parts[2])))
    return values


def build_one_hot_list(qubo_meta: dict[str, Any]) -> list[list[str]]:
    """C1 as a VA one-hot flip group: one hub per active (zip, part) demand row.

    Per the manual, flip-option constraints "must be included in Hamiltonian's
    formulation" -- so this is declared IN ADDITION to the C1 penalty terms,
    which the pyqubo expression already carries.
    """
    groups: list[list[str]] = []
    for entries in qubo_meta["demand_groups"].values():
        names = [zn for _, zn in entries]
        if names:
            groups.append(names)
    return groups


def va_sample_once(
    VectorAnnealing: Any,
    Q: dict[tuple[str, str], float],
    args: argparse.Namespace,
    num_reads: int,
    one_hot_list: list[list[str]] | None,
    beta_range: list[float] | None,
    offset: float,
) -> tuple[list[Any], float]:
    """One VectorAnnealing.sample() call. Returns (results, wall seconds).

    `offset` is the constant pyqubo's to_qubo() returned, or 0.0 when
    --va-include-offset is not set (the default -- see the module docstring).
    """
    model_kwargs: dict[str, Any] = {}
    if one_hot_list:
        model_kwargs["onehot"] = one_hot_list

    va_model = VectorAnnealing.model(Q, float(offset), **model_kwargs)
    sampler = VectorAnnealing.sampler()

    sample_kwargs: dict[str, Any] = {
        "num_reads": int(num_reads),
        # None/1 returns a single best solution; num_results == num_reads
        # returns every annealing result, which the precision audit needs.
        "num_results": int(num_reads),
        "vector_mode": str(args.va_vector_mode),
    }
    if int(args.num_sweeps or 0) > 0:
        sample_kwargs["num_sweeps"] = int(args.num_sweeps)
    if beta_range is not None:
        sample_kwargs["beta_range"] = beta_range

    t0 = time.perf_counter()
    result = sampler.sample(va_model, **sample_kwargs)
    elapsed = time.perf_counter() - t0

    results = list(result) if not isinstance(result, list) else result
    del va_model, sampler
    return results, elapsed


def va_sample_with_retries(
    VectorAnnealing: Any,
    Q: dict[tuple[str, str], float],
    args: argparse.Namespace,
    base_reads: int,
    one_hot_list: list[list[str]] | None,
    beta_range: list[float] | None,
    label: str,
    offset: float,
) -> tuple[list[Any], float, int, int]:
    """Sample until VA returns something usable.

    Retries only when EVERY read of a call came back with a broken constraint --
    the "output result with broken constraint / INVALID" case the manual
    documents, whose advice is to re-run. Returns
    (results, seconds, attempts_used, retries). An empty result list means every
    attempt failed; the caller decides what to do about it.
    """
    seconds = 0.0
    retries = 0
    attempts_used = 0

    for attempt in range(1, int(args.va_max_retries) + 2):
        attempts_used = attempt
        results, elapsed = va_sample_once(
            VectorAnnealing, Q, args, base_reads, one_hot_list, beta_range, offset
        )
        seconds += elapsed

        if not results:
            retries += 1
            print(f"    {label} attempt {attempt}: VA returned no results ({elapsed:.2f}s) - retrying", flush=True)
            continue

        flags = [getattr(r, "constraint", None) for r in results]
        all_broken = len(flags) > 0 and all(f is False for f in flags)

        if not all_broken:
            print(
                f"    {label} attempt {attempt}: {len(results)} reads in {elapsed:.2f}s "
                f"| constraint satisfied {sum(1 for f in flags if f is True)}/{len(flags)}"
                + (" (VA reported no constraint flag)" if all(f is None for f in flags) else ""),
                flush=True,
            )
            return results, seconds, attempts_used, retries

        retries += 1
        print(
            f"    {label} attempt {attempt}: VA returned INVALID (constraint broken on all "
            f"{len(flags)} reads, {elapsed:.2f}s) - retry {attempt}/{int(args.va_max_retries)}",
            flush=True,
        )

    return [], seconds, attempts_used, retries


def solve_va_batch(
    VectorAnnealing: Any,
    batch: BatchSpec,
    data: dict[str, Any],
    args: argparse.Namespace,
    beta_range: list[float] | None,
) -> tuple[BatchResult, list[dict[str, Any]], dict[str, Any]]:
    """Solve one batch on VA. Returns (BatchResult, precision rows, va stats).

    With --adaptive-penalty-mode within-batch: re-feed the pyqubo Placeholders
    with grown multipliers for whichever constraints are still violated,
    resample, repeat until feasible or the iteration budget runs out. The batch
    is compiled once; iterations only call to_qubo() again. There is no seed
    line -- VA has no seed parameter, and the escalation never depended on one.

    This matters most with objective scale OFF: C3's penalty starts at
    --min-penalty (50,000) while S_lim is 500,000, so stocking at a closed hub
    is initially 10x cheaper than opening one. Only the escalation fixes that.
    """
    batch_start = time.time()
    batch_df = data["active"].iloc[batch.row_indices].reset_index(drop=True)
    num_rows = len(batch_df)
    num_parts = int(batch_df["part_id"].nunique())

    print(
        f"\n=== VA Batch {batch.batch_id} | {num_parts} parts | {num_rows:,} demand rows | "
        f"estimated Z={batch.estimated_z_vars:,} ===",
        flush=True,
    )
    if batch.note:
        print(f"  note: {batch.note}", flush=True)

    adaptive = args.adaptive_penalty_mode == "within-batch"
    multipliers: dict[str, float] = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    max_iterations = int(args.adaptive_penalty_iterations) if adaptive else 1
    growth = float(args.adaptive_penalty_growth)
    stagnation_patience = int(args.adaptive_penalty_stagnation_patience)

    print("  [1/3] Formulating and compiling the PyQUBO model...", flush=True)
    t0 = time.time()
    batch_model = build_batch_model(batch_df, data, args)
    qubo_meta = build_qubo_for_batch(batch_df, data, args, batch_model=batch_model)
    build_seconds = time.time() - t0
    Q = qubo_meta["Q"]
    offset = float(qubo_meta["offset"])
    num_z = batch_model.num_z
    num_y = batch_model.num_y
    num_x = batch_model.num_x
    total_vars = batch_model.total_vars
    interactions = len(Q)
    print(
        f"    built in {build_seconds:.2f}s "
        f"(pyqubo expression {batch_model.express_seconds:.2f}s + compile {batch_model.compile_seconds:.2f}s) | "
        f"Z={num_z:,} Y={num_y:,} X={num_x:,} total_vars={total_vars:,} interactions={interactions:,}",
        flush=True,
    )
    print(
        "    penalties: "
        f"C1={qubo_meta['penalties']['c1']:,.2f} "
        f"C2={qubo_meta['penalties']['c2']:,.2f} "
        f"C3={qubo_meta['penalties']['c3']:,.2f}",
        flush=True,
    )
    va_offset = offset if bool(args.va_include_offset) else 0.0
    print(
        f"    to_qubo offset (C1 constant): {offset:,.2f} -> "
        + (
            "PASSED to VectorAnnealing.model()"
            if bool(args.va_include_offset)
            else "held at 0.0 for VA (fp32 headroom + OpenJij comparability)"
        ),
        flush=True,
    )

    # Defence in depth: the preflight already checked this, but a batch must
    # never reach VectorAnnealing.model() over the ceiling.
    if total_vars > int(args.va_max_vars_per_batch):
        raise SystemExit(
            f"Batch {batch.batch_id} has {total_vars:,} variables, over the "
            f"--va-max-vars-per-batch ceiling of {int(args.va_max_vars_per_batch):,}."
        )

    one_hot_list = build_one_hot_list(qubo_meta) if bool(args.va_onehot) else None
    if one_hot_list is not None:
        print(
            f"    flip option: onehot with {len(one_hot_list):,} groups "
            "(C1 penalty terms are RETAINED in the Hamiltonian, per the manual)",
            flush=True,
        )

    base_reads = suggested_num_reads(num_z, args)
    print(
        f"  [2/3] Sampling with VectorAnnealing | reads={base_reads} "
        f"sweeps={args.num_sweeps} vector_mode={args.va_vector_mode} "
        f"beta_range={beta_range if beta_range is not None else 'VA default [10,100,200]'} | "
        f"repeats={args.va_repeats} | adaptive={'within-batch' if adaptive else 'off'}"
        + (f" (max {max_iterations} iters, growth {growth})" if adaptive else ""),
        flush=True,
    )

    best_eval: dict[str, Any] | None = None
    precision_rows: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    iteration_log: list[dict[str, Any]] = []
    sample_seconds = 0.0
    eval_seconds = 0.0
    read_index = 0
    total_sample_calls = 0
    total_retries = 0
    constraint_ok_count = 0
    constraint_total_count = 0
    iterations_used = 0
    was_feasible = False
    exit_reason = "max_iterations" if adaptive else "static_single_pass"
    best_total_violations: int | None = None
    no_improve_count = 0

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration

        # Re-feed the Placeholders at this iteration's multipliers. Variable
        # counts are invariant under the weights (only coefficients change), so
        # the preflight ceiling check stays valid across iterations, and the
        # compiled model is reused rather than rebuilt.
        if iteration > 1:
            t_refeed = time.time()
            qubo_meta = build_qubo_for_batch(
                batch_df, data, args, multipliers=multipliers, batch_model=batch_model
            )
            build_seconds += time.time() - t_refeed
            Q = qubo_meta["Q"]
            offset = float(qubo_meta["offset"])
            va_offset = offset if bool(args.va_include_offset) else 0.0

        _, diag = penalty_weights(
            batch_df, data, args, multipliers=multipliers, return_diagnostics=True
        )
        if adaptive:
            print(
                f"    adaptive iter {iteration}/{max_iterations} | "
                f"multipliers c1={multipliers['c1']:.2f} c2={multipliers['c2']:.2f} "
                f"c3={multipliers['c3']:.2f} c4={multipliers['c4']:.2f} | "
                f"penalties C1={diag['c1']['final_value']:,.0f} C2={diag['c2']['final_value']:,.0f} "
                f"C3={diag['c3']['final_value']:,.0f} | offset={offset:,.0f}",
                flush=True,
            )

        iter_evals: list[dict[str, Any]] = []
        for repeat in range(1, int(args.va_repeats) + 1):
            label = f"iter {iteration} repeat {repeat}" if adaptive else f"repeat {repeat}"
            results, seconds, attempts_used, retries = va_sample_with_retries(
                VectorAnnealing, Q, args, base_reads, one_hot_list, beta_range, label, va_offset
            )
            sample_seconds += seconds
            total_sample_calls += attempts_used
            total_retries += retries

            if not results:
                raise RuntimeError(
                    f"VA batch {batch.batch_id} ({label}): every read had a broken constraint "
                    f"after {attempts_used} attempts (--va-max-retries "
                    f"{int(args.va_max_retries)}). Failing this batch."
                )

            t_eval = time.time()
            repeat_evals: list[dict[str, Any]] = []
            for r in results:
                spin = dict(getattr(r, "spin", {}) or {})
                va_energy = float(getattr(r, "energy", 0.0))
                constraint_flag = getattr(r, "constraint", None)

                recomputed = recompute_energy_float64(Q, spin, offset=va_offset)
                precision_rows.append(
                    {
                        "batch_id": int(batch.batch_id),
                        "read_index": int(read_index),
                        "num_vars": int(total_vars),
                        "va_reported_energy": float(va_energy),
                        "recomputed_energy": float(recomputed),
                        "abs_diff": float(abs(va_energy - recomputed)),
                        "rel_diff": float(relative_difference(va_energy, recomputed)),
                        "qubo_offset": float(offset),
                        "va_offset_applied": float(va_offset),
                        "constraint_ok": constraint_flag if constraint_flag is None else bool(constraint_flag),
                    }
                )
                read_index += 1

                constraint_total_count += 1
                if constraint_flag is True:
                    constraint_ok_count += 1

                # Identical accounting to the OpenJij path: VA's own reported
                # energy is fed to evaluate_sample, exactly as OpenJij's is.
                ev = evaluate_sample(spin, va_energy, qubo_meta, data)
                repeat_evals.append(ev)
                iter_evals.append(ev)

                key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
                if best_eval is None:
                    best_eval = ev
                else:
                    old_key = (
                        best_eval["total_violations"],
                        best_eval["c1"],
                        best_eval["c2"],
                        best_eval["c3"],
                        best_eval["cost"],
                        best_eval["energy"],
                    )
                    if key < old_key:
                        best_eval = ev
            eval_seconds += time.time() - t_eval

            energies = [float(e["energy"]) for e in repeat_evals]
            costs = [float(e["cost"]) for e in repeat_evals]
            feasible_reads = sum(1 for e in repeat_evals if int(e["total_violations"]) == 0)
            repeat_best = min(
                repeat_evals,
                key=lambda e: (e["total_violations"], e["c1"], e["c2"], e["c3"], e["cost"], e["energy"]),
            )
            repeat_records.append(
                {
                    "iteration": int(iteration),
                    "repeat": int(repeat),
                    "attempts_used": int(attempts_used),
                    "reads": int(len(repeat_evals)),
                    "energy_min": min(energies),
                    "energy_median": statistics.median(energies),
                    "energy_max": max(energies),
                    "cost_min": min(costs),
                    "cost_median": statistics.median(costs),
                    "cost_max": max(costs),
                    "constraint_satisfied_reads": int(
                        sum(1 for r in results if getattr(r, "constraint", None) is True)
                    ),
                    "structurally_feasible_reads": int(feasible_reads),
                    "best_total_violations": int(repeat_best["total_violations"]),
                    "best_cost": float(repeat_best["cost"]),
                }
            )
            print(
                f"    {label} | energy min/med/max = {min(energies):,.2f} / "
                f"{statistics.median(energies):,.2f} / {max(energies):,.2f} | cost min/med/max = "
                f"{min(costs):,.2f} / {statistics.median(costs):,.2f} / {max(costs):,.2f} | "
                f"structurally feasible reads {feasible_reads}/{len(repeat_evals)}",
                flush=True,
            )

        iter_best = min(
            iter_evals,
            key=lambda e: (e["total_violations"], e["c1"], e["c2"], e["c3"], e["cost"], e["energy"]),
        )

        # The OpenJij adaptive_iteration_log_dataframe schema, so both paths'
        # logs can be concatenated. seed_used is None: VA has no seed parameter,
        # but the column is kept so the CSVs line up.
        log_row: dict[str, Any] = {
            "batch_id": int(batch.batch_id),
            "iteration": int(iteration),
            "objective_scale": float(diag["c1"]["objective_scale"]),
            "total_violations": int(iter_best["total_violations"]),
            "cost": float(iter_best["cost"]),
            "energy": float(iter_best["energy"]),
            "num_vars": int(total_vars),
            "num_interactions": int(len(Q)),
            "seed_used": None,
            "num_reads": int(base_reads),
            "qubo_offset": float(offset),
            "exit_reason": "",
        }
        for c in ("c1", "c2", "c3", "c4"):
            log_row[f"min_pen_{c}"] = float(diag[c]["min_pen"])
            log_row[f"scaled_{c}"] = float(diag[c]["scaled_value"])
            log_row[f"chosen_{c}"] = float(diag[c]["chosen_value"])
            log_row[f"binding_branch_{c}"] = str(diag[c]["binding_branch"])
            log_row[f"mult_{c}"] = float(multipliers[c])
            log_row[f"viol_{c}"] = int(iter_best.get(c, 0))
        iteration_log.append(log_row)

        if int(iter_best["total_violations"]) == 0:
            was_feasible = True
            exit_reason = "feasible"
            iteration_log[-1]["exit_reason"] = exit_reason
            if adaptive:
                print(f"    adaptive feasible at iter {iteration}", flush=True)
            break

        if not adaptive:
            break

        # Stagnation exit, after the feasibility break so feasibility always wins.
        if best_total_violations is None or int(iter_best["total_violations"]) < best_total_violations:
            best_total_violations = int(iter_best["total_violations"])
            no_improve_count = 0
        else:
            no_improve_count += 1
        if stagnation_patience > 0 and no_improve_count >= stagnation_patience:
            exit_reason = "stagnated"
            iteration_log[-1]["exit_reason"] = exit_reason
            print(
                f"    adaptive stagnated at iter {iteration} "
                f"(no strict improvement for {no_improve_count} iters)",
                flush=True,
            )
            break

        # Grow multipliers for violated constraints only.
        for c in ("c1", "c2", "c3", "c4"):
            if int(iter_best.get(c, 0)) > 0:
                multipliers[c] *= growth
        print(
            f"    adaptive iter {iteration} violations C1={iter_best['c1']} C2={iter_best['c2']} "
            f"C3={iter_best['c3']} C4={iter_best.get('c4', 0)} | cost={iter_best['cost']:,.2f} "
            f"-> growing penalties",
            flush=True,
        )

    if adaptive and exit_reason == "max_iterations" and iteration_log:
        iteration_log[-1]["exit_reason"] = exit_reason
        if not was_feasible:
            print(
                f"    adaptive exhausted {iterations_used} iterations without feasibility "
                f"(exit_reason=max_iterations). Consider raising "
                f"--adaptive-penalty-iterations or --adaptive-penalty-growth.",
                flush=True,
            )

    if best_eval is None:
        raise RuntimeError(f"VA batch {batch.batch_id}: no sample was selected")

    total_seconds = time.time() - batch_start
    status = "OK" if best_eval["total_violations"] == 0 else f"{best_eval['total_violations']} structural violations"
    print(
        f"  [3/3] Batch selected | {status} | aligned cost={money(best_eval['cost'])} | "
        f"total_time={total_seconds:.2f}s",
        flush=True,
    )

    all_energies = [float(row["va_reported_energy"]) for row in precision_rows]
    all_rel = [row["rel_diff"] for row in precision_rows if math.isfinite(row["rel_diff"])]
    va_stats = {
        "batch_id": int(batch.batch_id),
        "total_vars": int(total_vars),
        "dense_matrix_bytes": int(dense_matrix_bytes(total_vars)),
        "pyqubo_express_seconds": float(batch_model.express_seconds),
        "pyqubo_compile_seconds": float(batch_model.compile_seconds),
        "qubo_offset": float(offset),
        "va_offset_applied": float(va_offset),
        "va_repeats": int(args.va_repeats),
        "va_sample_calls": int(total_sample_calls),
        "va_constraint_retries": int(total_retries),
        "va_reads_per_call": int(base_reads),
        "va_total_reads": int(len(precision_rows)),
        "va_constraint_satisfied_reads": int(constraint_ok_count),
        "va_constraint_flagged_reads": int(constraint_total_count),
        "va_onehot_groups": int(len(one_hot_list)) if one_hot_list else 0,
        "va_vector_mode": str(args.va_vector_mode),
        "energy_min": min(all_energies) if all_energies else float("nan"),
        "energy_median": statistics.median(all_energies) if all_energies else float("nan"),
        "energy_max": max(all_energies) if all_energies else float("nan"),
        "cost_min": min(r["cost_min"] for r in repeat_records),
        "cost_median": statistics.median([r["cost_median"] for r in repeat_records]),
        "cost_max": max(r["cost_max"] for r in repeat_records),
        "structurally_feasible_reads": int(sum(r["structurally_feasible_reads"] for r in repeat_records)),
        "max_abs_rel_diff": max(all_rel) if all_rel else float("nan"),
        "median_abs_rel_diff": statistics.median(all_rel) if all_rel else float("nan"),
        "adaptive_iterations_used": int(iterations_used),
        "adaptive_was_feasible": bool(was_feasible),
        "adaptive_exit_reason": str(exit_reason),
        "final_mult_c1": float(multipliers["c1"]),
        "final_mult_c2": float(multipliers["c2"]),
        "final_mult_c3": float(multipliers["c3"]),
        "final_penalty_c1": float(qubo_meta["penalties"]["c1"]),
        "final_penalty_c3": float(qubo_meta["penalties"]["c3"]),
        "repeat_records": repeat_records,
    }

    result = BatchResult(
        batch_id=batch.batch_id,
        num_rows=num_rows,
        num_parts=num_parts,
        num_z=num_z,
        num_y=num_y,
        num_x=num_x,
        interactions=interactions,
        build_seconds=build_seconds,
        sample_seconds=sample_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
        # stage_used is an OpenJij retry-stage concept with no VA analogue.
        # Held at 0; VA call counts live in va_batch_summary.csv.
        stage_used=0,
        suggested_reads=base_reads,
        energy=float(best_eval["energy"]),
        reconstructed_cost=float(best_eval["cost"]),
        c1_violations=int(best_eval["c1"]),
        c2_violations=int(best_eval["c2"]),
        c3_violations=int(best_eval["c3"]),
        c4_violations=int(best_eval["c4"]),
        penalty_c1=float(qubo_meta["penalties"]["c1"]),
        penalty_c2=float(qubo_meta["penalties"]["c2"]),
        penalty_c3=float(qubo_meta["penalties"]["c3"]),
        open_hubs=list(best_eval["open_hubs"]),
        stocked_pairs=list(best_eval["stocked_pairs"]),
        assignments=list(best_eval["assignments"]),
        adaptive_iterations_used=int(iterations_used),
        adaptive_was_feasible=bool(was_feasible),
        final_penalty_multipliers=dict(multipliers),
        adaptive_iteration_log=list(iteration_log),
        adaptive_exit_reason=str(exit_reason),
    )

    del Q, qubo_meta, batch_model
    gc.collect()
    return result, precision_rows, va_stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def va_batch_summary_dataframe(
    results: list[BatchResult], va_stats: list[dict[str, Any]]
) -> pd.DataFrame:
    """batch_summary_dataframe's columns plus the VA-specific ones."""
    base = batch_summary_dataframe(results)
    extra = pd.DataFrame(
        [{k: v for k, v in s.items() if k != "repeat_records"} for s in va_stats]
    )
    if base.empty or extra.empty:
        return base
    return base.merge(extra, on="batch_id", how="left")


def va_repeat_dataframe(va_stats: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in va_stats:
        for rec in s["repeat_records"]:
            row = {"batch_id": int(s["batch_id"]), "total_vars": int(s["total_vars"])}
            row.update(rec)
            rows.append(row)
    return pd.DataFrame(rows)


def print_precision_report(precision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The experimental headline: can fp32 VA resolve costs at this scale?"""
    rel = [r["rel_diff"] for r in precision_rows if math.isfinite(r["rel_diff"])]
    abs_diffs = [r["abs_diff"] for r in precision_rows]

    stats: dict[str, Any] = {
        "reads_audited": int(len(precision_rows)),
        "reads_with_finite_rel_diff": int(len(rel)),
        "max_abs_rel_diff": float(max(rel)) if rel else None,
        "median_abs_rel_diff": float(statistics.median(rel)) if rel else None,
        "max_abs_diff": float(max(abs_diffs)) if abs_diffs else None,
        "median_abs_diff": float(statistics.median(abs_diffs)) if abs_diffs else None,
    }

    bar = "#" * 92
    print("\n" + bar, flush=True)
    print("VA FLOAT32 PRECISION AUDIT", flush=True)
    print(bar, flush=True)
    print("  VA computes in single precision; recomputed_energy is the float64 host reference.", flush=True)
    print(f"  reads audited:            {stats['reads_audited']:,}", flush=True)
    if rel:
        print(f"  MAX  |rel_diff|:          {stats['max_abs_rel_diff']:.6e}", flush=True)
        print(f"  MEDIAN |rel_diff|:        {stats['median_abs_rel_diff']:.6e}", flush=True)
        print(f"  max  |abs_diff|:          {stats['max_abs_diff']:,.4f}", flush=True)
        print(f"  median |abs_diff|:        {stats['median_abs_diff']:,.4f}", flush=True)
        print("", flush=True)
        print("  Read this against your cost scale: an |abs_diff| larger than the cost", flush=True)
        print("  differences you care about means VA cannot rank those solutions.", flush=True)
    else:
        print("  No finite relative differences (all recomputed energies were ~0).", flush=True)
    print(f"  detail CSV:               va_precision_audit.csv", flush=True)
    print(bar, flush=True)
    return stats


def print_va_header(
    data: dict[str, Any], batches: list[BatchSpec], args: argparse.Namespace, run_dir: Path
) -> None:
    avg_hubs = sum(len(v) for v in data["zip_to_hubs"].values()) / max(1, len(data["zip_to_hubs"]))
    print("\n" + "=" * 76, flush=True)
    print("VA QUBO MODEL - ALIGNED COST BASIS (PyQUBO formulation)", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:                  {data['dataset_name']}", flush=True)
    print(f"  active demand pairs:      {len(data['active']):,}", flush=True)
    print(f"  parts loaded:             {len(data['K']):,}", flush=True)
    print(f"  hubs loaded:              {len(data['J']):,}", flush=True)
    print(
        f"  candidate hubs/zip:       "
        f"{'all' if data['top_hubs_per_zip'] is None else data['top_hubs_per_zip']} (avg {avg_hubs:.2f})",
        flush=True,
    )
    print(f"  total QUBO batches:       {len(batches):,}", flush=True)
    print(f"  max Z vars/batch:         {args.max_z_vars_per_batch:,}", flush=True)
    print(f"  VA max total vars/batch:  {args.va_max_vars_per_batch:,}", flush=True)
    print("  formulation:              pyqubo Binary + Placeholder -> compile().to_qubo()", flush=True)
    print(
        f"  to_qubo offset:           "
        f"{'passed to VectorAnnealing.model()' if args.va_include_offset else 'held out of VA (default); recorded in the CSVs'}",
        flush=True,
    )
    print(f"  sampler:                  NEC Vector Annealing", flush=True)
    print(f"    vector_mode:            {args.va_vector_mode}", flush=True)
    print(f"    num_reads (base):       {args.num_reads}", flush=True)
    print(f"    num_sweeps:             {args.num_sweeps}", flush=True)
    print(f"    beta_range:             {args.va_beta_range or 'VA default [10,100,200]'}", flush=True)
    print(f"    repeats:                {args.va_repeats}  (no seed parameter exists in VA)", flush=True)
    print(f"    max constraint retries: {args.va_max_retries}", flush=True)
    print(f"    onehot flip option:     {'ON' if args.va_onehot else 'OFF'}", flush=True)
    print(f"  penalty mode:             {args.penalty_mode}", flush=True)
    print(
        f"  objective scale:          "
        f"{'ON  (penalties ~ scale x multiplier)' if args.enable_objective_scale else 'OFF (penalties flat at --min-penalty)'}",
        flush=True,
    )
    print(f"  min penalty (floor):      {float(args.min_penalty):,.2f}", flush=True)
    print(f"  adaptive penalty:         {args.adaptive_penalty_mode}", flush=True)
    if args.adaptive_penalty_mode == "within-batch":
        print(f"    max iterations:         {args.adaptive_penalty_iterations}", flush=True)
        print(f"    growth factor:          {args.adaptive_penalty_growth}", flush=True)
        s_lim = float(data["scalar"]["S_lim"])
        base = float(args.min_penalty)
        if not args.enable_objective_scale and base < s_lim:
            need = math.ceil(math.log(s_lim / base) / math.log(float(args.adaptive_penalty_growth)))
            print(
                f"    NOTE: C3 starts at {base:,.0f} vs S_lim {s_lim:,.0f}; stocking at a closed "
                f"hub is initially cheaper than opening one.",
                flush=True,
            )
            print(
                f"          escalation needs ~{need} iterations to pass S_lim "
                f"(budget is {args.adaptive_penalty_iterations}).",
                flush=True,
            )
    print(f"  constraint multiplier:    {float(args.constraint_multiplier):,.2f}"
          f"{'' if args.enable_objective_scale else '  (inactive: scale is OFF)'}", flush=True)
    print(f"  base miles:               {data['scalar']['base_miles']}", flush=True)
    print(f"  penalty start miles:      {data['scalar']['penalty_start_miles']}", flush=True)
    print(f"  max service miles:        {data['scalar']['max_service_miles']}", flush=True)
    print(f"  output folder:            {run_dir}", flush=True)
    print("=" * 76, flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_va_solver(args: argparse.Namespace) -> dict[str, Any] | None:
    tracemalloc.start()
    start = time.perf_counter()

    data = load_problem_data(
        args.dataset_dir,
        max_service_miles_override=args.max_service_miles,
        penalty_start_miles_override=args.penalty_start_miles,
        top_hubs_per_zip=None if int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip),
        max_parts_total=None if int(args.max_parts_total) < 0 else int(args.max_parts_total),
    )

    run_dir = Path(args.run_root).expanduser().resolve() / "va"
    batches = build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )

    if not bool(args.dry_run):
        run_dir.mkdir(parents=True, exist_ok=True)
    print_va_header(data, batches, args, run_dir)

    # Preflight: compile every batch model, learn the TRUE variable counts, check
    # the ceiling. Nothing is sampled and VectorAnnealing is not imported yet.
    print("\nPreflight: compiling every batch QUBO to measure true variable counts...", flush=True)
    plan = build_batch_plan(data, batches, args)
    print_batch_plan(plan, args)
    check_ceiling(plan, args)

    if bool(args.dry_run):
        print(
            f"\n--dry-run: all {len(plan)} batch(es) fit under the "
            f"{int(args.va_max_vars_per_batch):,}-variable ceiling. "
            "VectorAnnealing was not imported. Exiting.",
            flush=True,
        )
        return None

    pd.DataFrame(plan).to_csv(run_dir / "va_batch_plan.csv", index=False)

    beta_range = parse_beta_range(args.va_beta_range)
    VectorAnnealing = import_vector_annealing()

    deadline = None
    if args.qubo_time_limit is not None and float(args.qubo_time_limit) > 0:
        deadline = time.perf_counter() + float(args.qubo_time_limit)

    results: list[BatchResult] = []
    all_precision_rows: list[dict[str, Any]] = []
    all_va_stats: list[dict[str, Any]] = []
    stopped_time_limit = False

    for batch in batches:
        if deadline is not None and time.perf_counter() >= deadline:
            print(
                f"\nVA wall-time limit reached before next batch. "
                f"Completed {len(results)}/{len(batches)} batches.",
                flush=True,
            )
            stopped_time_limit = True
            break

        result, precision_rows, va_stats = solve_va_batch(
            VectorAnnealing, batch, data, args, beta_range
        )
        results.append(result)
        all_precision_rows.extend(precision_rows)
        all_va_stats.append(va_stats)

        # Same checkpoint filenames the OpenJij runner writes.
        checkpoint_raw = aggregate_raw_results(results)
        pd.DataFrame(checkpoint_raw["assignments"], columns=["zip_id", "hub_id", "part_id"]).to_csv(
            run_dir / "checkpoint_raw_assignments.csv", index=False
        )
        pd.DataFrame({"hub_id": checkpoint_raw["open_hubs"]}).to_csv(
            run_dir / "checkpoint_raw_open_hubs.csv", index=False
        )
        batch_summary_dataframe(results).to_csv(run_dir / "batch_summary_checkpoint.csv", index=False)
        pd.DataFrame(all_precision_rows).to_csv(run_dir / "va_precision_audit.csv", index=False)

    if not results:
        raise RuntimeError("No VA batches completed; increase --qubo-time-limit or check the instance")

    raw = aggregate_raw_results(results)
    final = postprocess_qubo_solution(
        raw["assignments"],
        raw["stocked_pairs"],
        raw["open_hubs"],
        data,
        repair_assignments=not bool(args.no_repair_assignments),
        trim_unused=not bool(args.no_trim_unused),
        hub_prune=not bool(args.no_hub_prune),
        hub_prune_max_iterations=int(args.hub_prune_max_iterations),
    )

    # Shared schema for the comparison tooling, plus the VA-augmented variant.
    batch_summary_dataframe(results).to_csv(run_dir / "batch_summary.csv", index=False)
    va_batch_summary_dataframe(results, all_va_stats).to_csv(run_dir / "va_batch_summary.csv", index=False)
    va_repeat_dataframe(all_va_stats).to_csv(run_dir / "va_repeat_summary.csv", index=False)

    # Same filenames and schemas the OpenJij runner emits in within-batch mode, so
    # the VA and OpenJij adaptive logs can be concatenated and compared directly.
    if args.adaptive_penalty_mode == "within-batch":
        batch_adaptive_summary_dataframe(results).to_csv(
            run_dir / "batch_adaptive_summary.csv", index=False
        )
        adaptive_iteration_log_dataframe(results).to_csv(
            run_dir / "adaptive_iteration_log.csv", index=False
        )
    pd.DataFrame(all_precision_rows).to_csv(run_dir / "va_precision_audit.csv", index=False)

    assignment_rows_dataframe(raw["assignments"], data).to_csv(
        run_dir / "raw_qubo_hub_zip_part_pairings.csv", index=False
    )
    pd.DataFrame({"hub_id": raw["open_hubs"]}).to_csv(run_dir / "raw_qubo_open_hubs.csv", index=False)
    pd.DataFrame(raw["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(
        run_dir / "raw_qubo_stocked_pairs.csv", index=False
    )

    wall = time.perf_counter() - start
    peak_mb, current_mb = memory_report_mb()
    runtime = {
        "wall_seconds": float(wall),
        "qubo_build_seconds": float(sum(r.build_seconds for r in results)),
        "qubo_sample_seconds": float(sum(r.sample_seconds for r in results)),
        "sample_eval_seconds": float(sum(r.eval_seconds for r in results)),
        "batch_total_seconds": float(sum(r.total_seconds for r in results)),
        "peak_memory_mb": float(peak_mb),
        "current_memory_mb": float(current_mb),
    }

    precision_summary = print_precision_report(all_precision_rows)

    raw_cost = compute_solution_cost(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    raw_audit = global_audit(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    extra = {
        "completed_batches": int(len(results)),
        "total_batches": int(len(batches)),
        "full_batch_coverage": bool(len(results) == len(batches) and not stopped_time_limit),
        "stopped_due_to_time_limit": bool(stopped_time_limit),
        "raw_solution": {
            "open_hubs_count": int(len(raw["open_hubs"])),
            "stocked_pairs_count": int(len(raw["stocked_pairs"])),
            "assignments_count": int(len(raw["assignments"])),
            "cost": raw_cost,
            "audit": raw_audit,
        },
        "postprocess": {
            "repair_assignments": bool(not args.no_repair_assignments),
            "trim_unused_open_stock": bool(not args.no_trim_unused),
            "missing_unrepaired": int(final.get("missing_unrepaired", 0)),
            "hub_prune_enabled": bool(not args.no_hub_prune),
            "hub_prune_closures": int(final.get("hub_prune_stats", {}).get("closures", 0)),
            "hub_prune_relocations": int(final.get("hub_prune_stats", {}).get("relocations", 0)),
        },
        "formulation": {
            "library": "pyqubo",
            "method": "Binary/Placeholder expressions -> compile() -> to_qubo(feed_dict=...)",
            "penalty_placeholders": [PLACEHOLDER_C1, PLACEHOLDER_C2, PLACEHOLDER_C3],
            "compile_once_per_batch": True,
            "zero_coefficient_tolerance": float(QUBO_ZERO_TOLERANCE),
            "c4_encoding": "not hard-encoded in the QUBO; priced in the final cost as overflow",
            "offset_passed_to_va": bool(args.va_include_offset),
            "offset_note": (
                "to_qubo()'s offset is the C1 constant (lam_c1 per active demand row). "
                "Add it to a reported energy to recover the true Hamiltonian value."
            ),
            "max_batch_offset": float(max(p["qubo_offset"] for p in plan)),
            "total_express_seconds": float(sum(p["pyqubo_express_seconds"] for p in plan)),
            "total_compile_seconds": float(sum(p["pyqubo_compile_seconds"] for p in plan)),
        },
        "va": {
            "engine": "NEC Vector Annealing",
            "manual_reference": VA_MANUAL_REF,
            # Provenance: proves this run executed on a local VE card via the
            # on-prem module, not through any cloud/service-client API.
            "execution_mode": "local_ve_card",
            "service_client_used": False,
            "module_file": str(getattr(VectorAnnealing, "__file__", "") or ""),
            "hostname": socket.gethostname(),
            "ve_node_number": os.environ.get("VE_NODE_NUMBER"),
            "ve_devices_visible": visible_ve_devices(),
            "module_version": next(
                (
                    str(getattr(VectorAnnealing, a))
                    for a in ("__version__", "VERSION", "version")
                    if getattr(VectorAnnealing, a, None) is not None
                    and not callable(getattr(VectorAnnealing, a))
                ),
                None,
            ),
            "python_version": sys.version.splitlines()[0],
            "objective_scale_enabled": bool(args.enable_objective_scale),
            "min_penalty": float(args.min_penalty),
            "constraint_multiplier": float(args.constraint_multiplier),
            "penalty_note": (
                "objective scale ON: penalties = max(min_penalty, scale*multiplier)"
                if args.enable_objective_scale
                else "objective scale OFF (VA default): penalties flat at min_penalty; "
                     "comparable to the run_aligned_fsl_comparison_noscale.py arm, NOT to "
                     "scale-ON OpenJij baselines"
            ),
            "seeded": False,
            "seed_note": "The VA PoC API exposes no seed parameter; runs are not reproducible read-for-read.",
            "adaptive_penalty_mode": str(args.adaptive_penalty_mode),
            "adaptive_penalty_iterations_max": int(args.adaptive_penalty_iterations),
            "adaptive_penalty_growth": float(args.adaptive_penalty_growth),
            "batches_reaching_feasibility": int(
                sum(1 for s in all_va_stats if s["adaptive_was_feasible"])
            ),
            "adaptive_exit_reasons": {
                str(s["batch_id"]): str(s["adaptive_exit_reason"]) for s in all_va_stats
            },
            "num_reads_base": int(args.num_reads),
            "num_sweeps": int(args.num_sweeps),
            "vector_mode": str(args.va_vector_mode),
            "beta_range": beta_range if beta_range is not None else "VA default [10,100,200]",
            "repeats": int(args.va_repeats),
            "max_retries": int(args.va_max_retries),
            "onehot_flip_option": bool(args.va_onehot),
            "max_vars_per_batch_ceiling": int(args.va_max_vars_per_batch),
            "hard_max_vars": int(VA_HARD_MAX_VARS),
            "max_batch_vars_observed": int(max(p["total_vars"] for p in plan)),
            "max_batch_dense_bytes": int(max(p["dense_matrix_bytes"] for p in plan)),
            "total_sample_calls": int(sum(s["va_sample_calls"] for s in all_va_stats)),
            "total_constraint_retries": int(sum(s["va_constraint_retries"] for s in all_va_stats)),
        },
        "precision_audit": precision_summary,
    }

    summary = write_solution_outputs(
        run_dir,
        solver_name="va",
        data=data,
        assignments=final["assignments"],
        stocked_pairs=final["stocked_pairs"],
        open_hubs=final["open_hubs"],
        runtime=runtime,
        extra=extra,
        assignment_sources=final.get("assignment_sources", {}),
    )
    print(final_results_block(summary, "VA FINAL RESULTS"), flush=True)

    # Repeat the precision headline last so it is the final thing on screen.
    if precision_summary.get("max_abs_rel_diff") is not None:
        print(
            f"\n>>> VA fp32 precision: max |rel_diff| = {precision_summary['max_abs_rel_diff']:.6e}, "
            f"median |rel_diff| = {precision_summary['median_abs_rel_diff']:.6e} "
            f"over {precision_summary['reads_audited']:,} reads. See va_precision_audit.csv.",
            flush=True,
        )

    gc.collect()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the FSL QUBO pipeline on the NEC Vector Annealing engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model / dataset ---------------------------------------------------
    p.add_argument("--dataset-dir", default="instances_low",
                   help="Folder with demand/distances/hubs/parameters/parts/zips CSV files.")
    p.add_argument("--run-root", default="",
                   help="Run folder. Results go to <run-root>/va. Empty picks outputs/va_fsl_<timestamp>.")

    p.add_argument("--max-service-miles", "--max-distance", dest="max_service_miles", type=float, default=None,
                   help="Eligibility cutoff. Default uses parameters.csv max_service_miles.")
    p.add_argument("--penalty-start-miles", type=float, default=None,
                   help="Override parameters.csv penalty_start_miles.")
    p.add_argument("--top-hubs-per-zip", type=int, default=-1,
                   help="Nearest eligible hubs per ZIP. -1 keeps all within max service miles.")
    p.add_argument("--max-parts-total", type=int, default=-1, help="Limit parts for testing. -1 keeps all.")

    p.add_argument("--part-batch-size", type=int, default=1000, help="Soft part count per batch.")
    p.add_argument("--max-z-vars-per-batch", type=int, default=50000,
                   help="Hard Z-variable cap used by build_batches. Caps Z only, not Y or X.")
    p.add_argument("--num-reads", type=int, default=100,
                   help="Base reads; scaled per batch by suggested_num_reads, then passed as VA num_reads.")
    p.add_argument("--num-sweeps", type=int, default=3000, help="VA num_sweeps (VA default is 500).")
    p.add_argument("--penalty-mode", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--min-penalty", type=float, default=50000.0,
                   help="Penalty floor, and -- with objective scale OFF, the default on this path -- "
                        "the flat effective penalty for C1-C4.")
    p.add_argument("--constraint-multiplier", type=float, default=5.0,
                   help="Only takes effect with --enable-objective-scale; otherwise ignored.")
    p.add_argument("--enable-objective-scale", action="store_true",
                   help="Re-enable objective-scale normalization, which lifts penalties to "
                        "~scale*multiplier (~2.51M on instances_low). OFF by default on the VA path, "
                        "so penalties sit flat at --min-penalty (50,000). Matches the arm that "
                        "run_aligned_fsl_comparison_noscale.py produces.")
    for c in ("c1", "c2", "c3", "c4"):
        p.add_argument(f"--min-penalty-{c}", type=float, default=-1.0)
        p.add_argument(f"--constraint-multiplier-{c}", type=float, default=-1.0)
    p.add_argument("--c4-mode", choices=["off", "auto", "on"], default="auto")
    p.add_argument("--x-empty-penalty-factor", type=float, default=0.0,
                   help="QUBO-formulation flag: penalize open-but-empty hubs. Default 0 (disabled).")
    p.add_argument("--y-overflow-penalty-factor", type=float, default=0.0,
                   help="QUBO-formulation flag: linear S_var proxy per stocked Y. Default 0 (disabled).")
    p.add_argument("--qubo-time-limit", type=float, default=5400.0,
                   help="Wall time budget in seconds, checked between batches. 0 disables.")
    p.add_argument("--no-repair-assignments", action="store_true",
                   help="Do not repair missing/multiple assignments in the final output.")
    p.add_argument("--no-trim-unused", action="store_true",
                   help="Do not trim unused open hubs / stocked pairs after final assignments.")
    p.add_argument("--adaptive-penalty-mode", choices=["off", "within-batch"], default="within-batch",
                   help="Adaptive penalty strategy. 'within-batch' iteratively grows the penalties of "
                        "violated constraints and resamples. ON by default here because with objective "
                        "scale OFF, C3 starts at 50,000 against an S_lim of 500,000 and only the "
                        "escalation fixes that. 'off' gives a single static pass.")
    p.add_argument("--adaptive-penalty-iterations", type=int, default=8,
                   help="Max adaptive iterations per batch. Default 8, not the OpenJij path's 5: "
                        "escalating C3 past S_lim needs ceil(log(500000/50000)/log(1.5)) = 6 "
                        "iterations at growth 1.5.")
    p.add_argument("--adaptive-penalty-growth", type=float, default=1.5,
                   help="Multiplicative growth for violated constraint penalties.")
    p.add_argument("--adaptive-penalty-stagnation-patience", type=int, default=0,
                   help="Break if total_violations has not strictly improved for N iterations. 0 disables.")
    p.add_argument("--no-hub-prune", action="store_true", help="Disable the global hub-pruning post-pass.")
    p.add_argument("--hub-prune-max-iterations", type=int, default=500,
                   help="Max passes over open hubs in the hub-prune post-pass.")

    # --- VA-specific -------------------------------------------------------
    va = p.add_argument_group("Vector Annealing")
    va.add_argument("--va-max-vars-per-batch", type=int, default=60000,
                    help=f"Ceiling on TOTAL vars per batch (Z+Y+X). VA hard max is {VA_HARD_MAX_VARS:,}.")
    va.add_argument("--va-repeats", type=int, default=1,
                    help="Re-run each batch N times and record the distribution. VA has no seed parameter.")
    va.add_argument("--va-max-retries", type=int, default=3,
                    help="Retries when VA returns a broken constraint on every read of a call.")
    va.add_argument("--va-onehot", action="store_true",
                    help="Declare C1 as VA one-hot flip groups. C1 penalty terms stay in the Hamiltonian.")
    va.add_argument("--va-vector-mode", choices=["SPEED", "ACCURACY"], default="ACCURACY",
                    help="VA vector_mode.")
    va.add_argument("--va-beta-range", default="",
                    help="VA beta_range as 'start,end[,nsteps]'. Empty uses the VA default [10,100,200].")
    va.add_argument("--va-include-offset", action="store_true",
                    help="Pass pyqubo's to_qubo() offset (the C1 constant) to VectorAnnealing.model() "
                         "as the Hamiltonian constant. OFF by default: the offset runs to ~2.5e8 on "
                         "instances_low, which would cost ~5 significant digits of VA's fp32 energy "
                         "resolution and break energy comparability with the OpenJij arm. The offset "
                         "is recorded in the CSVs and summary.json either way.")
    va.add_argument("--dry-run", action="store_true",
                    help="Load data, build batches, compile every QUBO, print the plan, check the "
                         "ceiling, exit. Does not import VectorAnnealing; runs on any machine.")

    args = p.parse_args(argv)

    # penalty_weights reads `disable_objective_scale` via getattr. The VA path
    # defaults to objective scale OFF, so penalties equal --min-penalty flat rather
    # than being lifted into the millions. See the module docstring for why.
    args.disable_objective_scale = not bool(args.enable_objective_scale)

    if int(args.va_repeats) < 1:
        p.error("--va-repeats must be >= 1")
    if int(args.va_max_retries) < 0:
        p.error("--va-max-retries must be >= 0")
    if not args.run_root:
        args.run_root = str(Path("outputs") / f"va_fsl_{now_stamp()}")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_va_solver(args)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        peak_mb, current_mb = memory_report_mb()
        print("ERROR: VA run failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"peak/current memory before exit: {peak_mb:,.1f}/{current_mb:,.1f} MB", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
