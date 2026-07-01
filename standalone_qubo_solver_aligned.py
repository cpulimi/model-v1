r"""
Standalone batched QUBO solver for the Dell FSL instance.

Aligned version: the QUBO objective and all recomputed costs use the same
transport/risk semantics as solve_fsl_risk_optimization.py:
  - distance term: max(0, d_ij - base_miles)
  - SLA/risk term: max(0, d_ij - penalty_start_miles)
The final printed "global violations" count is the SLA violation count
(assignments with d_ij > penalty_start_miles), matching the Gurobi script.
Structural feasibility violations are reported separately.

This script is intentionally self-contained: it does not import any project files.
It reads the instance CSVs directly, builds QUBOs manually, solves them with
OpenJij, post-processes the decoded solution into a clean feasible assignment
set, and writes presentation-ready outputs.

Required Python packages:
    pip install pandas openjij psutil

Typical Windows command:
    python standalone_qubo_solver.py ^
      --dataset-dir "C:\Users\Akshay Bhatkhande\Desktop\Dell Project\Old Model (Tune)\DATA\instances_low" ^
      --output-dir "C:\Users\Akshay Bhatkhande\Desktop\Dell Project\Old Model (Tune)\VSC\outputs\standalone_qubo" ^
      --max-z-vars-per-batch 20000 ^
      --top-hubs-per-zip -1 ^
      --qubo-time-limit 2700

Key implementation choices:
- Manual QUBO dictionary construction. This avoids pyqubo's symbolic compile
  bottleneck entirely.
- QUBO is built once per batch and reused across retry stages.
- Wall-time is checked between batches. A batch that has already started is
  allowed to finish cleanly.
- Batches are constructed with a hard Z-variable cap. If a single demand row
  exceeds the cap, the script raises a clear error because that row cannot be
  split without changing the assignment unit.
- Final outputs are post-processed by default so the assignment CSV has exactly
  one hub per active (zip, part), and hubs/stocked pairs are trimmed to those
  actually used by assignments. Raw QUBO-decoded outputs are also written.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

try:
    import openjij  # type: ignore
except ModuleNotFoundError:  # handled in main with a clear error
    openjij = None

try:
    import psutil  # type: ignore
except ModuleNotFoundError:
    psutil = None


DEFAULT_DATASET_DIR = r"C:\Users\Akshay Bhatkhande\Desktop\Dell Project\Old Model (Tune)\DATA\instances_low"
DEFAULT_OUTPUT_DIR = "outputs/standalone_qubo_aligned"

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


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def money(x: float) -> str:
    return f"${x:,.2f}"


def peak_rss_mb() -> float:
    """Best-effort current/peak process memory in MB, cross-platform."""
    if psutil is not None:
        try:
            proc = psutil.Process()
            if hasattr(proc, "memory_full_info"):
                info = proc.memory_full_info()
                peak = getattr(info, "peak_wset", None)  # Windows peak working set
                if peak:
                    return float(peak) / (1024.0 * 1024.0)
            return float(proc.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass

    try:
        import resource  # Unix only

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return 0.0


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


def load_problem_data(
    dataset_dir: Path,
    *,
    max_distance_override: float | None,
    top_hubs_per_zip: int | None,
    max_parts_total: int | None,
) -> dict[str, Any]:
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    params = read_csv_required(root, "parameters.csv")
    if params.empty:
        raise ValueError(f"Empty parameters.csv in {root}")
    row = params.iloc[0].to_dict()
    missing = [c for c in SCALAR_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"parameters.csv missing columns: {missing}")

    scalar = {c: float(row[c]) for c in SCALAR_COLUMNS}
    scalar["L"] = int(float(scalar["L"]))

    hubs = read_csv_required(root, "hubs.csv")
    parts = read_csv_required(root, "parts.csv")
    zips = read_csv_required(root, "zips.csv")
    demand = read_csv_required(root, "demand.csv")
    distances = read_csv_required(root, "distances.csv")

    required = {
        "hubs.csv": (hubs, ["hub_id", "T_j"]),
        "parts.csv": (parts, ["part_id"]),
        "zips.csv": (zips, ["zip_id"]),
        "demand.csv": (demand, ["zip_id", "part_id", "Q_ik", "b_ik"]),
        "distances.csv": (distances, ["zip_id", "hub_id", "d_ij"]),
    }
    for filename, (df, cols) in required.items():
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{filename} missing columns: {missing_cols}")

    price_col = "P_k" if "P_k" in parts.columns else "price"
    if price_col not in parts.columns:
        raise ValueError("parts.csv must contain P_k or price")
    if price_col != "P_k":
        parts = parts.rename(columns={price_col: "P_k"})

    hubs["T_j"] = pd.to_numeric(hubs["T_j"], errors="coerce").fillna(0).astype(int)
    parts["P_k"] = pd.to_numeric(parts["P_k"], errors="coerce").fillna(0.0)
    demand["Q_ik"] = pd.to_numeric(demand["Q_ik"], errors="coerce").fillna(0).astype(int)
    demand["b_ik"] = pd.to_numeric(demand["b_ik"], errors="coerce").fillna(0.0)
    distances["d_ij"] = pd.to_numeric(distances["d_ij"], errors="coerce")
    distances = distances.dropna(subset=["d_ij"])

    all_parts = sorted(parts["part_id"].unique().tolist())
    if max_parts_total is not None and max_parts_total > 0:
        selected_parts = all_parts[: int(max_parts_total)]
    else:
        selected_parts = all_parts
    selected_parts_set = set(selected_parts)
    parts = parts[parts["part_id"].isin(selected_parts_set)].copy()
    demand = demand[demand["part_id"].isin(selected_parts_set)].copy()

    max_distance = (
        float(max_distance_override)
        if max_distance_override is not None and float(max_distance_override) > 0
        else float(scalar.get("max_service_miles", 180.0))
    )

    dist_f = distances[distances["d_ij"] <= max_distance].copy()
    if dist_f.empty:
        raise ValueError(
            f"No distance rows with d_ij <= {max_distance}. "
            "Increase --max-distance or check distances.csv."
        )

    dist_f = dist_f.sort_values(["zip_id", "d_ij", "hub_id"])
    if top_hubs_per_zip is not None and top_hubs_per_zip > 0:
        dist_f = dist_f.groupby("zip_id", sort=False).head(int(top_hubs_per_zip)).copy()

    zip_to_hubs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    distance_map: dict[tuple[str, str], float] = {}
    for r in dist_f[["zip_id", "hub_id", "d_ij"]].itertuples(index=False):
        i, j, dij = str(r.zip_id), str(r.hub_id), float(r.d_ij)
        zip_to_hubs[i].append((j, dij))
        distance_map[(i, j)] = dij
    for i in list(zip_to_hubs):
        zip_to_hubs[i].sort(key=lambda x: (x[1], x[0]))

    eligible_zips = set(zip_to_hubs)
    active = demand[
        (demand["Q_ik"] == 1)
        & (demand["zip_id"].isin(eligible_zips))
        & (demand["part_id"].isin(selected_parts_set))
    ][["zip_id", "part_id", "b_ik"]].copy()

    # Defensive de-duplication: one active row per (zip, part). If duplicates exist,
    # sum demand weight so objective terms remain conservative.
    active = (
        active.groupby(["zip_id", "part_id"], as_index=False, sort=False)["b_ik"]
        .sum()
        .reset_index(drop=True)
    )

    active_parts = set(active["part_id"].unique().tolist())
    part_order = [p for p in selected_parts if p in active_parts]

    p_map = {str(r.part_id): float(r.P_k) for r in parts[["part_id", "P_k"]].itertuples(index=False)}
    t_map = {str(r.hub_id): int(r.T_j) for r in hubs[["hub_id", "T_j"]].itertuples(index=False)}
    b_map = {(str(r.zip_id), str(r.part_id)): float(r.b_ik) for r in active.itertuples(index=False)}

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
        "D": distance_map,
        "B": b_map,
        "zip_to_hubs": dict(zip_to_hubs),
        "J": sorted(hubs["hub_id"].unique().tolist()),
        "K": selected_parts,
        "max_distance": max_distance,
        "top_hubs_per_zip": top_hubs_per_zip,
        "max_parts_total": max_parts_total,
        "baseline_part_homes_rows": len(pd.read_csv(baseline_path)) if baseline_path.is_file() else 0,
        "parameter_key_rows": len(pd.read_csv(parameter_key_path)) if parameter_key_path.is_file() else 0,
    }


def build_batches(
    active: pd.DataFrame,
    part_order: list[str],
    zip_to_hubs: dict[str, list[tuple[str, float]]],
    *,
    part_batch_size: int,
    max_z_vars_per_batch: int,
) -> list[BatchSpec]:
    """Build batches with a hard cap on estimated Z variables.

    The normal unit is a part: all demand rows for a part stay together. If a
    single part exceeds the Z cap, that part is split by demand rows while still
    keeping each (zip, part) assignment row intact.
    """
    if part_batch_size <= 0:
        raise ValueError("--part-batch-size must be > 0")
    if max_z_vars_per_batch <= 0:
        raise ValueError("--max-z-vars-per-batch must be > 0")

    df = active.reset_index(drop=True).copy()
    row_z = df["zip_id"].map(lambda z: len(zip_to_hubs.get(str(z), ())))
    if (row_z <= 0).any():
        bad = df.loc[row_z <= 0, "zip_id"].head(5).tolist()
        raise ValueError(f"Active demand rows without candidate hubs, examples: {bad}")
    if (row_z > max_z_vars_per_batch).any():
        bad_rows = df.loc[row_z > max_z_vars_per_batch, ["zip_id", "part_id"]].head(5).to_dict("records")
        raise ValueError(
            "At least one single demand row has more candidate hubs than --max-z-vars-per-batch. "
            f"Examples: {bad_rows}. Increase the cap or reduce --top-hubs-per-zip."
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
            batches.append(
                BatchSpec(
                    batch_id=len(batches) + 1,
                    row_indices=list(current),
                    estimated_z_vars=int(current_z),
                    note=note,
                )
            )
        current = []
        current_parts = set()
        current_z = 0

    for part_id in part_order:
        idxs = rows_by_part.get(part_id, [])
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
                        BatchSpec(
                            batch_id=len(batches) + 1,
                            row_indices=list(chunk),
                            estimated_z_vars=int(chunk_z),
                            note=f"split_large_part:{part_id}",
                        )
                    )
                    chunk = []
                    chunk_z = 0
                chunk.append(idx)
                chunk_z += rz
            if chunk:
                batches.append(
                    BatchSpec(
                        batch_id=len(batches) + 1,
                        row_indices=list(chunk),
                        estimated_z_vars=int(chunk_z),
                        note=f"split_large_part:{part_id}",
                    )
                )
            continue

        would_exceed_z = current and (current_z + part_z > max_z_vars_per_batch)
        would_exceed_parts = current and (len(current_parts) >= part_batch_size)
        if would_exceed_z or would_exceed_parts:
            flush()

        current.extend(idxs)
        current_parts.add(part_id)
        current_z += part_z

    flush()
    return batches


def var_x(hub_id: str) -> str:
    return f"X|{hub_id}"


def var_y(hub_id: str, part_id: str) -> str:
    return f"Y|{hub_id}|{part_id}"


def var_z(zip_id: str, hub_id: str, part_id: str) -> str:
    return f"Z|{zip_id}|{hub_id}|{part_id}"


def add_qubo(Q: dict[tuple[str, str], float], u: str, v: str, coeff: float) -> None:
    if abs(coeff) <= 1e-12:
        return
    if u != v and v < u:
        u, v = v, u
    key = (u, v)
    new_val = Q.get(key, 0.0) + float(coeff)
    if abs(new_val) <= 1e-12:
        Q.pop(key, None)
    else:
        Q[key] = new_val


def estimate_objective_scale(batch_df: pd.DataFrame, data: dict[str, Any]) -> float:
    if batch_df.empty:
        return 1.0
    scalar = data["scalar"]
    part_ids = batch_df["part_id"].unique().tolist()
    max_part_cost = max((float(data["P"].get(k, 0.0)) for k in part_ids), default=0.0)
    max_b = float(batch_df["b_ik"].max()) if "b_ik" in batch_df.columns else 0.0
    max_d = max((float(v) for v in data["D"].values()), default=0.0)
    base_miles = float(scalar.get("base_miles", 0.0))
    penalty_start = float(scalar.get("penalty_start_miles", scalar.get("d_s", 0.0)))
    max_transport = (
        float(scalar["lambda_1"]) * float(scalar["h_s"]) * max_b
        + float(scalar["lambda_2"]) * float(scalar["h_d"]) * max_b * max(0.0, max_d - base_miles)
        + float(scalar["lambda_3"]) * max_b * max(0.0, max_d - penalty_start)
    )
    return max(1.0, max_part_cost + float(scalar["C"]) + float(scalar["S_lim"]) + max_transport)


def penalty_weights(
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    scale = estimate_objective_scale(batch_df, data)
    if args.penalty_mode == "fixed":
        base = float(args.min_penalty)
    else:
        base = max(float(args.min_penalty), scale * float(args.constraint_multiplier))

    out = {"c1": base, "c2": base, "c3": base, "c4": base}
    for c in ("c1", "c2", "c3", "c4"):
        override_min = getattr(args, f"min_penalty_{c}")
        override_mult = getattr(args, f"constraint_multiplier_{c}")
        if override_min is not None and float(override_min) >= 0:
            out[c] = float(override_min) if args.penalty_mode == "fixed" else max(float(override_min), scale * float(args.constraint_multiplier))
        if override_mult is not None and float(override_mult) >= 0:
            out[c] = max(float(args.min_penalty), scale * float(override_mult))

    # Adaptive multipliers (within-batch adaptive penalty). When None, the
    # returned penalties are bit-identical to the static behavior.
    if multipliers is not None:
        for c in ("c1", "c2", "c3", "c4"):
            if c in multipliers:
                out[c] = out[c] * float(multipliers[c])
    return out


def assignment_cost(zip_id: str, hub_id: str, part_id: str, data: dict[str, Any]) -> dict[str, float]:
    """Assignment cost using the same transport/risk semantics as the Gurobi model.

    Distance files store raw miles.  The economic distance term charges only
    miles above base_miles, while the SLA/risk penalty charges only miles above
    penalty_start_miles.  This is the key alignment change versus the older
    standalone QUBO, which charged raw d_ij and used d_s as the penalty start.
    """
    scalar = data["scalar"]
    b = float(data["B"].get((zip_id, part_id), 0.0))
    dij = float(data["D"].get((zip_id, hub_id), 0.0))
    base_miles = float(scalar.get("base_miles", 0.0))
    penalty_start = float(scalar.get("penalty_start_miles", scalar.get("d_s", 0.0)))
    miles_after_base = max(0.0, dij - base_miles)
    miles_after_penalty_start = max(0.0, dij - penalty_start)
    linehaul = float(scalar["lambda_1"]) * float(scalar["h_s"]) * b
    distance = float(scalar["lambda_2"]) * float(scalar["h_d"]) * b * miles_after_base
    penalty = float(scalar["lambda_3"]) * b * miles_after_penalty_start
    return {
        "b_ik": b,
        "d_ij": dij,
        "base_miles": base_miles,
        "penalty_start_miles": penalty_start,
        "miles_after_base": miles_after_base,
        "miles_after_penalty_start": miles_after_penalty_start,
        "sla_violation": int(dij > penalty_start),
        "linehaul_cost": linehaul,
        "distance_cost": distance,
        "distance_penalty_cost": penalty,
        "assignment_cost": linehaul + distance + penalty,
    }


def compute_solution_cost(
    assignments: Iterable[tuple[str, str, str]],
    stocked_pairs: Iterable[tuple[str, str]],
    open_hubs: Iterable[str],
    data: dict[str, Any],
) -> dict[str, float]:
    scalar = data["scalar"]
    assignments_set = set(assignments)
    stocked_set = set(stocked_pairs)
    open_set = set(open_hubs)

    inventory_cost = sum(float(data["P"].get(k, 0.0)) for _, k in stocked_set)
    fixed_open_cost = float(scalar["S_lim"]) * float(len(open_set))
    transfer_cost = sum((1 - int(data["T"].get(j, 0))) * float(scalar["C"]) for j, _ in stocked_set)

    by_hub: dict[str, int] = defaultdict(int)
    for j, _ in stocked_set:
        by_hub[j] += 1
    overflow_units = sum(max(0, cnt - int(scalar["L"])) for cnt in by_hub.values())
    overflow_cost = float(scalar["S_var"]) * float(overflow_units)

    transport_cost = 0.0
    for i, j, k in assignments_set:
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


def build_qubo_for_batch(
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    scalar = data["scalar"]
    zip_to_hubs = data["zip_to_hubs"]
    penalties = penalty_weights(batch_df, data, args, multipliers=multipliers)

    Q: dict[tuple[str, str], float] = {}
    z_name: dict[tuple[str, str, str], str] = {}
    y_name: dict[tuple[str, str], str] = {}
    x_name: dict[str, str] = {}
    demand_groups: dict[tuple[str, str], list[tuple[str, str]]] = {}

    # Objective terms and variable creation.
    for r in batch_df[["zip_id", "part_id", "b_ik"]].itertuples(index=False):
        i = str(r.zip_id)
        k = str(r.part_id)
        b = float(r.b_ik)
        group_entries: list[tuple[str, str]] = []
        for j, dij in zip_to_hubs[i]:
            zn = var_z(i, j, k)
            yn = var_y(j, k)
            xn = var_x(j)
            z_name[(i, j, k)] = zn
            y_name[(j, k)] = yn
            x_name[j] = xn
            group_entries.append((j, zn))

            base_miles = float(scalar.get("base_miles", 0.0))
            penalty_start = float(scalar.get("penalty_start_miles", scalar.get("d_s", 0.0)))
            transport = (
                float(scalar["lambda_1"]) * float(scalar["h_s"]) * b
                + float(scalar["lambda_2"]) * float(scalar["h_d"]) * b * max(0.0, float(dij) - base_miles)
                + float(scalar["lambda_3"]) * b * max(0.0, float(dij) - penalty_start)
            )
            add_qubo(Q, zn, zn, transport)
        demand_groups[(i, k)] = group_entries

    for (j, k), yn in y_name.items():
        add_qubo(Q, yn, yn, float(data["P"].get(k, 0.0)))
        add_qubo(Q, yn, yn, (1 - int(data["T"].get(j, 0))) * float(scalar["C"]))

    for j, xn in x_name.items():
        add_qubo(Q, xn, xn, float(scalar["S_lim"]))

    # C1: exact one assignment per active (zip, part).
    # penalty * (sum_z - 1)^2, constant omitted.
    lam1 = float(penalties["c1"])
    for entries in demand_groups.values():
        zvars = [zn for _, zn in entries]
        for zn in zvars:
            add_qubo(Q, zn, zn, -lam1)
        for a in range(len(zvars)):
            za = zvars[a]
            for b in range(a + 1, len(zvars)):
                add_qubo(Q, za, zvars[b], 2.0 * lam1)

    # C2: Z_ijk <= Y_jk, encoded as penalty * z * (1 - y).
    lam2 = float(penalties["c2"])
    for (i, j, k), zn in z_name.items():
        yn = y_name[(j, k)]
        add_qubo(Q, zn, zn, lam2)
        add_qubo(Q, zn, yn, -lam2)

    # C3: Y_jk <= X_j, encoded as penalty * y * (1 - x).
    lam3 = float(penalties["c3"])
    for (j, k), yn in y_name.items():
        xn = x_name[j]
        add_qubo(Q, yn, yn, lam3)
        add_qubo(Q, yn, xn, -lam3)

    # Tier 3.8: approximate X<=sum(Y) penalty. Pure QUBO can't express OR(Y),
    # so we add lam_xy on each X-diagonal and refund -lam_xy/n_k per (X,Y) pair.
    # Default factor is 0 (disabled) — the hub-prune post-pass is the primary fix.
    x_empty_factor = float(getattr(args, "x_empty_penalty_factor", 0.0))
    if x_empty_factor > 0:
        lam_xy = float(scalar["S_lim"]) * x_empty_factor
        parts_per_hub: dict[str, list[str]] = defaultdict(list)
        for (j, k) in y_name:
            parts_per_hub[j].append(k)
        for j, xn in x_name.items():
            parts = parts_per_hub.get(j, [])
            if not parts:
                continue
            n_k = len(parts)
            add_qubo(Q, xn, xn, lam_xy)
            per_pair_refund = lam_xy / float(n_k)
            for k in parts:
                yn = y_name[(j, k)]
                add_qubo(Q, xn, yn, -per_pair_refund)

    # Tier 3.9: linear S_var-style overflow penalty per stocked Y. Approximates
    # max(0, sum(Y) - L) without slack vars. Default factor is 0 (off);
    # inactive on instances_low since L=50000 is never approached.
    y_overflow_factor = float(getattr(args, "y_overflow_penalty_factor", 0.0))
    if y_overflow_factor > 0:
        s_var_coeff = float(scalar["S_var"]) * y_overflow_factor
        for (j, k), yn in y_name.items():
            add_qubo(Q, yn, yn, s_var_coeff)

    c4_note = "not_active"
    if args.c4_mode == "on":
        # The provided instances_low has L=50000 and only 600 parts, so C4 is
        # inactive. A correct inequality encoding for general tight capacities
        # requires additional slack design; forcing the old equality encoding
        # would be mathematically wrong when stock < L. Fail loudly instead.
        max_stock = max((sum(1 for jj, _ in y_name if jj == j) for j in x_name), default=0)
        if max_stock > int(scalar["L"]):
            raise NotImplementedError(
                "C4 is active for this batch, but this standalone script intentionally "
                "does not use the old incorrect equality encoding. Use --c4-mode auto/off "
                "or add a correct inequality slack formulation."
            )
        c4_note = "forced_but_inactive"

    return {
        "Q": Q,
        "z_name": z_name,
        "y_name": y_name,
        "x_name": x_name,
        "demand_groups": demand_groups,
        "penalties": penalties,
        "c4_note": c4_note,
    }


def sample_is_one(value: Any) -> bool:
    try:
        return int(round(float(value))) == 1
    except Exception:
        return bool(value)


def iter_openjij_samples(response: Any) -> Iterator[tuple[dict[str, int], float]]:
    """Yield (sample, energy) from common OpenJij/dimod response variants."""
    # dimod-like SampleSet
    if hasattr(response, "data"):
        for kwargs in ({"fields": ["sample", "energy"]}, {}):
            try:
                iterator = response.data(**kwargs) if kwargs else response.data(["sample", "energy"])
                for item in iterator:
                    sample = getattr(item, "sample", None)
                    energy = getattr(item, "energy", None)
                    if sample is None and isinstance(item, tuple) and len(item) >= 2:
                        sample, energy = item[0], item[1]
                    if sample is not None and energy is not None:
                        yield {str(k): int(round(float(v))) for k, v in dict(sample).items()}, float(energy)
                return
            except Exception:
                pass

    # OpenJij Response often exposes states/energies with variable indices.
    if hasattr(response, "states") and hasattr(response, "energies"):
        variables = None
        for attr in ("variables", "indices"):
            if hasattr(response, attr):
                variables = list(getattr(response, attr))
                break
        if variables is None:
            raise RuntimeError("OpenJij response has states/energies but no variables/indices labels.")
        for state, energy in zip(response.states, response.energies):
            yield {str(variables[i]): int(round(float(v))) for i, v in enumerate(state)}, float(energy)
        return

    # dimod record fallback
    if hasattr(response, "record") and hasattr(response, "variables"):
        variables = list(response.variables)
        for rec in response.record:
            sample_arr = getattr(rec, "sample", rec[0])
            energy = getattr(rec, "energy", rec[1])
            yield {str(variables[i]): int(round(float(v))) for i, v in enumerate(sample_arr)}, float(energy)
        return

    raise RuntimeError("Could not iterate samples from OpenJij response. Unknown response format.")


def evaluate_sample(
    sample: dict[str, int],
    energy: float,
    qubo_meta: dict[str, Any],
    batch_df: pd.DataFrame,
    data: dict[str, Any],
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
    scale = max(1.0, math.sqrt(num_z / 5000.0))
    return max(1, int(math.ceil(float(args.num_reads) * scale)))


def run_adaptive_penalty_loop(
    batch: BatchSpec,
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    sampler: Any,
    base_reads: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, float], int, bool]:
    """Within-batch adaptive penalty: iteratively grow penalties for violated
    constraints, rebuild the QUBO, resample. Returns (best_eval, qubo_meta_final,
    multipliers_final, adaptive_iterations_used, was_feasible).
    """
    multipliers = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    growth = float(args.adaptive_penalty_growth)
    max_iter = int(args.adaptive_penalty_iterations)
    best_eval: dict[str, Any] | None = None
    qubo_meta_final: dict[str, Any] | None = None
    adaptive_iter_used = 0
    was_feasible = False

    for iteration in range(1, max_iter + 1):
        adaptive_iter_used = iteration

        # Rebuild the QUBO with the current per-constraint multipliers.
        qubo_meta = build_qubo_for_batch(batch_df, data, args, multipliers=multipliers)
        qubo_meta_final = qubo_meta
        Q = qubo_meta["Q"]

        # Single sampling pass at base reads (no retry-reads escalation here).
        sample_kwargs: dict[str, Any] = {"num_reads": base_reads}
        if args.seed is not None and int(args.seed) >= 0:
            sample_kwargs["seed"] = int(args.seed) + batch.batch_id * 1000 + iteration
        if int(args.num_sweeps or 0) > 0:
            sample_kwargs["num_sweeps"] = int(args.num_sweeps)

        print(
            f"    adaptive iter {iteration}/{max_iter} | "
            f"multipliers c1={multipliers['c1']:.2f} c2={multipliers['c2']:.2f} "
            f"c3={multipliers['c3']:.2f}",
            flush=True,
        )
        response = sampler.sample_qubo(Q, **sample_kwargs)

        iter_best: dict[str, Any] | None = None
        for sample, energy in iter_openjij_samples(response):
            ev = evaluate_sample(sample, energy, qubo_meta, batch_df, data)
            key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
            if iter_best is None:
                iter_best = ev
            else:
                old_key = (
                    iter_best["total_violations"],
                    iter_best["c1"],
                    iter_best["c2"],
                    iter_best["c3"],
                    iter_best["cost"],
                    iter_best["energy"],
                )
                if key < old_key:
                    iter_best = ev

        if iter_best is None:
            raise RuntimeError("OpenJij returned no samples during adaptive penalty loop.")

        if best_eval is None:
            best_eval = iter_best
        else:
            old_key = (
                best_eval["total_violations"],
                best_eval["c1"],
                best_eval["c2"],
                best_eval["c3"],
                best_eval["cost"],
                best_eval["energy"],
            )
            new_key = (
                iter_best["total_violations"],
                iter_best["c1"],
                iter_best["c2"],
                iter_best["c3"],
                iter_best["cost"],
                iter_best["energy"],
            )
            if new_key < old_key:
                best_eval = iter_best

        print(
            f"    adaptive iter {iteration} | violations C1={iter_best['c1']} "
            f"C2={iter_best['c2']} C3={iter_best['c3']} | "
            f"cost={iter_best['cost']:.2f}",
            flush=True,
        )

        if int(iter_best["total_violations"]) == 0:
            was_feasible = True
            print(f"    adaptive feasible at iter {iteration}", flush=True)
            break

        # Grow multipliers for violated constraints only.
        if int(iter_best["c1"]) > 0:
            multipliers["c1"] *= growth
        if int(iter_best["c2"]) > 0:
            multipliers["c2"] *= growth
        if int(iter_best["c3"]) > 0:
            multipliers["c3"] *= growth
        if int(iter_best.get("c4", 0)) > 0:
            multipliers["c4"] *= growth

    return best_eval, qubo_meta_final, multipliers, adaptive_iter_used, was_feasible


def solve_batch(
    batch: BatchSpec,
    active: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
) -> BatchResult:
    if openjij is None:
        raise RuntimeError("Missing dependency: openjij. Install it with: pip install openjij")

    batch_start = time.time()
    batch_df = active.iloc[batch.row_indices].reset_index(drop=True)
    num_rows = len(batch_df)
    num_parts = batch_df["part_id"].nunique()

    print(f"\n=== Batch {batch.batch_id} | {num_parts} parts | {num_rows:,} demand rows | estimated Z={batch.estimated_z_vars:,} ===", flush=True)
    if batch.note:
        print(f"  note: {batch.note}", flush=True)

    print("  [1/3] Building manual QUBO dictionary...", flush=True)
    t0 = time.time()
    qubo_meta = build_qubo_for_batch(batch_df, data, args)
    build_seconds = time.time() - t0
    Q = qubo_meta["Q"]
    num_z = len(qubo_meta["z_name"])
    num_y = len(qubo_meta["y_name"])
    num_x = len(qubo_meta["x_name"])
    interactions = len(Q)
    print(
        f"    built in {build_seconds:.2f}s | Z={num_z:,} Y={num_y:,} X={num_x:,} interactions={interactions:,}",
        flush=True,
    )
    print(
        "    penalties: "
        f"C1={qubo_meta['penalties']['c1']:,.2f} "
        f"C2={qubo_meta['penalties']['c2']:,.2f} "
        f"C3={qubo_meta['penalties']['c3']:,.2f}",
        flush=True,
    )

    sampler = openjij.SASampler()
    base_reads = suggested_num_reads(num_z, args)
    best_eval: dict[str, Any] | None = None
    sample_seconds = 0.0
    eval_seconds = 0.0
    stage_used = 0
    adaptive_iters_used = 0
    adaptive_was_feasible = False
    final_multipliers = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}

    # Within-batch adaptive penalty phase (runs before the retry-reads loop).
    if args.adaptive_penalty_mode == "within-batch":
        print("  [2a/3] Adaptive penalty phase...", flush=True)
        t_adapt = time.time()
        (
            best_eval,
            qubo_meta,
            final_multipliers,
            adaptive_iters_used,
            adaptive_was_feasible,
        ) = run_adaptive_penalty_loop(batch, batch_df, data, args, sampler, base_reads)
        sample_seconds += time.time() - t_adapt
        Q = qubo_meta["Q"]
        if adaptive_was_feasible:
            print("  [2b/3] Adaptive succeeded - skipping retry-reads fallback", flush=True)
        else:
            print(
                f"  [2b/3] Adaptive exhausted ({adaptive_iters_used} iters) - "
                f"running retry-reads fallback with final multipliers",
                flush=True,
            )

    # Existing retry-reads loop. Runs always when adaptive is off, or as the
    # fallback when adaptive did not reach feasibility.
    if args.adaptive_penalty_mode == "off" or not adaptive_was_feasible:
        if args.adaptive_penalty_mode == "within-batch":
            # Rebuild the QUBO with the final adapted multipliers for the fallback.
            qubo_meta = build_qubo_for_batch(batch_df, data, args, multipliers=final_multipliers)
            Q = qubo_meta["Q"]

        print("  [2/3] Sampling QUBO with OpenJij...", flush=True)
        for stage in range(1, int(args.max_stages) + 1):
            stage_used = stage
            stage_reads = max(1, int(math.ceil(base_reads * (float(args.retry_reads_boost) ** (stage - 1)))))
            sample_kwargs: dict[str, Any] = {"num_reads": stage_reads}
            if args.seed is not None and int(args.seed) >= 0:
                sample_kwargs["seed"] = int(args.seed) + batch.batch_id * 1000 + stage
            if int(args.num_sweeps or 0) > 0:
                sample_kwargs["num_sweeps"] = int(args.num_sweeps)

            print(f"    stage {stage}/{args.max_stages}: sampling {stage_reads} reads...", flush=True)
            t_sample = time.time()
            response = sampler.sample_qubo(Q, **sample_kwargs)
            stage_sample_seconds = time.time() - t_sample
            sample_seconds += stage_sample_seconds

            print(f"    stage {stage}/{args.max_stages}: evaluating samples...", flush=True)
            t_eval = time.time()
            stage_best: dict[str, Any] | None = None
            sample_count = 0
            for sample, energy in iter_openjij_samples(response):
                sample_count += 1
                ev = evaluate_sample(sample, energy, qubo_meta, batch_df, data)
                key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
                if stage_best is None:
                    stage_best = ev
                else:
                    old_key = (
                        stage_best["total_violations"],
                        stage_best["c1"],
                        stage_best["c2"],
                        stage_best["c3"],
                        stage_best["cost"],
                        stage_best["energy"],
                    )
                    if key < old_key:
                        stage_best = ev
            stage_eval_seconds = time.time() - t_eval
            eval_seconds += stage_eval_seconds

            if stage_best is None:
                raise RuntimeError("OpenJij returned no samples.")

            if best_eval is None:
                best_eval = stage_best
            else:
                key = (
                    stage_best["total_violations"],
                    stage_best["c1"],
                    stage_best["c2"],
                    stage_best["c3"],
                    stage_best["cost"],
                    stage_best["energy"],
                )
                old_key = (
                    best_eval["total_violations"],
                    best_eval["c1"],
                    best_eval["c2"],
                    best_eval["c3"],
                    best_eval["cost"],
                    best_eval["energy"],
                )
                if key < old_key:
                    best_eval = stage_best

            print(
                f"    stage {stage} done | samples={sample_count} "
                f"sample_time={stage_sample_seconds:.2f}s eval_time={stage_eval_seconds:.2f}s "
                f"best_violations={stage_best['total_violations']} "
                f"(C1={stage_best['c1']}, C2={stage_best['c2']}, C3={stage_best['c3']}) "
                f"energy={stage_best['energy']:.2f}",
                flush=True,
            )
            if int(stage_best["total_violations"]) == 0:
                print("    early stop: feasible batch sample found", flush=True)
                break

    if best_eval is None:
        raise RuntimeError("No QUBO sample was selected.")

    total_seconds = time.time() - batch_start
    status = "OK" if best_eval["total_violations"] == 0 else f"{best_eval['total_violations']} violations"
    print(
        f"  [3/3] Batch selected | {status} | cost={money(best_eval['cost'])} "
        f"| total_time={total_seconds:.2f}s",
        flush=True,
    )

    return BatchResult(
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
        stage_used=stage_used,
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
        adaptive_iterations_used=int(adaptive_iters_used),
        adaptive_was_feasible=bool(adaptive_was_feasible),
        final_penalty_multipliers=dict(final_multipliers),
    )


def aggregate_raw_results(results: list[BatchResult]) -> dict[str, Any]:
    open_hubs = sorted({hub for r in results for hub in r.open_hubs})
    stocked_pairs = sorted({pair for r in results for pair in r.stocked_pairs})
    assignments = sorted({assignment for r in results for assignment in r.assignments})
    return {
        "open_hubs": open_hubs,
        "stocked_pairs": stocked_pairs,
        "assignments": assignments,
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

    Mirror of the implementation in run_aligned_fsl_comparison.py. For each open
    hub j (lowest-traffic first), compute the marginal cost of moving each
    assignment at j to the cheapest already-open alternative. If S_lim plus
    j's transport and stocking exceeds the relocation cost, close j.
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

        candidates = sorted(opened, key=lambda j: len(hub_assignment_indices.get(j, [])))
        closed_this_round = False

        for j in candidates:
            indices = hub_assignment_indices.get(j, [])

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
                for idx, new_j in relocate_plan:
                    i, _, k = asg[idx]
                    asg[idx] = (i, new_j, k)
                    stocked.add((new_j, k))
                stocked = {(jj, k) for jj, k in stocked if jj != j}
                opened.discard(j)
                closures += 1
                relocations += len(relocate_plan)
                closed_this_round = True
                break

        if not closed_this_round:
            break

    return {
        "assignments": sorted(asg),
        "stocked_pairs": sorted(stocked),
        "open_hubs": sorted(opened),
        "closures": int(closures),
        "relocations": int(relocations),
    }


def postprocess_solution(
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
        raw_by_pair[(i, k)].append(j)

    current_open = set(raw_open_hubs)
    current_stocked = set(raw_stocked_pairs)

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

        assert chosen is not None
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


def global_feasibility_audit(
    assignments: list[tuple[str, str, str]],
    stocked_pairs: list[tuple[str, str]],
    open_hubs: list[str],
    data: dict[str, Any],
) -> dict[str, int]:
    """Return both structural feasibility and SLA/global distance violations.

    The older standalone QUBO used total_global_violations for structural
    post-processing checks.  solve_fsl_risk_optimization.py uses the same
    phrase for distance/SLA violations.  This aligned version keeps structural
    checks explicit and sets total_global_violations equal to the SLA count so
    the top-line output is directly comparable to Gurobi.
    """
    active_pairs = set(
        (str(r.zip_id), str(r.part_id))
        for r in data["active"][["zip_id", "part_id"]].itertuples(index=False)
    )
    zip_to_hubs = data["zip_to_hubs"]
    candidate_by_zip = {i: {j for j, _ in hubs} for i, hubs in zip_to_hubs.items()}

    assignment_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    invalid_hub = 0
    sla_violations = 0
    penalty_start = float(data["scalar"].get("penalty_start_miles", data["scalar"].get("d_s", 0.0)))
    for i, j, k in assignments:
        assignment_map[(i, k)].append(j)
        if j not in candidate_by_zip.get(i, set()):
            invalid_hub += 1
        if float(data["D"].get((i, j), 0.0)) > penalty_start:
            sla_violations += 1

    missing = 0
    multiple = 0
    for pair in active_pairs:
        count = len(assignment_map.get(pair, []))
        if count == 0:
            missing += 1
        elif count != 1:
            multiple += 1

    extra = sum(1 for pair in assignment_map if pair not in active_pairs)
    stocked_set = set(stocked_pairs)
    open_set = set(open_hubs)
    c2 = sum(1 for _, j, k in assignments if (j, k) not in stocked_set)
    c3 = sum(1 for j, _ in stocked_pairs if j not in open_set)

    by_hub: dict[str, int] = defaultdict(int)
    for j, _ in stocked_pairs:
        by_hub[j] += 1
    l_cap = int(data["scalar"]["L"])
    c4_hubs_over_l = sum(1 for c in by_hub.values() if c > l_cap)
    c4_overflow_units = sum(max(0, c - l_cap) for c in by_hub.values())

    c1 = missing + multiple + invalid_hub + extra
    structural_total = int(c1 + c2 + c3)
    return {
        "c1_violations": int(c1),
        "c1_missing_assignments": int(missing),
        "c1_multiple_assignments": int(multiple),
        "c1_invalid_hub_assignments": int(invalid_hub),
        "c1_extra_assignments": int(extra),
        "c2_violations": int(c2),
        "c3_violations": int(c3),
        "c4_hubs_over_L": int(c4_hubs_over_l),
        "c4_total_overflow_units": int(c4_overflow_units),
        "total_structural_violations": structural_total,
        "structural_violations": structural_total,
        "sla_distance_violations": int(sla_violations),
        "total_global_violations": int(sla_violations),
    }


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
                "final_mult_c1": float(mult.get("c1", 1.0)),
                "final_mult_c2": float(mult.get("c2", 1.0)),
                "final_mult_c3": float(mult.get("c3", 1.0)),
                "final_mult_c4": float(mult.get("c4", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def assignment_rows_dataframe(
    assignments: list[tuple[str, str, str]],
    data: dict[str, Any],
    assignment_sources: dict[tuple[str, str, str], str] | None = None,
) -> pd.DataFrame:
    rows = []
    sources = assignment_sources or {}
    for i, j, k in sorted(assignments):
        c = assignment_cost(i, j, k, data)
        rows.append(
            {
                "zip_id": i,
                "hub_id": j,
                "part_id": k,
                "b_ik": c["b_ik"],
                "d_ij": c["d_ij"],
                "base_miles": c["base_miles"],
                "penalty_start_miles": c["penalty_start_miles"],
                "miles_after_base": c["miles_after_base"],
                "miles_after_penalty_start": c["miles_after_penalty_start"],
                "sla_violation": c["sla_violation"],
                "linehaul_cost": c["linehaul_cost"],
                "distance_cost": c["distance_cost"],
                "distance_penalty_cost": c["distance_penalty_cost"],
                "assignment_cost": c["assignment_cost"],
                "source": sources.get((i, j, k), "raw"),
            }
        )
    return pd.DataFrame(rows)


def stocked_pairs_dataframe(stocked_pairs: list[tuple[str, str]], data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for j, k in sorted(stocked_pairs):
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
    open_hubs: list[str],
    stocked_pairs: list[tuple[str, str]],
    assignments: list[tuple[str, str, str]],
    data: dict[str, Any],
) -> pd.DataFrame:
    open_set = set(open_hubs)
    stock_count: dict[str, int] = defaultdict(int)
    assign_count: dict[str, int] = defaultdict(int)
    for j, _ in stocked_pairs:
        stock_count[j] += 1
    for _, j, _ in assignments:
        assign_count[j] += 1

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


def write_outputs(
    run_dir: Path,
    data: dict[str, Any],
    args: argparse.Namespace,
    results: list[BatchResult],
    raw: dict[str, Any],
    final: dict[str, Any],
    runtime: dict[str, Any],
    stopped_time_limit: bool,
    total_batches: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_cost = compute_solution_cost(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    raw_audit = global_feasibility_audit(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)

    final_cost = compute_solution_cost(final["assignments"], final["stocked_pairs"], final["open_hubs"], data)
    final_audit = global_feasibility_audit(final["assignments"], final["stocked_pairs"], final["open_hubs"], data)

    batch_summary_dataframe(results).to_csv(run_dir / "batch_summary.csv", index=False)

    # Per-batch adaptive metadata: only emitted in within-batch mode so off-mode
    # produces zero new artifacts and batch_summary.csv stays byte-identical.
    if args.adaptive_penalty_mode == "within-batch":
        batch_adaptive_summary_dataframe(results).to_csv(
            run_dir / "batch_adaptive_summary.csv", index=False
        )

    assignment_rows_dataframe(final["assignments"], data, final.get("assignment_sources", {})).to_csv(
        run_dir / "hub_zip_part_pairings.csv", index=False
    )
    hub_status_dataframe(final["open_hubs"], final["stocked_pairs"], final["assignments"], data).to_csv(
        run_dir / "hubs_open_closed.csv", index=False
    )
    stocked_pairs_dataframe(final["stocked_pairs"], data).to_csv(run_dir / "stocked_pairs.csv", index=False)

    # Compatibility aliases with solve_fsl_risk_optimization_aligned.py.
    pd.DataFrame({"hub_id": final["open_hubs"]}).to_csv(run_dir / "open_hubs.csv", index=False)
    pd.DataFrame({"hub_id": sorted(set(data["J"]) - set(final["open_hubs"]))}).to_csv(
        run_dir / "closed_hubs.csv", index=False
    )
    pd.DataFrame(final["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(
        run_dir / "stocked_hub_part_pairs.csv", index=False
    )
    pd.DataFrame(final["assignments"], columns=["zip_id", "hub_id", "part_id"]).to_csv(
        run_dir / "assignments.csv", index=False
    )

    assignment_rows_dataframe(raw["assignments"], data).to_csv(run_dir / "raw_qubo_hub_zip_part_pairings.csv", index=False)
    pd.DataFrame({"hub_id": raw["open_hubs"]}).to_csv(run_dir / "raw_qubo_open_hubs.csv", index=False)
    pd.DataFrame(raw["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(
        run_dir / "raw_qubo_stocked_pairs.csv", index=False
    )

    completed_batches = len(results)
    full_coverage = completed_batches == total_batches and not stopped_time_limit
    summary = {
        "dataset": {
            "dataset_name": data["dataset_name"],
            "dataset_dir": data["dataset_dir"],
            "hubs": int(len(data["J"])),
            "parts": int(len(data["K"])),
            "zips": int(len(data["zips"])),
            "active_demand_pairs": int(len(data["active"])),
            "max_distance": float(data["max_distance"]),
            "top_hubs_per_zip": data["top_hubs_per_zip"],
        },
        "cost_model": {
            "distance_term": "lambda_2 * h_d * b_ik * max(0, d_ij - base_miles)",
            "risk_penalty_term": "lambda_3 * b_ik * max(0, d_ij - penalty_start_miles)",
            "global_violations_definition": "count(assignments where d_ij > penalty_start_miles)",
            "structural_violations_definition": "C1+C2+C3 post-processing feasibility checks",
        },
        "run_config": {
            "max_z_vars_per_batch": int(args.max_z_vars_per_batch),
            "part_batch_size": int(args.part_batch_size),
            "num_reads": int(args.num_reads),
            "num_sweeps": int(args.num_sweeps or 0),
            "max_stages": int(args.max_stages),
            "retry_reads_boost": float(args.retry_reads_boost),
            "penalty_mode": args.penalty_mode,
            "constraint_multiplier": float(args.constraint_multiplier),
            "min_penalty": float(args.min_penalty),
            "qubo_time_limit_seconds": float(args.qubo_time_limit or 0.0),
            "repair_assignments": bool(not args.no_repair_assignments),
            "trim_unused_open_stock": bool(not args.no_trim_unused),
            "hub_prune_enabled": bool(not getattr(args, "no_hub_prune", False)),
            "hub_prune_closures": int(final.get("hub_prune_stats", {}).get("closures", 0)),
            "hub_prune_relocations": int(final.get("hub_prune_stats", {}).get("relocations", 0)),
            "x_empty_penalty_factor": float(getattr(args, "x_empty_penalty_factor", 0.0)),
            "y_overflow_penalty_factor": float(getattr(args, "y_overflow_penalty_factor", 0.0)),
            "seed": int(args.seed) if args.seed is not None else None,
            "adaptive_penalty_mode": args.adaptive_penalty_mode,
            "adaptive_penalty_iterations_max": int(args.adaptive_penalty_iterations),
            "adaptive_penalty_growth": float(args.adaptive_penalty_growth),
        },
        "coverage": {
            "completed_batches": int(completed_batches),
            "total_batches": int(total_batches),
            "full_coverage": bool(full_coverage),
            "stopped_due_to_time_limit": bool(stopped_time_limit),
        },
        "runtime": runtime,
        "memory": {"peak_or_current_rss_mb": float(peak_rss_mb())},
        "raw_qubo_solution": {
            "open_hubs_count": int(len(raw["open_hubs"])),
            "stocked_pairs_count": int(len(raw["stocked_pairs"])),
            "assignments_count": int(len(raw["assignments"])),
            "cost": raw_cost,
            "global_feasibility": raw_audit,
        },
        "final_solution": {
            "open_hubs_count": int(len(final["open_hubs"])),
            "closed_hubs_count": int(len(data["J"]) - len(final["open_hubs"])),
            "stocked_pairs_count": int(len(final["stocked_pairs"])),
            "assignments_count": int(len(final["assignments"])),
            "missing_unrepaired": int(final.get("missing_unrepaired", 0)),
            "cost": final_cost,
            "global_feasibility": final_audit,
        },
        "outputs": {
            "hub_zip_part_pairings_csv": str(run_dir / "hub_zip_part_pairings.csv"),
            "assignments_csv": str(run_dir / "assignments.csv"),
            "hubs_open_closed_csv": str(run_dir / "hubs_open_closed.csv"),
            "open_hubs_csv": str(run_dir / "open_hubs.csv"),
            "closed_hubs_csv": str(run_dir / "closed_hubs.csv"),
            "stocked_pairs_csv": str(run_dir / "stocked_pairs.csv"),
            "stocked_hub_part_pairs_csv": str(run_dir / "stocked_hub_part_pairs.csv"),
            "batch_summary_csv": str(run_dir / "batch_summary.csv"),
            "raw_qubo_pairings_csv": str(run_dir / "raw_qubo_hub_zip_part_pairings.csv"),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Standalone QUBO Solver Summary",
        "=" * 32,
        f"Dataset: {data['dataset_name']}",
        f"Dataset dir: {data['dataset_dir']}",
        f"Output dir: {run_dir}",
        "",
        "Coverage",
        f"  completed batches: {completed_batches}/{total_batches}",
        f"  full coverage: {full_coverage}",
        f"  stopped due to time limit: {stopped_time_limit}",
        "",
        "Final post-processed solution",
        f"  total cost: {money(final_cost['total_cost'])}",
        f"  inventory cost: {money(final_cost['inventory_cost'])}",
        f"  fixed open-hub cost: {money(final_cost['fixed_open_hub_cost'])}",
        f"  overflow storage cost: {money(final_cost['overflow_storage_cost'])}",
        f"  new-hub transfer cost: {money(final_cost['new_hub_transfer_cost'])}",
        f"  assignment transport cost: {money(final_cost['assignment_transport_cost'])}",
        f"  open hubs: {len(final['open_hubs']):,}",
        f"  closed hubs: {len(data['J']) - len(final['open_hubs']):,}",
        f"  stocked hub-part pairs: {len(final['stocked_pairs']):,}",
        f"  hub-zip-part assignments: {len(final['assignments']):,}",
        f"  SLA distance violations: {final_audit['sla_distance_violations']}",
        f"  structural violations: {final_audit['total_structural_violations']}",
        "",
        "Raw decoded QUBO solution",
        f"  raw total cost: {money(raw_cost['total_cost'])}",
        f"  raw open hubs: {len(raw['open_hubs']):,}",
        f"  raw stocked pairs: {len(raw['stocked_pairs']):,}",
        f"  raw assignments: {len(raw['assignments']):,}",
        f"  raw SLA distance violations: {raw_audit['sla_distance_violations']}",
        f"  raw structural violations: {raw_audit['total_structural_violations']}",
        "",
        "Runtime and memory",
        f"  wall time: {runtime['wall_seconds']:.2f}s",
        f"  QUBO build time: {runtime['qubo_build_seconds']:.2f}s",
        f"  QUBO sample time: {runtime['qubo_sample_seconds']:.2f}s",
        f"  sample evaluation time: {runtime['sample_eval_seconds']:.2f}s",
        f"  peak/current RSS: {peak_rss_mb():.2f} MB",
        "",
        "Files written",
        "  hub_zip_part_pairings.csv",
        "  hubs_open_closed.csv",
        "  stocked_pairs.csv",
        "  batch_summary.csv",
        "  raw_qubo_hub_zip_part_pairings.csv",
        "  summary.json",
        "  summary.txt",
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone batched QUBO solver for Dell FSL instances.")
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Folder with hubs/parts/zips/demand/distances/parameters CSVs.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Root output folder.")
    p.add_argument("--run-name", default="", help="Optional run folder name. Default uses timestamp.")
    p.add_argument("--max-distance", type=float, default=-1.0, help="Distance cutoff. <=0 uses parameters.csv max_service_miles.")
    p.add_argument("--top-hubs-per-zip", type=int, default=-1, help="Nearest eligible hubs per zip. -1 keeps all within max distance.")
    p.add_argument("--max-parts-total", type=int, default=-1, help="Limit parts for testing. -1 keeps all.")
    p.add_argument("--part-batch-size", type=int, default=1000, help="Soft part count per batch; Z cap is the hard limiter. Bumped from 200 (Tier 1.1).")
    p.add_argument("--max-z-vars-per-batch", type=int, default=50000, help="Hard cap on Z variables per batch. Bumped from 20000 (Tier 1.1).")
    p.add_argument("--num-reads", type=int, default=100, help="Base OpenJij reads before sqrt(Z/5000) scaling. Bumped from 10 (Tier 1.3).")
    p.add_argument("--num-sweeps", type=int, default=3000, help="OpenJij sweeps per read. Bumped from default (Tier 1.3).")
    p.add_argument("--max-stages", type=int, default=3, help="Retry stages. QUBO is reused; only sampling reads change. Bumped from 2 (Tier 1.3).")
    p.add_argument("--retry-reads-boost", type=float, default=2.0, help="Read multiplier per retry stage.")
    p.add_argument("--penalty-mode", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--min-penalty", type=float, default=50000.0, help="Penalty floor. Bumped from 10000 (Tier 2.5) to dominate hub fixed costs.")
    p.add_argument("--constraint-multiplier", type=float, default=5.0, help="Adaptive penalty multiplier. Lower than old 20 to reduce stiffness.")
    for c in ("c1", "c2", "c3", "c4"):
        p.add_argument(f"--min-penalty-{c}", type=float, default=-1.0)
        p.add_argument(f"--constraint-multiplier-{c}", type=float, default=-1.0)
    p.add_argument("--c4-mode", choices=["off", "auto", "on"], default="auto", help="C4 is inactive for provided low instance.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qubo-time-limit", type=float, default=5400.0, help="Wall-time budget in seconds, checked between batches. 0 disables. Bumped from 2700 (Tier 1.3).")
    p.add_argument("--no-repair-assignments", action="store_true", help="Do not repair missing/multiple assignments in final output.")
    p.add_argument("--no-trim-unused", action="store_true", help="Do not trim unused open hubs / stocked pairs after final assignments.")
    p.add_argument("--no-hub-prune", action="store_true", help="Disable the global hub-pruning post-pass (Tier 1.2).")
    p.add_argument("--hub-prune-max-iterations", type=int, default=10, help="Max passes over open hubs in the hub-prune post-pass.")
    p.add_argument("--x-empty-penalty-factor", type=float, default=0.0, help="Tier 3.8: extra X-diagonal penalty as a multiple of S_lim, partially refunded per stocked Y. Default 0 (disabled).")
    p.add_argument("--y-overflow-penalty-factor", type=float, default=0.0, help="Tier 3.9: linear S_var-style penalty per stocked Y as a multiple of S_var. Default 0 (disabled).")
    p.add_argument("--adaptive-penalty-mode", type=str, default="off",
                   choices=["off", "within-batch"],
                   help="Adaptive penalty strategy. 'off'=current static behavior. "
                        "'within-batch'=iteratively scale violated constraint penalties.")
    p.add_argument("--adaptive-penalty-iterations", type=int, default=5,
                   help="Max adaptive penalty iterations per batch (only used when mode != off).")
    p.add_argument("--adaptive-penalty-growth", type=float, default=1.5,
                   help="Multiplicative growth factor for violated constraint penalties. "
                        "Typical range 1.5-2.0. Lemonge & Barbosa (2004) default ~1.5.")
    p.add_argument("--adaptive-penalty-initial-source", type=str, default="batch-scale",
                   choices=["batch-scale", "external"],
                   help="Where initial penalties come from. 'batch-scale'=current "
                        "penalty_weights() behavior. 'external'=hook for cross-batch "
                        "(not used in v1).")
    return p.parse_args()


def print_header(data: dict[str, Any], batches: list[BatchSpec], args: argparse.Namespace, run_dir: Path) -> None:
    avg_hubs = sum(len(v) for v in data["zip_to_hubs"].values()) / max(1, len(data["zip_to_hubs"]))
    print("\n" + "=" * 70)
    print("STANDALONE MANUAL-QUBO SOLVER - ALIGNED COST MODEL")
    print("=" * 70)
    print(f"  dataset:              {data['dataset_name']}")
    print(f"  dataset dir:          {data['dataset_dir']}")
    print(f"  hubs:                 {len(data['J']):,}")
    print(f"  parts loaded:          {len(data['K']):,}")
    print(f"  zips loaded:           {len(data['zips']):,}")
    print(f"  active demand pairs:   {len(data['active']):,}")
    print(f"  max distance:          {data['max_distance']}")
    th = "all within max distance" if data["top_hubs_per_zip"] is None else f"top {data['top_hubs_per_zip']}"
    print(f"  candidate hubs/zip:    {th} (avg eligible: {avg_hubs:.2f})")
    print(f"  part batch size:       {args.part_batch_size}")
    print(f"  max Z vars/batch:      {args.max_z_vars_per_batch:,}")
    print(f"  total batches:         {len(batches):,}")
    print(f"  penalty mode:          {args.penalty_mode}")
    print(f"  constraint multiplier: {args.constraint_multiplier}")
    print(f"  num reads base:        {args.num_reads}")
    print(f"  max stages:            {args.max_stages}")
    print(f"  QUBO wall time limit:  {args.qubo_time_limit}s")
    if args.adaptive_penalty_mode != "off":
        print(f"  adaptive penalty:      {args.adaptive_penalty_mode}")
        print(f"    max iterations:      {args.adaptive_penalty_iterations}")
        print(f"    growth factor:       {args.adaptive_penalty_growth}")
    print(f"  output folder:         {run_dir}")
    print("=" * 70, flush=True)


def main() -> None:
    args = parse_args()
    if openjij is None:
        raise SystemExit(
            "Missing dependency: openjij. Install it in your active environment with:\n"
            "    pip install openjij\n"
            "Then rerun this script."
        )

    if args.seed is not None and int(args.seed) >= 0:
        random.seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir)
    max_distance_override = None if args.max_distance is None or float(args.max_distance) <= 0 else float(args.max_distance)
    top_hubs = None if args.top_hubs_per_zip is None or int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip)
    max_parts_total = None if args.max_parts_total is None or int(args.max_parts_total) < 0 else int(args.max_parts_total)

    run_name = args.run_name or f"standalone_qubo_{now_stamp()}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    deadline = None
    if args.qubo_time_limit is not None and float(args.qubo_time_limit) > 0:
        deadline = start_wall + float(args.qubo_time_limit)

    print("Loading data...", flush=True)
    data = load_problem_data(
        dataset_dir,
        max_distance_override=max_distance_override,
        top_hubs_per_zip=top_hubs,
        max_parts_total=max_parts_total,
    )

    batches = build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )

    print_header(data, batches, args, run_dir)

    results: list[BatchResult] = []
    stopped_time_limit = False
    for batch in batches:
        if deadline is not None and time.time() >= deadline:
            print(
                "\nQUBO wall-time limit reached before launching next batch. "
                f"Completed {len(results)}/{len(batches)} batches.",
                flush=True,
            )
            stopped_time_limit = True
            break
        result = solve_batch(batch, data["active"], data, args)
        results.append(result)

        # Incremental checkpoint after every batch.
        checkpoint_raw = aggregate_raw_results(results)
        pd.DataFrame(checkpoint_raw["assignments"], columns=["zip_id", "hub_id", "part_id"]).to_csv(
            run_dir / "checkpoint_raw_assignments.csv", index=False
        )
        pd.DataFrame({"hub_id": checkpoint_raw["open_hubs"]}).to_csv(
            run_dir / "checkpoint_raw_open_hubs.csv", index=False
        )
        batch_summary_dataframe(results).to_csv(run_dir / "batch_summary_checkpoint.csv", index=False)

    if not results:
        raise RuntimeError("No batches completed; increase --qubo-time-limit or check the instance.")

    raw = aggregate_raw_results(results)
    final = postprocess_solution(
        raw["assignments"],
        raw["stocked_pairs"],
        raw["open_hubs"],
        data,
        repair_assignments=not bool(args.no_repair_assignments),
        trim_unused=not bool(args.no_trim_unused),
        hub_prune=not bool(getattr(args, "no_hub_prune", False)),
        hub_prune_max_iterations=int(getattr(args, "hub_prune_max_iterations", 10)),
    )

    runtime = {
        "wall_seconds": float(time.time() - start_wall),
        "qubo_build_seconds": float(sum(r.build_seconds for r in results)),
        "qubo_sample_seconds": float(sum(r.sample_seconds for r in results)),
        "sample_eval_seconds": float(sum(r.eval_seconds for r in results)),
        "batch_total_seconds": float(sum(r.total_seconds for r in results)),
    }

    summary = write_outputs(
        run_dir,
        data,
        args,
        results,
        raw,
        final,
        runtime,
        stopped_time_limit=stopped_time_limit,
        total_batches=len(batches),
    )

    final_cost = summary["final_solution"]["cost"]
    final_audit = summary["final_solution"]["global_feasibility"]
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  total cost:             {money(final_cost['total_cost'])}")
    print(f"  open hubs:              {summary['final_solution']['open_hubs_count']:,}")
    print(f"  closed hubs:            {summary['final_solution']['closed_hubs_count']:,}")
    print(f"  stocked hub-part pairs: {summary['final_solution']['stocked_pairs_count']:,}")
    print(f"  hub-zip-part pairings:  {summary['final_solution']['assignments_count']:,}")
    print(f"  SLA distance violations:{final_audit['sla_distance_violations']}")
    print(f"  structural violations:  {final_audit['total_structural_violations']}")
    print(f"  wall time:              {runtime['wall_seconds']:.2f}s")
    print(f"  peak/current memory:    {summary['memory']['peak_or_current_rss_mb']:.2f} MB")
    print("\nOutput files:")
    print(f"  {run_dir / 'hub_zip_part_pairings.csv'}")
    print(f"  {run_dir / 'hubs_open_closed.csv'}")
    print(f"  {run_dir / 'summary.txt'}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
