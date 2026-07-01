"""
Standalone Gurobi MILP for the FSL hub/part/ZIP risk optimization instance.

Aligned version: writes machine-readable summary.json/summary.csv in addition
to the original text and solution CSVs so a single wrapper can run Gurobi,
then QUBO, then print a consolidated comparison.

Expected folder layout on your machine:

    C:\\Users\\Akshay Bhatkhande\\Desktop\\Dell Project\\GPT\\solve_fsl_risk_optimization.py
    C:\\Users\\Akshay Bhatkhande\\Desktop\\Dell Project\\GPT\\instances_low\\*.csv
    C:\\Users\\Akshay Bhatkhande\\Desktop\\Dell Project\\GPT\\outputs\\

Run from VS Code terminal:

    cd "C:\\Users\\Akshay Bhatkhande\\Desktop\\Dell Project\\GPT"
    python solve_fsl_risk_optimization.py

Required packages:
    pip install pandas gurobipy

Notes on the model:
- Z is only created for ZIP-hub-part combinations with distance <= max_service_miles.
- The distance file stores raw miles. The dynamic distance term uses max(0, miles - base_miles).
- The SLA/risk penalty uses max(0, miles - penalty_start_miles).
- "global violations" means selected hub-ZIP-part assignments with miles > penalty_start_miles.
- By default, B_j from hubs.csv is NOT enforced as a hard capacity because the PDF defines B as a
  big-M upper bound, and the provided low instance can become infeasible if B_j is treated as a
  hard stocking capacity. Use --enforce-hub-capacity only if you intentionally want that behavior.
"""

from __future__ import annotations

import argparse
import json
import gc
import math
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

WINDOWS_PROJECT_DIR = Path(r"C:\Users\Akshay Bhatkhande\Desktop\Dell Project\GPT")

REQUIRED_FILES = {
    "demand": "demand.csv",
    "distances": "distances.csv",
    "hubs": "hubs.csv",
    "parameters": "parameters.csv",
    "parts": "parts.csv",
    "zips": "zips.csv",
}


@dataclass(frozen=True)
class Scalars:
    C: float
    h_s: float
    h_d: float
    d_s: float
    L: float
    S_lim: float
    S_var: float
    lambda_1: float
    lambda_2: float
    lambda_3: float
    base_miles: float
    penalty_start_miles: float
    max_service_miles: float


def project_dir_default() -> Path:
    if WINDOWS_PROJECT_DIR.exists():
        return WINDOWS_PROJECT_DIR
    return Path(__file__).resolve().parent


def positive_float_or_none(text: Optional[str]) -> Optional[float]:
    if text is None or text == "":
        return None
    val = float(text)
    if val <= 0:
        return None
    return val


def parse_args() -> argparse.Namespace:
    project_dir = project_dir_default()
    default_data_dir = project_dir / "instances_low"
    default_output_dir = project_dir / "outputs"

    parser = argparse.ArgumentParser(
        description="Solve the FSL risk optimization MILP with Gurobi."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Folder containing demand.csv, distances.csv, hubs.csv, parameters.csv, parts.csv, and zips.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Folder for solution CSVs and gurobi.log.",
    )
    parser.add_argument(
        "--time-limit",
        type=positive_float_or_none,
        default=None,
        help="Optional Gurobi time limit in seconds. Omit for no time limit.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.001,
        help="Relative MIP gap. Default: 0.001. Set 0 for exact optimality proof.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Gurobi thread count. Default 0 lets Gurobi choose.",
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Show Gurobi log in the console. By default it is written only to gurobi.log.",
    )
    parser.add_argument(
        "--no-solution-csvs",
        action="store_true",
        help="Do not write assignment/open-hub/stocked-pair CSV files.",
    )
    parser.add_argument(
        "--enforce-hub-capacity",
        action="store_true",
        help="Treat hubs.csv B_j as a hard maximum number of stocked parts at each hub.",
    )
    parser.add_argument(
        "--max-parts-total",
        type=int,
        default=-1,
        help="Optional cap on number of distinct parts to include. Use -1 for all parts.",
    )
    parser.add_argument(
        "--top-hubs-per-zip",
        type=int,
        default=-1,
        help="Optional cap on candidate hubs per ZIP after distance filtering. Use -1 for all eligible hubs.",
    )
    parser.add_argument(
        "--max-service-miles",
        type=float,
        default=None,
        help="Override parameters.csv max_service_miles for eligibility filtering.",
    )
    parser.add_argument(
        "--penalty-start-miles",
        type=float,
        default=None,
        help="Override parameters.csv penalty_start_miles for SLA violation counting and penalty.",
    )
    return parser.parse_args()


def import_required_packages():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing package: pandas. Install with: pip install pandas") from exc

    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing package: gurobipy. Install with: pip install gurobipy") from exc

    return pd, gp, GRB


def resolve_data_dir(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    if data_dir.exists():
        return data_dir

    # Helpful fallback if the CSVs are placed next to the Python file instead of in instances_low.
    script_dir = Path(__file__).resolve().parent
    if all((script_dir / fname).exists() for fname in REQUIRED_FILES.values()):
        return script_dir

    missing = [str(data_dir / fname) for fname in REQUIRED_FILES.values()]
    raise FileNotFoundError(
        "Data folder not found or required CSVs are missing. Expected files:\n  "
        + "\n  ".join(missing)
    )


def require_columns(df, name: str, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_inputs(pd, data_dir: Path, args: argparse.Namespace):
    data_dir = resolve_data_dir(data_dir)
    paths = {key: data_dir / fname for key, fname in REQUIRED_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV files:\n  " + "\n  ".join(missing))

    demand = pd.read_csv(paths["demand"])
    distances = pd.read_csv(paths["distances"])
    hubs = pd.read_csv(paths["hubs"])
    parameters = pd.read_csv(paths["parameters"])
    parts = pd.read_csv(paths["parts"])
    zips = pd.read_csv(paths["zips"])

    require_columns(demand, "demand.csv", ["zip_id", "part_id", "Q_ik", "b_ik"])
    require_columns(distances, "distances.csv", ["zip_id", "hub_id", "d_ij"])
    require_columns(hubs, "hubs.csv", ["hub_id", "T_j", "B_j"])
    require_columns(parameters, "parameters.csv", [
        "C", "h_s", "h_d", "d_s", "L", "S_lim", "S_var",
        "lambda_1", "lambda_2", "lambda_3", "base_miles",
        "penalty_start_miles", "max_service_miles",
    ])
    require_columns(parts, "parts.csv", ["part_id", "P_k"])
    require_columns(zips, "zips.csv", ["zip_id"])

    if parameters.empty:
        raise ValueError("parameters.csv must contain exactly one parameter row.")
    p = parameters.iloc[0].to_dict()

    scalars = Scalars(
        C=float(p["C"]),
        h_s=float(p["h_s"]),
        h_d=float(p["h_d"]),
        d_s=float(p["d_s"]),
        L=float(p["L"]),
        S_lim=float(p["S_lim"]),
        S_var=float(p["S_var"]),
        lambda_1=float(p["lambda_1"]),
        lambda_2=float(p["lambda_2"]),
        lambda_3=float(p["lambda_3"]),
        base_miles=float(p["base_miles"]),
        penalty_start_miles=float(args.penalty_start_miles if args.penalty_start_miles is not None else p["penalty_start_miles"]),
        max_service_miles=float(args.max_service_miles if args.max_service_miles is not None else p["max_service_miles"]),
    )

    # Normalize dtypes and remove rows with no active demand indicator.
    demand = demand.copy()
    demand["Q_ik"] = pd.to_numeric(demand["Q_ik"], errors="raise")
    demand["b_ik"] = pd.to_numeric(demand["b_ik"], errors="raise")
    demand = demand[demand["Q_ik"] > 0].copy()

    distances = distances.copy()
    distances["d_ij"] = pd.to_numeric(distances["d_ij"], errors="raise")

    hubs = hubs.copy()
    hubs["T_j"] = pd.to_numeric(hubs["T_j"], errors="raise").astype(int)
    hubs["B_j"] = pd.to_numeric(hubs["B_j"], errors="raise")

    parts = parts.copy()
    parts["P_k"] = pd.to_numeric(parts["P_k"], errors="raise")

    if args.max_parts_total is not None and int(args.max_parts_total) > 0:
        keep_parts = sorted(parts["part_id"].unique().tolist())[: int(args.max_parts_total)]
        parts = parts[parts["part_id"].isin(keep_parts)].copy()
        demand = demand[demand["part_id"].isin(keep_parts)].copy()
        if demand.empty:
            raise ValueError("--max-parts-total removed all active demand rows.")

    # Basic key checks.
    if demand.duplicated(["zip_id", "part_id"]).any():
        dupes = demand[demand.duplicated(["zip_id", "part_id"], keep=False)].head(10)
        raise ValueError(f"demand.csv has duplicate zip_id/part_id rows. Examples:\n{dupes}")
    if hubs.duplicated("hub_id").any():
        raise ValueError("hubs.csv has duplicate hub_id values.")
    if parts.duplicated("part_id").any():
        raise ValueError("parts.csv has duplicate part_id values.")

    missing_parts = sorted(set(demand["part_id"]) - set(parts["part_id"]))
    if missing_parts:
        raise ValueError(f"parts.csv is missing demanded parts. First examples: {missing_parts[:10]}")
    missing_zips = sorted(set(demand["zip_id"]) - set(zips["zip_id"]))
    if missing_zips:
        raise ValueError(f"zips.csv is missing demanded ZIPs. First examples: {missing_zips[:10]}")

    return demand, distances, hubs, parameters, parts, zips, scalars, data_dir


def prepare_model_data(pd, demand, distances, hubs, parts, scalars: Scalars, args: argparse.Namespace):
    demand = demand.reset_index(drop=True).copy()
    demand["demand_idx"] = demand.index.astype("int64")

    eligible = distances.loc[distances["d_ij"] <= scalars.max_service_miles, ["zip_id", "hub_id", "d_ij"]].copy()
    eligible = eligible.rename(columns={"d_ij": "distance_miles"})

    if eligible.empty:
        raise ValueError("No eligible ZIP-hub pairs after max_service_miles filtering.")

    if args.top_hubs_per_zip is not None and int(args.top_hubs_per_zip) > 0:
        eligible = (
            eligible.sort_values(["zip_id", "distance_miles", "hub_id"])
            .groupby("zip_id", sort=False)
            .head(int(args.top_hubs_per_zip))
            .reset_index(drop=True)
        )

    cand = demand[["demand_idx", "zip_id", "part_id", "b_ik"]].merge(
        eligible,
        on="zip_id",
        how="left",
        validate="many_to_many",
    )

    no_candidate = cand[cand["hub_id"].isna()][["zip_id", "part_id"]].drop_duplicates()
    if not no_candidate.empty:
        examples = no_candidate.head(10).to_string(index=False)
        raise ValueError(
            "Some demanded ZIP-part pairs have no hub within max_service_miles. Examples:\n"
            + examples
        )

    cand = cand.reset_index(drop=True)
    cand["candidate_idx"] = cand.index.astype("int64")

    # Cost terms for selected Z variables.
    miles_after_base = (cand["distance_miles"] - scalars.base_miles).clip(lower=0.0)
    miles_after_penalty_start = (cand["distance_miles"] - scalars.penalty_start_miles).clip(lower=0.0)
    cand["z_cost"] = (
        scalars.lambda_1 * scalars.h_s * cand["b_ik"]
        + scalars.lambda_2 * scalars.h_d * cand["b_ik"] * miles_after_base
        + scalars.lambda_3 * cand["b_ik"] * miles_after_penalty_start
    ).astype(float)
    cand["violation_flag"] = cand["distance_miles"] > scalars.penalty_start_miles

    # Hub and part mappings.
    hubs = hubs.reset_index(drop=True).copy()
    hubs["hub_idx"] = hubs.index.astype("int64")
    hub_to_idx = dict(zip(hubs["hub_id"], hubs["hub_idx"]))
    hub_t = dict(zip(hubs["hub_id"], hubs["T_j"]))
    hub_b = dict(zip(hubs["hub_id"], hubs["B_j"]))

    part_price = dict(zip(parts["part_id"], parts["P_k"]))

    missing_hubs_in_dist = sorted(set(cand["hub_id"]) - set(hub_to_idx))
    if missing_hubs_in_dist:
        raise ValueError(f"distances.csv references hubs not in hubs.csv. Examples: {missing_hubs_in_dist[:10]}")

    hp = cand[["hub_id", "part_id"]].drop_duplicates().sort_values(["hub_id", "part_id"]).reset_index(drop=True)
    hp["hp_idx"] = hp.index.astype("int64")
    hp["hub_idx"] = hp["hub_id"].map(hub_to_idx).astype("int64")
    hp["part_price"] = hp["part_id"].map(part_price).astype(float)
    hp["hub_initial_active"] = hp["hub_id"].map(hub_t).astype(int)
    hp["y_cost"] = hp["part_price"] + (1 - hp["hub_initial_active"]) * scalars.C

    hp_key_to_idx = dict(zip(hp["hub_id"] + "\x00" + hp["part_id"], hp["hp_idx"]))
    cand["hp_idx"] = (cand["hub_id"] + "\x00" + cand["part_id"]).map(hp_key_to_idx).astype("int64")
    cand["hub_idx"] = cand["hub_id"].map(hub_to_idx).astype("int64")

    model_data = {
        "demand": demand,
        "cand": cand,
        "hubs": hubs,
        "hp": hp,
        "hub_to_idx": hub_to_idx,
        "hub_b": hub_b,
        "part_price": part_price,
    }
    return model_data


def add_quicksum(gp, variables):
    # Wrapper to make empty sums safe for Gurobi linear expressions.
    return gp.quicksum(variables)


def build_and_solve(pd, gp, GRB, model_data: dict, scalars: Scalars, args: argparse.Namespace):
    cand = model_data["cand"]
    hubs = model_data["hubs"]
    hp = model_data["hp"]

    n_cand = int(len(cand))
    n_hp = int(len(hp))
    n_hubs = int(len(hubs))
    n_parts_in_model = int(hp["part_id"].nunique())
    big_m_parts = max(1, n_parts_in_model)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = gp.Model("fsl_risk_optimization")
    model.Params.LogToConsole = 1 if args.console_log else 0
    model.Params.LogFile = str(output_dir / "gurobi.log")
    model.Params.MIPGap = max(0.0, float(args.mip_gap))
    if args.time_limit is not None:
        model.Params.TimeLimit = float(args.time_limit)
    if args.threads and args.threads > 0:
        model.Params.Threads = int(args.threads)

    z = model.addVars(range(n_cand), vtype=GRB.BINARY, name="Z")
    y = model.addVars(range(n_hp), vtype=GRB.BINARY, name="Y")
    x = model.addVars(range(n_hubs), vtype=GRB.BINARY, name="X")
    w = model.addVars(range(n_hubs), lb=0.0, vtype=GRB.CONTINUOUS, name="W")

    # Objective.
    z_costs = cand["z_cost"].astype(float).tolist()
    y_costs = hp["y_cost"].astype(float).tolist()

    obj = gp.LinExpr(z_costs, [z[i] for i in range(n_cand)])
    obj.add(gp.LinExpr(y_costs, [y[i] for i in range(n_hp)]))
    obj.add(gp.LinExpr([scalars.S_lim] * n_hubs, [x[i] for i in range(n_hubs)]))
    obj.add(gp.LinExpr([scalars.S_var] * n_hubs, [w[i] for i in range(n_hubs)]))
    model.setObjective(obj, GRB.MINIMIZE)

    print(
        f"Building MILP: Z={n_cand:,}, Y={n_hp:,}, X={n_hubs:,}, "
        f"active ZIP-part pairs={model_data['demand']['demand_idx'].nunique():,}",
        flush=True,
    )

    # Every demanded ZIP-part pair is assigned to exactly one eligible hub.
    # Do not use addConstrs over groupby.indices.values(): those values are numpy arrays,
    # and gurobipy tries to use the generator index as a dict key, which raises
    # "unhashable type: 'numpy.ndarray'". Explicit loops are safer and clearer.
    print("Adding assignment constraints...", flush=True)
    cand_by_demand = cand.groupby("demand_idx", sort=False).indices
    for demand_idx, rows in cand_by_demand.items():
        model.addConstr(
            gp.quicksum(z[int(r)] for r in rows) == 1,
            name=f"assign_one_hub_{int(demand_idx)}",
        )

    # If a ZIP-part is assigned to a hub, the part must be stocked at that hub.
    print("Adding stock-link constraints...", flush=True)
    hp_idx_array = cand["hp_idx"].to_numpy(dtype="int64")
    model.addConstrs(
        (z[i] <= y[int(hp_idx_array[i])] for i in range(n_cand)),
        name="assignment_requires_stock",
    )

    # Hub activation and storage limit constraints.
    print("Adding hub activation/storage constraints...", flush=True)
    hp_by_hub = hp.groupby("hub_idx", sort=False).indices
    hub_b_by_idx = dict(zip(hubs["hub_idx"].astype(int), hubs["B_j"].astype(float)))

    for hub_idx in range(n_hubs):
        hp_rows = [int(r) for r in hp_by_hub.get(hub_idx, [])]
        sum_y = gp.quicksum(y[r] for r in hp_rows)

        if args.enforce_hub_capacity:
            cap = float(hub_b_by_idx.get(hub_idx, big_m_parts))
            if cap <= 0:
                cap = big_m_parts
        else:
            cap = big_m_parts

        # PDF activation constraint: sum_k Y_jk <= B * X_j.
        model.addConstr(sum_y <= cap * x[hub_idx], name=f"activate_hub_{hub_idx}")

        # Stronger semantic constraint: stocked parts require an open hub.
        for r in hp_rows:
            model.addConstr(y[r] <= x[hub_idx], name=f"stock_requires_open_{hub_idx}_{r}")

        # Avoid semantically open hubs with zero stocked parts.
        model.addConstr(x[hub_idx] <= sum_y, name=f"no_empty_open_hub_{hub_idx}")

        # Variable storage over soft limit L.
        model.addConstr(sum_y - scalars.L <= w[hub_idx], name=f"over_storage_limit_{hub_idx}")

    model.update()
    print(
        f"Starting Gurobi optimize: vars={model.NumVars:,}, constraints={model.NumConstrs:,}, "
        f"mip_gap={float(args.mip_gap):g}",
        flush=True,
    )
    model.optimize()

    return model, {"z": z, "y": y, "x": x, "w": w}


def get_status_name(GRB, status_code: int) -> str:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_map.get(status_code, f"STATUS_{status_code}")


def memory_report_mb() -> Tuple[float, float]:
    current, peak = tracemalloc.get_traced_memory()
    current_mb = current / (1024.0 * 1024.0)
    peak_mb = peak / (1024.0 * 1024.0)

    # If psutil is available, use RSS for current process memory. This captures more than tracemalloc.
    try:
        import psutil  # type: ignore

        rss_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        current_mb = max(current_mb, rss_mb)
    except Exception:
        pass

    # On Unix, resource can provide peak RSS. On Windows this module may be absent.
    try:
        import resource  # type: ignore

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_peak_mb = ru_maxrss / (1024.0 * 1024.0)
        else:
            rss_peak_mb = ru_maxrss / 1024.0
        peak_mb = max(peak_mb, rss_peak_mb)
    except Exception:
        pass

    peak_mb = max(peak_mb, current_mb)
    return peak_mb, current_mb


def solution_values(model, vars_dict: dict):
    z = vars_dict["z"]
    y = vars_dict["y"]
    x = vars_dict["x"]
    return {
        "z": model.getAttr("X", z),
        "y": model.getAttr("X", y),
        "x": model.getAttr("X", x),
    }


def summarize_solution(pd, model, vals: dict, model_data: dict, scalars: Scalars, wall_time: float) -> Tuple[dict, object, object, object]:
    cand = model_data["cand"]
    hubs = model_data["hubs"]
    hp = model_data["hp"]

    selected_z = [int(i) for i, val in vals["z"].items() if val > 0.5]
    selected_y = [int(i) for i, val in vals["y"].items() if val > 0.5]
    selected_x = [int(i) for i, val in vals["x"].items() if val > 0.5]

    assignments = cand.loc[selected_z, [
        "zip_id", "part_id", "hub_id", "distance_miles", "b_ik", "violation_flag", "z_cost"
    ]].copy()
    assignments = assignments.rename(columns={"b_ik": "dispatches", "z_cost": "transport_risk_cost"})
    assignments = assignments.sort_values(["zip_id", "part_id", "hub_id"]).reset_index(drop=True)

    stocked_pairs = hp.loc[selected_y, ["hub_id", "part_id", "part_price", "hub_initial_active", "y_cost"]].copy()
    stocked_pairs = stocked_pairs.rename(columns={"y_cost": "inventory_transfer_cost"})
    stocked_pairs = stocked_pairs.sort_values(["hub_id", "part_id"]).reset_index(drop=True)

    open_hubs = hubs.loc[selected_x].copy()
    open_hubs = open_hubs.sort_values("hub_id").reset_index(drop=True)

    closed_hubs = hubs.loc[~hubs.index.isin(selected_x)].copy()
    closed_hubs = closed_hubs.sort_values("hub_id").reset_index(drop=True)

    global_violations = int(assignments["violation_flag"].sum())
    peak_mb, current_mb = memory_report_mb()

    result = {
        "solver": "gurobi_milp_aligned",
        "cost_model": "distance=max(0,d_ij-base_miles); penalty=max(0,d_ij-penalty_start_miles)",
        "total_cost": float(model.ObjVal),
        "open_hubs": int(len(open_hubs)),
        "closed_hubs": int(len(closed_hubs)),
        "stocked_hub_part_pairs": int(len(stocked_pairs)),
        "hub_zip_part_pairings": int(len(assignments)),
        "global_violations": global_violations,
        "wall_time": wall_time,
        "peak_memory_mb": peak_mb,
        "current_memory_mb": current_mb,
    }
    return result, assignments, stocked_pairs, open_hubs, closed_hubs


def final_results_block(result: dict) -> str:
    lines = [
        "=" * 70,
        "FINAL RESULTS",
        "=" * 70,
        f"  total cost:            {result['total_cost']:,.2f}",
        f"  open hubs:             {result['open_hubs']:,}",
        f"  closed hubs:           {result['closed_hubs']:,}",
        f"  stocked hub-part pairs: {result['stocked_hub_part_pairs']:,}",
        f"  hub-zip-part pairings:  {result['hub_zip_part_pairings']:,}",
        f"  global violations:     {result['global_violations']:,}",
        f"  wall time:             {result['wall_time']:,.2f} sec",
        f"  peak/current memory:   {result['peak_memory_mb']:,.1f}/{result['current_memory_mb']:,.1f} MB",
    ]
    return "\n".join(lines)


def write_outputs(output_dir: Path, result: dict, block: str, assignments, stocked_pairs, open_hubs, closed_hubs) -> None:
    import pandas as pd  # imported here so the original friendly dependency check remains intact

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.txt").write_text(block + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(output_dir / "summary.csv", index=False)

    assignments.to_csv(output_dir / "hub_zip_part_pairings.csv", index=False)
    stocked_pairs.to_csv(output_dir / "stocked_hub_part_pairs.csv", index=False)
    open_hubs.to_csv(output_dir / "open_hubs.csv", index=False)
    closed_hubs.to_csv(output_dir / "closed_hubs.csv", index=False)


def main() -> int:
    start = time.perf_counter()
    tracemalloc.start()

    args = parse_args()
    pd, gp, GRB = import_required_packages()

    try:
        demand, distances, hubs, parameters, parts, zips, scalars, data_dir = read_inputs(pd, args.data_dir, args)
        model_data = prepare_model_data(pd, demand, distances, hubs, parts, scalars, args)

        model, vars_dict = build_and_solve(pd, gp, GRB, model_data, scalars, args)
        status_name = get_status_name(GRB, model.Status)

        if model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi ended with status {status_name} and found no feasible solution. "
                "Check max_service_miles and whether --enforce-hub-capacity made the instance infeasible."
            )

        vals = solution_values(model, vars_dict)
        wall_time = time.perf_counter() - start
        result, assignments, stocked_pairs, open_hubs, closed_hubs = summarize_solution(
            pd, model, vals, model_data, scalars, wall_time
        )
        result["status"] = status_name
        try:
            result["mip_gap"] = float(model.MIPGap)
        except Exception:
            result["mip_gap"] = None
        result["data_dir"] = str(data_dir)
        result["base_miles"] = float(scalars.base_miles)
        result["penalty_start_miles"] = float(scalars.penalty_start_miles)
        result["max_service_miles"] = float(scalars.max_service_miles)

        block = final_results_block(result)
        print(block)

        if not args.no_solution_csvs:
            write_outputs(args.output_dir.expanduser(), result, block, assignments, stocked_pairs, open_hubs, closed_hubs)

        # Free Gurobi/Python memory before process exit, useful when running repeatedly from VS Code.
        model.dispose()
        gc.collect()
        return 0

    except Exception as exc:
        peak_mb, current_mb = memory_report_mb()
        print("ERROR: solve failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(f"peak/current memory before exit: {peak_mb:,.1f}/{current_mb:,.1f} MB", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
