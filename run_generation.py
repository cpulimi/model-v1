#!/usr/bin/env python3
"""
Headless runner for data_generation_V7.ipynb.

The generation logic is NOT reimplemented here. This script extracts the code
cells from the notebook, execs them in one namespace (so the notebook stays the
single source of truth and cannot drift from this wrapper), then drives
run_batch() itself and QA-checks the result.

    python run_generation.py                      # batch_runs.csv -> outputs/
    python run_generation.py --dry-run            # show the plan, generate nothing

Two things the notebook assumes about Google Colab are handled here:

  1. DEFAULT_ANCHOR_CITY_FILE points at "/content/default_anchor_cities.csv",
     which does not exist off Colab. When that path is missing, this script
     repoints it at ./default_anchor_cities.csv. Rows whose `anchor_city_file`
     column is blank therefore fall back to the local default, exactly as the
     notebook intends. The notebook file itself is never modified.
  2. The notebook's last code cell calls run_batch() at import time. It is
     skipped during exec so the batch runs once, under this script's control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# The base package every generated instance directory must contain. The task
# spec's seven files; the notebook also writes two optional_* extras, which are
# reported but not required.
REQUIRED_FILES = (
    "parameters.csv",
    "hubs.csv",
    "parts.csv",
    "zips.csv",
    "demand.csv",
    "distances.csv",
    "summary_report.csv",
)
OPTIONAL_FILES = (
    "optional_baseline_part_homes.csv",
    "optional_parameter_key.csv",
)

# Any code cell containing this is the notebook's own execution cell; it would
# run the batch a second time with hardcoded paths, so it is skipped.
EXECUTION_CELL_MARKER = "run_batch(BATCH_CSV"


def extract_notebook_source(nb_path: Path) -> tuple[str, int, int]:
    """Concatenate the notebook's code cells into one execable source string.

    Returns (source, cells_included, cells_skipped). `from __future__` imports
    are hoisted implicitly: the first code cell carries one, and any duplicate
    in a later cell is dropped, since a future import is only legal at the top
    of a compilation unit.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    included = skipped = 0

    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if EXECUTION_CELL_MARKER in src:
            skipped += 1
            continue
        if included > 0:
            src = "\n".join(
                line for line in src.splitlines()
                if not line.startswith("from __future__ import")
            )
        chunks.append(f"# ---- notebook cell {index} ----\n{src}")
        included += 1

    return "\n\n".join(chunks) + "\n", included, skipped


def load_notebook_namespace(nb_path: Path) -> dict[str, Any]:
    """Exec the notebook's code cells and hand back the resulting namespace."""
    source, included, skipped = extract_notebook_source(nb_path)
    namespace: dict[str, Any] = {"__name__": "_notebook_", "__file__": str(nb_path)}
    code = compile(source, f"<{nb_path.name}>", "exec")
    exec(code, namespace)  # noqa: S102 - executing the user's own notebook, by design
    print(f"  loaded {included} code cell(s) from {nb_path.name} "
          f"({skipped} execution cell(s) skipped)")
    return namespace


def resolve_default_anchor_file(namespace: dict[str, Any], project_root: Path) -> Path:
    """Point DEFAULT_ANCHOR_CITY_FILE at a file that exists on this machine."""
    declared = str(namespace.get("DEFAULT_ANCHOR_CITY_FILE", ""))
    if declared and Path(declared).is_file():
        print(f"  anchor default:   {declared} (notebook value, exists)")
        return Path(declared)

    local = project_root / "default_anchor_cities.csv"
    if not local.is_file():
        raise SystemExit(
            f"ERROR: the notebook's DEFAULT_ANCHOR_CITY_FILE is {declared!r}, which does\n"
            f"       not exist here, and there is no fallback at {local}.\n"
            "       Every batch row leaves 'anchor_city_file' blank, so generation cannot\n"
            "       proceed without it. Create default_anchor_cities.csv with columns\n"
            "       anchor_id,city,lat,lon,region_code (one row per candidate hub city)."
        )

    namespace["DEFAULT_ANCHOR_CITY_FILE"] = str(local)
    print(f"  anchor default:   {local}")
    print(f"                    (notebook value {declared!r} is a Colab path and is absent here)")
    return local


def describe_batch(batch_csv: Path, anchor_file: Path) -> pd.DataFrame:
    """Print the requested sweep and pre-flight it against the anchor universe."""
    batch = pd.read_csv(batch_csv)
    anchors = pd.read_csv(anchor_file)

    print(f"\n  batch file:       {batch_csv}  ({len(batch)} instance(s))")
    print(f"  anchor universe:  {len(anchors):,} cities\n")
    header = f"    {'instance_name':<16} {'seed':>5} {'n_hubs':>7} {'n_zips':>7} {'n_parts':>8}  anchor_city_file"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for _, row in batch.iterrows():
        anchor = row.get("anchor_city_file")
        shown = "<blank -> default>" if pd.isna(anchor) or not str(anchor).strip() else str(anchor)
        print(f"    {str(row['instance_name']):<16} {int(row['seed']):>5} {int(row['n_hubs']):>7} "
              f"{int(row['n_zips']):>7} {int(row['n_parts']):>8}  {shown}")

    # The notebook samples one hub per anchor city without replacement, so this
    # is a hard limit. Fail here rather than midway through the sweep.
    over = batch[batch["n_hubs"] > len(anchors)]
    if not over.empty:
        raise SystemExit(
            f"\nERROR: {len(over)} row(s) request more hubs than the anchor universe has "
            f"({len(anchors)} cities): {over['instance_name'].tolist()}.\n"
            "       The generator samples one hub per city, so add anchor rows or lower n_hubs."
        )
    return batch


def qa_check_instance(out_dir: Path) -> dict[str, Any]:
    """Confirm one instance directory holds the complete base package."""
    missing = [name for name in REQUIRED_FILES if not (out_dir / name).is_file()]
    empty = [
        name for name in REQUIRED_FILES
        if (out_dir / name).is_file() and (out_dir / name).stat().st_size == 0
    ]
    present_optional = [name for name in OPTIONAL_FILES if (out_dir / name).is_file()]
    return {
        "instance": out_dir.name,
        "missing": missing,
        "empty": empty,
        "optional_present": present_optional,
        "ok": not missing and not empty,
    }


def print_qa_report(out_root: Path, batch: pd.DataFrame) -> bool:
    print("\n" + "=" * 78)
    print("QA 1/2: BASE PACKAGE COMPLETENESS")
    print("=" * 78)
    print(f"  required per instance: {', '.join(REQUIRED_FILES)}\n")

    all_ok = True
    for name in batch["instance_name"].astype(str):
        out_dir = out_root / name
        if not out_dir.is_dir():
            print(f"  [FAIL] {name:<16} directory not created: {out_dir}")
            all_ok = False
            continue
        result = qa_check_instance(out_dir)
        if result["ok"]:
            print(f"  [ OK ] {name:<16} {len(REQUIRED_FILES)}/{len(REQUIRED_FILES)} required files"
                  f"  (+{len(result['optional_present'])} optional)")
        else:
            all_ok = False
            detail = []
            if result["missing"]:
                detail.append(f"missing={result['missing']}")
            if result["empty"]:
                detail.append(f"empty={result['empty']}")
            print(f"  [FAIL] {name:<16} {'; '.join(detail)}")
    return all_ok


def print_comparison_table(out_root: Path) -> pd.DataFrame:
    """Parse batch_summary.csv and render the consolidated scaling comparison."""
    summary_path = out_root / "batch_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"ERROR: {summary_path} was not written; cannot build the comparison table.")
    summary = pd.read_csv(summary_path)

    print("\n" + "=" * 78)
    print("QA 2/2: CONSOLIDATED SCALING COMPARISON  (from batch_summary.csv)")
    print("=" * 78)

    header = (f"  {'instance':<16} {'seed':>5} {'hubs':>6} {'zips':>7} {'parts':>7} "
              f"{'demand_rows':>13} {'distance_rows':>15}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in summary.iterrows():
        print(f"  {str(r['instance_name']):<16} {int(r['seed']):>5} {int(r['n_hubs']):>6,} "
              f"{int(r['n_zips']):>7,} {int(r['n_parts']):>7,} "
              f"{int(r['n_demand_rows']):>13,} {int(r['n_distance_rows']):>15,}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<16} {'':>5} {summary['n_hubs'].sum():>6,} {summary['n_zips'].sum():>7,} "
          f"{'':>7} {summary['n_demand_rows'].sum():>13,} {summary['n_distance_rows'].sum():>15,}")

    # Scaling-consistency assertions the suite exists to guarantee.
    print("\n  scaling consistency:")
    parts_constant = summary["n_parts"].nunique() == 1
    print(f"    part catalog constant across instances: "
          f"{'YES' if parts_constant else 'NO'} ({sorted(summary['n_parts'].unique().tolist())})")
    ratios = (summary["n_zips"] / summary["n_hubs"]).unique()
    print(f"    ZIPs per hub constant:                  "
          f"{'YES' if len(ratios) == 1 else 'NO'} ({[round(float(x), 2) for x in ratios]})")
    exact = bool((summary["n_hubs_generated"] == summary["n_hubs_requested"]).all())
    print(f"    hubs generated == requested:            {'YES' if exact else 'NO'}")
    print(f"    distance rows == hubs x zips:           "
          f"{'YES' if bool((summary['n_distance_rows'] == summary['n_hubs'] * summary['n_zips']).all()) else 'NO'}")
    anchors_used = summary["anchor_city_file"].nunique()
    print(f"    single anchor universe for all rows:    "
          f"{'YES' if anchors_used == 1 else 'NO'} ({summary['anchor_city_file'].iloc[0]})")

    if {"within_max_service_share", "avg_eligible_hubs_per_zip"} <= set(summary.columns):
        print("\n  feasibility (from the notebook's verify_feasibility):")
        for _, r in summary.iterrows():
            print(f"    {str(r['instance_name']):<16} "
                  f"within_max_service={float(r['within_max_service_share']):6.1%}  "
                  f"within_penalty_start={float(r['within_penalty_start_share']):6.1%}  "
                  f"avg_eligible_hubs/zip={float(r['avg_eligible_hubs_per_zip']):6.2f}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", default="data_generation_V7.ipynb", help="Notebook holding the generator.")
    p.add_argument("--batch-csv", default="batch_runs.csv", help="Batch sweep definition.")
    p.add_argument("--out-root", default="outputs", help="Root directory for generated instances.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load the notebook and validate the sweep, but generate nothing.")
    args = p.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    nb_path = Path(args.notebook)
    batch_csv = Path(args.batch_csv)
    out_root = Path(args.out_root)

    for path, label in ((nb_path, "notebook"), (batch_csv, "batch CSV")):
        if not path.is_file():
            raise SystemExit(f"ERROR: {label} not found: {path.resolve()}")

    print("=" * 78)
    print("FSL DIGITAL TWIN - SCALING SUITE GENERATION")
    print("=" * 78)
    namespace = load_notebook_namespace(nb_path)
    anchor_file = resolve_default_anchor_file(namespace, project_root)
    print(f"  output root:      {out_root.resolve()}")

    batch = describe_batch(batch_csv, anchor_file)

    if args.dry_run:
        print("\n--dry-run: notebook loaded and sweep validated. Nothing generated.")
        return 0

    run_batch = namespace["run_batch"]
    print("\n" + "=" * 78)
    print("GENERATING")
    print("=" * 78)
    run_batch(batch_csv, out_root)

    packages_ok = print_qa_report(out_root, batch)
    print_comparison_table(out_root)

    print("\n" + "=" * 78)
    if packages_ok:
        print(f"RESULT: PASS - {len(batch)} instance(s) generated with complete base packages.")
        print(f"        {out_root / 'batch_summary.csv'}")
    else:
        print("RESULT: FAIL - at least one instance is missing part of its base package.")
    print("=" * 78)
    return 0 if packages_ok else 1


if __name__ == "__main__":
    sys.exit(main())
