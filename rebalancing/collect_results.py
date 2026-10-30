"""Collect every <run>/summary.json (+ score.json) under a directory into one table.

    python -m rebalancing.collect_results results/rebalancing/small [--csv out.csv]

Prints a markdown table; --csv also writes the rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLS = [
    "sites", "parts", "suppliers", "seed", "linking", "vars", "bin", "cons", "status",
    "ObjVal", "ObjBound", "gap", "build_s", "solve_s", "nodes",
    "shipped", "bought", "unmet", "need", "Y_on", "Z_on", "score",
]


def row(run_dir: Path) -> dict:
    s = json.loads((run_dir / "summary.json").read_text())
    sc_path = run_dir / "score.json"
    sc = json.loads(sc_path.read_text()) if sc_path.exists() else None
    st = s.get("solution_stats") or {}
    inst = s["instance"]
    return {
        "sites": inst.get("n_sites"),
        "parts": inst.get("n_parts"),
        "suppliers": inst.get("n_suppliers"),
        "seed": inst.get("seed"),
        "linking": "strong" if s["params"].get("strong_linking") else "base",
        "vars": s["num_vars"]["total"],
        "bin": s["num_vars"]["binary"],
        "cons": s["num_constraints"]["total"],
        "status": s["status"],
        "ObjVal": s["ObjVal"],
        "ObjBound": s["ObjBound"],
        "gap": s["MIPGap"],
        "build_s": round(s["timing_seconds"]["model_build"], 4),
        "solve_s": round(s["Runtime"], 4),
        "nodes": int(s["NodeCount"]),
        "shipped": st.get("units_shipped"),
        "bought": st.get("units_bought"),
        "unmet": st.get("units_unmet"),
        "need": st.get("total_need"),
        "Y_on": st.get("sites_shipping"),
        "Z_on": st.get("sites_receiving"),
        "score": "n/a" if sc is None else ("PASS" if sc["pass"] else "FAIL"),
    }


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}" if abs(v) < 1e6 else f"{v:,.1f}"
    return "" if v is None else str(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    rows = [row(d.parent) for d in Path(args.root).rglob("summary.json")]
    rows.sort(key=lambda r: (r["sites"], r["parts"], r["suppliers"], r["linking"], r["seed"]))
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in COLS) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
