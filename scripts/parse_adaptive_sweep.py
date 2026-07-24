#!/usr/bin/env python3
"""Parse the adaptive data-scaling sweep logs into a per-batch and per-run summary.
Reads outputs/adaptive_scale_test/sweep_console/p*.log and the combined_summary.json
per run, prints two tables: (1) per-run final result, (2) per-batch adaptive trajectory.
"""
import re, json, glob, os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
# Args: [output_root]  (console logs live in <output_root>/sweep_console)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs/adaptive_scale_test")
if not os.path.isabs(OUT):
    OUT = os.path.join(ROOT, OUT)
CON = os.path.join(OUT, "sweep_console")

batch_hdr = re.compile(r"=== QUBO Batch (\d+) \| (\d+) parts \| ([\d,]+) demand rows \| estimated Z=([\d,]+)")
iter_line = re.compile(r"adaptive iter (\d+) \| violations C1=(\d+) C2=(\d+) C3=(\d+) C4=(\d+)")
feasible = re.compile(r"adaptive feasible at iter (\d+)")
exhausted = re.compile(r"Adaptive exhausted \((\d+) iters\)")

def num(s): return int(s.replace(",", ""))

runs = sorted(glob.glob(os.path.join(CON, "p*.log")))
per_batch_rows = []
per_run = []

for lg in runs:
    run = os.path.basename(lg)[:-4]
    parts_req = int(run[1:])
    txt = open(lg).read().splitlines()
    cur = None
    batches = []
    for line in txt:
        m = batch_hdr.search(line)
        if m:
            if cur: batches.append(cur)
            cur = dict(batch=int(m.group(1)), parts=int(m.group(2)),
                       demand=num(m.group(3)), z=num(m.group(4)),
                       iters=[], outcome="?", conv_iter=None)
            continue
        if cur is None: continue
        mi = iter_line.search(line)
        if mi:
            cur["iters"].append((int(mi.group(1)), num(mi.group(2)), num(mi.group(3)), num(mi.group(4))))
        mf = feasible.search(line)
        if mf:
            cur["outcome"] = "feasible"; cur["conv_iter"] = int(mf.group(1))
        me = exhausted.search(line)
        if me:
            cur["outcome"] = "EXHAUSTED"; cur["conv_iter"] = int(me.group(1))
    if cur: batches.append(cur)

    # final result from combined_summary.json
    js = os.path.join(OUT, run, "combined_summary.json")
    cost = openh = sviol = wall = None
    if os.path.exists(js):
        d = json.load(open(js))
        q = next((r for r in d.get("rows", []) if r.get("solver") == "qubo"), {})
        cost = q.get("total_cost"); openh = q.get("open_hubs")
        sviol = q.get("structural_violations"); wall = q.get("wall_seconds")
    n_exhaust = sum(1 for b in batches if b["outcome"] == "EXHAUSTED")
    max_iter = max((b["conv_iter"] or 0) for b in batches) if batches else 0
    per_run.append(dict(run=run, parts_req=parts_req, nbatch=len(batches),
                        max_conv_iter=max_iter, n_exhausted=n_exhaust,
                        cost=cost, openh=openh, sviol=sviol))
    for b in batches:
        per_batch_rows.append((run, b))

print("\n================ PER-RUN SUMMARY (SA, weak budget 10r/150s/2stage) ================")
print(f"{'run':>6} {'parts':>6} {'batches':>8} {'max_conv_iter':>14} {'exhausted_batches':>18} {'struct_viol':>12} {'open_hubs':>10} {'total_cost':>16}")
for r in per_run:
    cost = f"${r['cost']:,.0f}" if isinstance(r['cost'], (int, float)) else "-"
    print(f"{r['run']:>6} {r['parts_req']:>6} {r['nbatch']:>8} {r['max_conv_iter']:>14} {r['n_exhausted']:>18} "
          f"{str(r['sviol']):>12} {str(r['openh']):>10} {cost:>16}")

print("\n================ PER-BATCH ADAPTIVE TRAJECTORY ================")
print(f"{'run':>6} {'b':>3} {'parts':>5} {'Zvars':>7} {'iters(C1/C2 per iter)':<44} {'outcome':>10} {'@iter':>5}")
for run, b in per_batch_rows:
    traj = "  ".join(f"{it}:{c1}/{c2}" for (it, c1, c2) in b["iters"])
    print(f"{run:>6} {b['batch']:>3} {b['parts']:>5} {b['z']:>7} {traj:<44} {b['outcome']:>10} {str(b['conv_iter']):>5}")
print()
