#!/usr/bin/env python3
"""
Parallel batch runner for the QUBO solver (Option 3: parallelize batches).

This file does NOT change the solver math. It reuses the exact functions from
run_aligned_fsl_comparison.py (build_batches, solve_qubo_batch,
aggregate_raw_results, postprocess_qubo_solution, write_solution_outputs), so
each batch's result is bit-identical to a sequential run. The only thing that
changes is *where* batches run: instead of one process solving every batch back
to back, you launch one job per batch and then merge.

Why this preserves quality
---------------------------
- Batches are independent until the final post-process (hub-prune/repair).
- solve_qubo_batch seeds OpenJij as seed + batch_id*1000 + stage. Because
  build_batches is deterministic, the same batch_id always gets the same seed,
  so a batch solved alone == the same batch solved in sequence.
- Merge runs the SAME aggregate + postprocess as run_qubo_solver, once, on the
  union of all batch results.

Net effect: wall clock ~= max(batch_time) + merge, instead of sum(batch_time).
No reduction in num_reads / num_sweeps / stages / penalties.

Workflow
--------
1) split  - list the batches (how many jobs you need) and write a manifest.
2) solve  - solve ONE batch (use --batch-id) and checkpoint it to work-dir.
3) merge  - combine all solved batches, post-process, write final qubo outputs.

All three modes MUST be given identical dataset/QUBO flags, the same
--output-dir and the same --run-name, so they share one run folder and an
identical batch decomposition.

Example (SLURM array, instances_medium, 2 batches)
--------------------------------------------------
    # 0) discover batches
    python run_parallel_batches.py --mode split \
      --dataset-dir instances_medium --run-name med_par \
      --part-batch-size 1000 --max-z-vars-per-batch 600000

    # 1) solve batches in parallel (array index = batch id)
    #    in an sbatch script: #SBATCH --array=1-2
    python run_parallel_batches.py --mode solve --batch-id $SLURM_ARRAY_TASK_ID \
      --dataset-dir instances_medium --run-name med_par \
      --seed 42 --part-batch-size 1000 --max-z-vars-per-batch 600000 \
      --num-reads 30 --num-sweeps 500 --max-stages 2 --retry-reads-boost 2.0 \
      --penalty-mode adaptive --min-penalty 50000.0 --constraint-multiplier 5.0 \
      --constraint-multiplier-c2 3.0 --constraint-multiplier-c3 2.0 --c4-mode auto \
      --adaptive-penalty-mode within-batch --adaptive-penalty-iterations 5 \
      --adaptive-penalty-growth 1.5

    # 2) merge (after all solve jobs finish; use sbatch --dependency=afterok)
    python run_parallel_batches.py --mode merge \
      --dataset-dir instances_medium --run-name med_par \
      --part-batch-size 1000 --max-z-vars-per-batch 600000 \
      --hub-prune-max-iterations 500

Gurobi is intentionally out of scope here; run it separately with
run_aligned_fsl_comparison.py --solver gurobi if you want a comparison.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Reuse the verified orchestrator implementation. Importing is side-effect free
# because run_aligned_fsl_comparison.py guards execution behind __main__.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aligned_fsl_comparison as orch  # noqa: E402


def load_data_and_batches(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    """Identical data load + batch decomposition to run_qubo_solver."""
    data = orch.load_problem_data(
        args.dataset_dir,
        max_service_miles_override=args.max_service_miles,
        penalty_start_miles_override=args.penalty_start_miles,
        top_hubs_per_zip=None if int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip),
        max_parts_total=None if int(args.max_parts_total) < 0 else int(args.max_parts_total),
    )
    batches = orch.build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )
    return data, batches


def resolve_dirs(args: argparse.Namespace, work_dir_override: str) -> tuple[Path, Path, Path]:
    run_name = args.run_name or f"parallel_{Path(args.dataset_dir).name}"
    run_root = Path(args.output_dir).expanduser().resolve() / run_name
    qubo_dir = run_root / "qubo"
    work_dir = Path(work_dir_override).expanduser().resolve() if work_dir_override else run_root / "parallel_work"
    return run_root, qubo_dir, work_dir


def batching_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Flags that must match across split/solve/merge for batch ids to align."""
    return {
        "dataset_dir": str(args.dataset_dir),
        "max_service_miles": args.max_service_miles,
        "penalty_start_miles": args.penalty_start_miles,
        "top_hubs_per_zip": int(args.top_hubs_per_zip),
        "max_parts_total": int(args.max_parts_total),
        "part_batch_size": int(args.part_batch_size),
        "max_z_vars_per_batch": int(args.max_z_vars_per_batch),
    }


def do_split(args: argparse.Namespace, work_dir: Path, run_root: Path) -> int:
    data, batches = load_data_and_batches(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    active = data["active"]
    rows = [
        {
            "batch_id": b.batch_id,
            "num_parts": int(active.iloc[b.row_indices]["part_id"].nunique()),
            "num_rows": len(b.row_indices),
            "estimated_z_vars": int(b.estimated_z_vars),
            "note": b.note,
        }
        for b in batches
    ]
    manifest = {
        "run_name": args.run_name or f"parallel_{Path(args.dataset_dir).name}",
        "total_batches": len(batches),
        "batch_ids": [b.batch_id for b in batches],
        "batching_signature": batching_signature(args),
        "batches": rows,
    }
    (work_dir / "batches_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 76, flush=True)
    print("PARALLEL BATCH SPLIT", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:            {data['dataset_name']}", flush=True)
    print(f"  active demand pairs:{len(data['active']):,}", flush=True)
    print(f"  total batches:      {len(batches)}", flush=True)
    print(f"  work dir:           {work_dir}", flush=True)
    print("  ----------------------------------------------------------------", flush=True)
    print("  batch_id   demand_rows   est_Z_vars   note", flush=True)
    for b in batches:
        print(f"  {b.batch_id:>8}   {len(b.row_indices):>11,}   {int(b.estimated_z_vars):>10,}   {b.note}", flush=True)
    print("=" * 76, flush=True)
    print(f"\nLaunch one solve job per batch id 1..{len(batches)} (e.g. SLURM --array=1-{len(batches)}),", flush=True)
    print("then run --mode merge with the SAME flags.", flush=True)
    return 0


def do_solve(args: argparse.Namespace, batch_id: int, work_dir: Path) -> int:
    if batch_id is None or batch_id < 1:
        raise SystemExit("--mode solve requires --batch-id N (N >= 1)")

    if args.seed is not None and int(args.seed) >= 0:
        random.seed(int(args.seed))

    data, batches = load_data_and_batches(args)
    match = [b for b in batches if b.batch_id == int(batch_id)]
    if not match:
        raise SystemExit(
            f"batch-id {batch_id} not found. This run has {len(batches)} batches "
            f"(ids 1..{len(batches)}). Check that batching flags match --mode split."
        )
    batch = match[0]
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76, flush=True)
    print(f"PARALLEL SOLVE - batch {batch_id}/{len(batches)}", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:      {data['dataset_name']}", flush=True)
    print(f"  demand rows:  {len(batch.row_indices):,}", flush=True)
    print(f"  est Z vars:   {int(batch.estimated_z_vars):,}", flush=True)
    print(f"  work dir:     {work_dir}", flush=True)
    print("=" * 76, flush=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    result = orch.solve_qubo_batch(batch, data["active"], data, args)
    solve_seconds = time.perf_counter() - t0
    current_b, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = float(peak_b) / (1024.0 * 1024.0)

    payload = {
        "batch_id": int(batch_id),
        "total_batches": int(len(batches)),
        "batch_result": result,
        "solve_seconds": float(solve_seconds),
        "peak_memory_mb": float(peak_mb),
        "batching_signature": batching_signature(args),
        "adaptive_penalty_mode": args.adaptive_penalty_mode,
    }
    out_path = work_dir / f"batch_{int(batch_id):04d}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"\n  saved batch {batch_id} -> {out_path.name} | "
        f"solve={solve_seconds:.2f}s peak={peak_mb:,.1f} MB | "
        f"violations={result.c1_violations + result.c2_violations + result.c3_violations}",
        flush=True,
    )
    return 0


def do_merge(args: argparse.Namespace, work_dir: Path, qubo_dir: Path) -> int:
    if args.seed is not None and int(args.seed) >= 0:
        random.seed(int(args.seed))

    data, batches = load_data_and_batches(args)
    expected_ids = [b.batch_id for b in batches]

    merge_start = time.perf_counter()
    tracemalloc.start()

    payloads: list[dict[str, Any]] = []
    for bid in expected_ids:
        pkl = work_dir / f"batch_{int(bid):04d}.pkl"
        if not pkl.is_file():
            raise SystemExit(
                f"Missing solved batch {bid}: {pkl}. Solve all batches "
                f"(ids {expected_ids}) before merging."
            )
        with open(pkl, "rb") as fh:
            payloads.append(pickle.load(fh))

    payloads.sort(key=lambda p: int(p["batch_id"]))
    results = [p["batch_result"] for p in payloads]

    qubo_dir.mkdir(parents=True, exist_ok=True)

    # Identical aggregate + post-process to run_qubo_solver.
    raw = orch.aggregate_raw_results(results)
    final = orch.postprocess_qubo_solution(
        raw["assignments"],
        raw["stocked_pairs"],
        raw["open_hubs"],
        data,
        repair_assignments=not bool(args.no_repair_assignments),
        trim_unused=not bool(args.no_trim_unused),
        hub_prune=not bool(getattr(args, "no_hub_prune", False)),
        hub_prune_max_iterations=int(getattr(args, "hub_prune_max_iterations", 10)),
    )

    # Same output files as run_qubo_solver.
    orch.batch_summary_dataframe(results).to_csv(qubo_dir / "batch_summary.csv", index=False)
    if args.adaptive_penalty_mode == "within-batch":
        orch.batch_adaptive_summary_dataframe(results).to_csv(
            qubo_dir / "batch_adaptive_summary.csv", index=False
        )
    orch.assignment_rows_dataframe(raw["assignments"], data).to_csv(
        qubo_dir / "raw_qubo_hub_zip_part_pairings.csv", index=False
    )
    orch.pd.DataFrame({"hub_id": raw["open_hubs"]}).to_csv(qubo_dir / "raw_qubo_open_hubs.csv", index=False)
    orch.pd.DataFrame(raw["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(
        qubo_dir / "raw_qubo_stocked_pairs.csv", index=False
    )

    merge_seconds = time.perf_counter() - merge_start
    current_m, peak_m = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    merge_peak_mb = float(peak_m) / (1024.0 * 1024.0)

    # Timing: report both the parallel critical path and the sequential-equivalent
    # total so the speedup is visible.
    per_batch_total = [float(r.total_seconds) for r in results]
    per_batch_peak = [float(p.get("peak_memory_mb", 0.0)) for p in payloads]
    sequential_equiv = float(sum(per_batch_total))
    parallel_critical_path = float(max(per_batch_total) if per_batch_total else 0.0) + float(merge_seconds)
    peak_memory_mb = float(max(per_batch_peak + [merge_peak_mb])) if per_batch_peak else float(merge_peak_mb)

    runtime = {
        "wall_seconds": parallel_critical_path,
        "qubo_build_seconds": float(sum(r.build_seconds for r in results)),
        "qubo_sample_seconds": float(sum(r.sample_seconds for r in results)),
        "sample_eval_seconds": float(sum(r.eval_seconds for r in results)),
        "batch_total_seconds": sequential_equiv,
        "peak_memory_mb": peak_memory_mb,
        "current_memory_mb": float(current_m) / (1024.0 * 1024.0),
        "parallel_critical_path_seconds": parallel_critical_path,
        "sequential_equivalent_seconds": sequential_equiv,
        "merge_seconds": float(merge_seconds),
        "per_batch_total_seconds": per_batch_total,
        "per_batch_peak_memory_mb": per_batch_peak,
    }

    raw_cost = orch.compute_solution_cost(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    raw_audit = orch.global_audit(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    extra = {
        "completed_batches": int(len(results)),
        "total_batches": int(len(batches)),
        "full_batch_coverage": bool(len(results) == len(batches)),
        "stopped_due_to_time_limit": False,
        "execution_mode": "parallel_batches",
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
            "hub_prune_enabled": bool(not getattr(args, "no_hub_prune", False)),
            "hub_prune_closures": int(final.get("hub_prune_stats", {}).get("closures", 0)),
            "hub_prune_relocations": int(final.get("hub_prune_stats", {}).get("relocations", 0)),
        },
        "adaptive_penalty_mode": args.adaptive_penalty_mode,
        "adaptive_penalty_iterations_max": int(args.adaptive_penalty_iterations),
        "adaptive_penalty_growth": float(args.adaptive_penalty_growth),
    }

    summary = orch.write_solution_outputs(
        qubo_dir,
        solver_name="qubo",
        data=data,
        assignments=final["assignments"],
        stocked_pairs=final["stocked_pairs"],
        open_hubs=final["open_hubs"],
        runtime=runtime,
        extra=extra,
        assignment_sources=final.get("assignment_sources", {}),
    )
    print(orch.final_results_block(summary, "QUBO FINAL RESULTS (PARALLEL MERGE)"), flush=True)
    print(
        f"\n  parallel critical path: {parallel_critical_path:,.1f}s "
        f"(sequential-equivalent would be {sequential_equiv:,.1f}s)",
        flush=True,
    )
    print(f"  outputs: {qubo_dir}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", choices=["split", "solve", "merge"], required=True)
    pre.add_argument("--batch-id", type=int, default=None, help="Batch id to solve (1-based). Required for --mode solve.")
    pre.add_argument("--work-dir", default="", help="Where per-batch checkpoints live. Default: <output-dir>/<run-name>/parallel_work")
    known, rest = pre.parse_known_args(argv)

    # Reuse the orchestrator's exact argument set/defaults for all solver flags.
    args = orch.parse_args(rest)

    run_root, qubo_dir, work_dir = resolve_dirs(args, known.work_dir)

    if known.mode == "split":
        return do_split(args, work_dir, run_root)
    if known.mode == "solve":
        return do_solve(args, known.batch_id, work_dir)
    if known.mode == "merge":
        return do_merge(args, work_dir, qubo_dir)
    raise SystemExit(f"unknown mode: {known.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
