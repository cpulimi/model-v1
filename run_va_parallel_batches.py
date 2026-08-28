#!/usr/bin/env python3
"""
Parallel batch runner for the VA solver -- the split/solve/merge pipeline.

This is to run_va_fsl_solver.py what run_parallel_batches.py is to the OpenJij
path: it does NOT change the solver math. Every batch is solved by the same
solve_va_batch(), and the merge runs the same aggregate_raw_results() +
postprocess_qubo_solution() the sequential runner does, so the result is what a
single-process run would have produced.

Workflow
--------
1) split  - load data, build batches, COMPILE each one to learn its true
            variable count, check the VA ceiling, write a manifest. No card
            needed, nothing is sampled.
2) solve  - solve ONE batch (--batch-id N) on the card, checkpoint it.
3) merge  - combine every solved batch, post-process (repair / trim /
            hub-prune), write the final va/ outputs. No card needed.

All three modes MUST get identical dataset/QUBO flags and the same
--output-dir / --run-name, so they share a run folder and an identical batch
decomposition. The batching signature is recorded in the manifest and in every
batch checkpoint, and merge refuses to combine mismatched pieces.

ONE CARD, SO SOLVES SERIALIZE
-----------------------------
The OpenJij version runs its whole array side by side, because those batches
only need CPU. sfpga01n exposes /dev/veslot0 and /dev/ve0 -- two device nodes
for ONE physical Vector Engine -- so concurrent array tasks would contend for
the same card. Submit the solve array throttled to one at a time:

    sbatch --array=1-N%1 sbatch_scripts/va_par_solve.sh

va_par_launch.sh does this for you. The win here is not wall clock, it is
checkpointing: a batch that finishes is saved, so a walltime kill costs you the
batch in flight rather than the whole run, and you resubmit only what is
missing.

MERGE DOES NOT NEED THE CARD
----------------------------
Merge is pure CPU: aggregation, repair, hub-prune, cost accounting. It runs on
a normal partition, which keeps the scarce VE node free.

    python run_va_parallel_batches.py --mode split \\
        --dataset-dir instances_low --run-name va_low --output-dir results/va_par

    python run_va_parallel_batches.py --mode solve --batch-id $SLURM_ARRAY_TASK_ID \\
        --dataset-dir instances_low --run-name va_low --output-dir results/va_par

    python run_va_parallel_batches.py --mode merge \\
        --dataset-dir instances_low --run-name va_low --output-dir results/va_par
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import run_va_fsl_solver as solver  # noqa: E402


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def load_data_and_batches(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    """Identical data load + batch decomposition to run_va_solver."""
    data = solver.load_problem_data(
        args.dataset_dir,
        max_service_miles_override=args.max_service_miles,
        penalty_start_miles_override=args.penalty_start_miles,
        top_hubs_per_zip=None if int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip),
        max_parts_total=None if int(args.max_parts_total) < 0 else int(args.max_parts_total),
    )
    batches = solver.build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )
    return data, batches


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_name = args.run_name or f"va_parallel_{Path(args.dataset_dir).name}"
    run_root = Path(args.output_dir).expanduser().resolve() / run_name
    va_dir = run_root / "va"
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir else run_root / "parallel_work"
    )
    return run_root, va_dir, work_dir


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


def check_signature(where: str, expected: dict[str, Any], got: dict[str, Any]) -> None:
    """Refuse to combine pieces that came from different decompositions.

    Batch ids are positional. If part_batch_size differed between solve and
    merge, batch 3 means two different things and the merged solution would be
    silently wrong rather than loudly broken.
    """
    if expected == got:
        return
    diffs = [
        f"      {k}: split/merge={expected.get(k)!r}  vs  {where}={got.get(k)!r}"
        for k in sorted(set(expected) | set(got))
        if expected.get(k) != got.get(k)
    ]
    raise SystemExit(
        f"ABORT: batching signature mismatch against {where}.\n"
        + "\n".join(diffs)
        + "\n  Batch ids are positional, so mixing decompositions corrupts the merge.\n"
        "  Re-run split/solve/merge with identical batching flags."
    )


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def do_split(args: argparse.Namespace, work_dir: Path, run_root: Path) -> int:
    data, batches = load_data_and_batches(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    # Compile every batch to learn its TRUE variable count (Z+Y+X), then check
    # the ceiling. This is the solver's own preflight, run once up front, so a
    # batch can never reach the card oversized.
    print("\nPreflight: compiling every batch QUBO to measure true variable counts...", flush=True)
    plan = solver.build_batch_plan(data, batches, args)
    solver.print_batch_plan(plan, args)
    solver.check_ceiling(plan, args)

    pd.DataFrame(plan).to_csv(work_dir / "va_batch_plan.csv", index=False)

    manifest = {
        "run_name": args.run_name or f"va_parallel_{Path(args.dataset_dir).name}",
        "dataset_name": data["dataset_name"],
        "total_batches": len(batches),
        "batch_ids": [b.batch_id for b in batches],
        "batching_signature": batching_signature(args),
        "max_batch_vars": int(max(p["total_vars"] for p in plan)) if plan else 0,
        "va_max_vars_per_batch": int(args.va_max_vars_per_batch),
        "batches": [
            {
                "batch_id": int(p["batch_id"]),
                "num_rows": int(p["num_rows"]),
                "num_z": int(p["num_z"]),
                "num_y": int(p["num_y"]),
                "num_x": int(p["num_x"]),
                "total_vars": int(p["total_vars"]),
                "dense_matrix_bytes": int(p["dense_matrix_bytes"]),
            }
            for p in plan
        ],
    }
    (work_dir / "batches_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  manifest -> {work_dir / 'batches_manifest.json'}", flush=True)
    print(f"  total batches: {len(batches)}", flush=True)
    print(
        f"\nSubmit the solve array THROTTLED to one at a time -- there is a single VE card:\n"
        f"    sbatch --array=1-{len(batches)}%1 sbatch_scripts/va_par_solve.sh\n"
        f"then --mode merge with the SAME flags.",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def do_solve(args: argparse.Namespace, batch_id: int, work_dir: Path, va_dir: Path) -> int:
    if batch_id is None or int(batch_id) < 1:
        raise SystemExit("--mode solve requires --batch-id N (N >= 1)")

    data, batches = load_data_and_batches(args)
    match = [b for b in batches if b.batch_id == int(batch_id)]
    if not match:
        raise SystemExit(
            f"batch-id {batch_id} not found. This run has {len(batches)} batches "
            f"(ids 1..{len(batches)}). Check that batching flags match --mode split."
        )
    batch = match[0]
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / "batches_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        check_signature("this solve", manifest.get("batching_signature", {}), batching_signature(args))

    print("\n" + "=" * 76, flush=True)
    print(f"VA PARALLEL SOLVE - batch {batch_id}/{len(batches)}", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:      {data['dataset_name']}", flush=True)
    print(f"  demand rows:  {len(batch.row_indices):,}", flush=True)
    print(f"  est Z vars:   {int(batch.estimated_z_vars):,}", flush=True)
    print(f"  work dir:     {work_dir}", flush=True)
    print("=" * 76, flush=True)

    beta_range = solver.parse_beta_range(args.va_beta_range)
    VectorAnnealing = solver.import_vector_annealing()

    tracemalloc.start()
    sampler = solver.start_memory_sampler(args, batch_id=int(batch_id))
    t0 = time.perf_counter()
    result, precision_rows, va_stats = solver.solve_va_batch(
        VectorAnnealing, batch, data, args, beta_range
    )
    solve_seconds = time.perf_counter() - t0
    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solver.stop_memory_sampler(sampler)

    # Trace goes beside the precision audit in the run's va/ directory.
    trace_path = None
    if sampler is not None:
        va_dir.mkdir(parents=True, exist_ok=True)
        trace_path = sampler.to_csv(va_dir / f"va_memory_trace_batch_{int(batch_id):04d}.csv")
        sampler.print_report(f"HOST MEMORY BY PHASE - batch {batch_id}")

    payload = {
        "batch_id": int(batch_id),
        "total_batches": int(len(batches)),
        "batch_result": result,
        "precision_rows": precision_rows,
        "va_stats": va_stats,
        "solve_seconds": float(solve_seconds),
        "memory_accounting_version": solver.MEMORY_ACCOUNTING_VERSION,
        # Was this solve's set/dict iteration order pinned? Merge cross-checks it.
        "reproducibility": solver.reproducibility_snapshot(),
        # Python objects ONLY -- blind to pandas, the pyqubo compiled model, and
        # everything the VectorAnnealing extension allocates. Kept for continuity
        # with profiling_regression_record.md; never the headline figure.
        "python_peak_tracemalloc_mb": float(peak_b) / (1024.0 * 1024.0),
        # The honest host figure: this process's RSS high-water mark.
        "rss_peak_mb": float(solver.peak_rss_mb()),
        # Every host memory scope, separately named. rss_peak_children_mb is
        # expected to read 0.0: nothing here forks (see peak_rss_children_mb).
        "memory": solver.memory_snapshot(),
        "phase_peaks": (sampler.phase_summary() if sampler is not None else []),
        "memory_trace_path": ("" if trace_path is None else str(trace_path)),
        "ve_device": solver.ve_device_report(
            predicted_dense_bytes=int((va_stats or {}).get("dense_matrix_bytes") or 0),
            hbm_gb_override=getattr(args, "ve_hbm_gb", None),
        ),
        "build_seconds": float(result.build_seconds),
        "sample_seconds": float(result.sample_seconds),
        "eval_seconds": float(result.eval_seconds),
        "total_seconds": float(result.total_seconds),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "batching_signature": batching_signature(args),
        "adaptive_penalty_mode": args.adaptive_penalty_mode,
        # Provenance is captured HERE, in the process that actually touched the
        # card. Merge runs on a CPU node and cannot observe any of this.
        "va_provenance": {
            "execution_mode": "local_ve_card",
            "service_client_used": False,
            "module_file": str(getattr(VectorAnnealing, "__file__", "") or ""),
            "hostname": solver.socket.gethostname(),
            "ve_node_number": os.environ.get("VE_NODE_NUMBER"),
            "ve_devices_visible": solver.visible_ve_devices(),
            "ve_card_count": int(solver.ve_card_count()),
            "python_version": sys.version.splitlines()[0],
        },
    }
    out_path = work_dir / f"batch_{int(batch_id):04d}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    viol = result.c1_violations + result.c2_violations + result.c3_violations
    print(
        f"\n  saved batch {batch_id} -> {out_path.name} | solve={solve_seconds:.2f}s "
        f"rss_peak={payload['rss_peak_mb']:,.1f} MB | raw violations={viol}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def do_merge(args: argparse.Namespace, work_dir: Path, va_dir: Path) -> int:
    data, batches = load_data_and_batches(args)
    expected_ids = [b.batch_id for b in batches]

    merge_start = time.perf_counter()
    tracemalloc.start()
    sampler = solver.start_memory_sampler(args)

    payloads: list[dict[str, Any]] = []
    missing: list[int] = []
    with solver.phase("merge.load"):
        for bid in expected_ids:
            pkl = work_dir / f"batch_{int(bid):04d}.pkl"
            if not pkl.is_file():
                missing.append(int(bid))
                continue
            with open(pkl, "rb") as fh:
                payloads.append(pickle.load(fh))

    if missing:
        raise SystemExit(
            f"ABORT: {len(missing)} of {len(expected_ids)} batches are not solved: {missing}\n"
            f"  Looked in {work_dir}\n"
            f"  Solve them and merge again -- already-solved batches are reused:\n"
            f"      sbatch --array={','.join(str(m) for m in missing)}%1 "
            f"sbatch_scripts/va_par_solve.sh"
        )

    payloads.sort(key=lambda p: int(p["batch_id"]))
    mine = batching_signature(args)
    for p in payloads:
        check_signature(f"batch {p['batch_id']}", mine, p.get("batching_signature", {}))

    results = [p["batch_result"] for p in payloads]
    all_precision_rows = [r for p in payloads for r in (p.get("precision_rows") or [])]
    all_va_stats = [p["va_stats"] for p in payloads if p.get("va_stats")]

    va_dir.mkdir(parents=True, exist_ok=True)

    # Identical aggregate + post-process to run_va_solver.
    with solver.phase("merge.aggregate"):
        raw = solver.aggregate_raw_results(results)
    t_post = time.perf_counter()
    with solver.phase("merge.postprocess"):
        final = solver.postprocess_qubo_solution(
            raw["assignments"],
            raw["stocked_pairs"],
            raw["open_hubs"],
            data,
            repair_assignments=not bool(args.no_repair_assignments),
            trim_unused=not bool(args.no_trim_unused),
            hub_prune=not bool(args.no_hub_prune),
            hub_prune_max_iterations=int(args.hub_prune_max_iterations),
        )
    postprocess_seconds = time.perf_counter() - t_post
    solver.active_sampler() and solver.active_sampler().push("merge.write_outputs")

    # Same filenames the sequential runner writes.
    solver.batch_summary_dataframe(results).to_csv(va_dir / "batch_summary.csv", index=False)
    if all_va_stats:
        solver.va_batch_summary_dataframe(results, all_va_stats).to_csv(
            va_dir / "va_batch_summary.csv", index=False
        )
        solver.va_repeat_dataframe(all_va_stats).to_csv(
            va_dir / "va_repeat_summary.csv", index=False
        )
    if args.adaptive_penalty_mode == "within-batch":
        solver.batch_adaptive_summary_dataframe(results).to_csv(
            va_dir / "batch_adaptive_summary.csv", index=False
        )
        solver.adaptive_iteration_log_dataframe(results).to_csv(
            va_dir / "adaptive_iteration_log.csv", index=False
        )
    pd.DataFrame(all_precision_rows).to_csv(va_dir / "va_precision_audit.csv", index=False)
    solver.assignment_rows_dataframe(raw["assignments"], data).to_csv(
        va_dir / "raw_qubo_hub_zip_part_pairings.csv", index=False
    )
    pd.DataFrame({"hub_id": raw["open_hubs"]}).to_csv(va_dir / "raw_qubo_open_hubs.csv", index=False)
    pd.DataFrame(raw["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(
        va_dir / "raw_qubo_stocked_pairs.csv", index=False
    )

    if solver.active_sampler() is not None:
        solver.active_sampler().pop()          # closes merge.write_outputs

    merge_seconds = time.perf_counter() - merge_start
    _, peak_m = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solver.stop_memory_sampler(sampler)
    merge_trace_path = None
    if sampler is not None:
        merge_trace_path = sampler.to_csv(va_dir / "va_memory_trace_merge.csv")

    raw_cost = solver.compute_solution_cost(
        raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data
    )
    raw_audit = solver.global_audit(
        raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data
    )
    precision_summary = solver.print_precision_report(all_precision_rows)

    # Wall clock across the pipeline. Solves ran in separate jobs, so "wall" is
    # their sum (what a sequential run would have taken) -- the array is
    # throttled to one card anyway. Elapsed clock time is a SLURM question.
    solve_total = float(sum(float(p["solve_seconds"]) for p in payloads))
    runtime = {
        "wall_seconds": solve_total + merge_seconds,
        "qubo_build_seconds": float(sum(r.build_seconds for r in results)),
        "qubo_sample_seconds": float(sum(r.sample_seconds for r in results)),
        "sample_eval_seconds": float(sum(r.eval_seconds for r in results)),
        "batch_total_seconds": float(sum(r.total_seconds for r in results)),
        "merge_seconds": float(merge_seconds),
        "postprocess_seconds": float(postprocess_seconds),

        # ------------------------------------------------------------------
        # HOST MEMORY. Device memory lives in extra["va"]["ve_device"] and the
        # two are NEVER summed -- they are separate physical resources.
        #
        # memory_accounting_version 2 == the semantics below. A summary.json
        # WITHOUT this key predates the fix, and its peak_memory_mb is a
        # tracemalloc figure (Python objects only), understating true host
        # memory by however much of the footprint was pandas/pyqubo/extension.
        # ------------------------------------------------------------------
        "memory_accounting_version": solver.MEMORY_ACCOUNTING_VERSION,
        # Hash-order pinning for the MERGE process. Each solve records its own;
        # a repeat study needs every process pinned, so the seeds the batches
        # actually ran under are carried here too rather than assumed to match.
        **solver.reproducibility_snapshot(),
        "batch_python_hash_seeds": sorted({
            str((p.get("reproducibility") or {}).get("python_hash_seed", "<unrecorded>"))
            for p in payloads
        }),

        # peak_memory_mb is the MAXIMUM SINGLE-PROCESS HOST RSS HIGH-WATER MARK
        # reached at any point in the pipeline -- the largest one process ever
        # got. It is NOT a concurrent total and must never be summed or read as
        # "how much memory the job needs at once": the batches ran as separate
        # SLURM array tasks, serialised on one card, so they were never all
        # resident together. Size --mem against cgroup_peak_mb, which covers the
        # whole job step.
        #
        # run_va_fsl_solver.py's single-process runtime dict computes the same
        # quantity from memory_report_mb(), so this field means exactly the same
        # thing in both code paths.
        #
        # It used to be max(per-batch tracemalloc peak), which is a different
        # quantity entirely: on va_20hubs it read 114.5 MB against a true RSS
        # peak of 480.7 MB.
        "peak_memory_mb": float(max(
            max(float(p["rss_peak_mb"]) for p in payloads),
            float(solver.peak_rss_mb()),
        )),
        # An actual CURRENT reading, from the merge process. Previously this
        # held the merge process's tracemalloc PEAK -- mislabelled twice over.
        "current_memory_mb": float(solver.current_rss_mb()),

        "merge_rss_peak_mb": float(solver.peak_rss_mb()),
        "max_batch_rss_peak_mb": float(max(float(p["rss_peak_mb"]) for p in payloads)),

        # tracemalloc retained, under names that say what it is: Python-object
        # allocations only.
        "merge_python_peak_tracemalloc_mb": float(peak_m) / (1024.0 * 1024.0),
        "max_batch_python_peak_tracemalloc_mb": float(
            max(float(p["python_peak_tracemalloc_mb"]) for p in payloads)
        ),

        # Whole-job-step scope: the only figure --mem can be sized against.
        # 0.0 off-cgroup. Batch payloads written before this change carry no
        # "memory" key, hence the .get() chain.
        "max_batch_cgroup_peak_mb": float(max(
            (float((p.get("memory") or {}).get("cgroup_peak_mb", 0.0)) for p in payloads),
            default=0.0,
        )),
        "merge_cgroup_peak_mb": float(solver.cgroup_peak_mb()),
        # Peak RSS of the largest finished CHILD, not a sum. Expected to be 0.0:
        # nothing in this pipeline forks. Non-zero means that changed.
        "max_batch_rss_children_peak_mb": float(max(
            (float((p.get("memory") or {}).get("rss_peak_children_mb", 0.0)) for p in payloads),
            default=0.0,
        )),
        "merge_rss_children_peak_mb": float(solver.peak_rss_children_mb()),
        "pyqubo_express_seconds": float(sum(s["pyqubo_express_seconds"] for s in all_va_stats)),
        "pyqubo_compile_seconds": float(sum(s["pyqubo_compile_seconds"] for s in all_va_stats)),
        "annealing_seconds": float(sum(r.sample_seconds for r in results)),
        "annealing_share_of_wall": (
            float(sum(r.sample_seconds for r in results)) / (solve_total + merge_seconds)
            if (solve_total + merge_seconds) > 0 else 0.0
        ),
        # Which resource binds the batch ceiling is a REGIME, not a constant:
        # device cost is quadratic in vars, host cost is roughly linear in
        # interactions because the host never densifies. Recomputed per run from
        # this run's own measurement rather than frozen from an old fit.
        # Regression over PER-BATCH pairs, both halves from the SAME batch.
        # Previously this paired max(rss_peak) over payloads with max(total_vars)
        # over va_stats -- independent maxima that can come from different
        # batches, describing a batch that never existed.
        "host_device_crossover": solver.host_device_crossover_vars(
            solver.host_rss_points_from_stats([
                {**(s or {}), "rss_peak_mb": float(pay.get("rss_peak_mb") or 0.0)}
                for pay, s in zip(payloads, [p.get("va_stats") for p in payloads])
            ])
        ),
        "phase_peaks_by_batch": {
            str(p["batch_id"]): (p.get("phase_peaks") or []) for p in payloads
        },
        "merge_phase_peaks": (sampler.phase_summary() if sampler is not None else []),
        "merge_memory_trace_path": ("" if merge_trace_path is None else str(merge_trace_path)),
    }

    # Device memory report. Prefer whatever the solve jobs recorded (only they
    # touched the card); fall back to a locally-computed prediction so the block
    # is populated even for batch payloads written before this field existed.
    # The prediction is always present; an OBSERVED number appears only if some
    # source on the node actually produced one.
    merged_ve_device = max(
        (p.get("ve_device") or {} for p in payloads),
        key=lambda d: int(d.get("predicted_dense_bytes") or 0),
        default={},
    )
    if not merged_ve_device:
        merged_ve_device = solver.ve_device_report(
            predicted_dense_bytes=int(max(
                (s["dense_matrix_bytes"] for s in all_va_stats), default=0
            )),
            hbm_gb_override=getattr(args, "ve_hbm_gb", None),
        )

    prov = payloads[0].get("va_provenance", {})
    extra = {
        "completed_batches": int(len(results)),
        "total_batches": int(len(batches)),
        "full_batch_coverage": True,
        "stopped_due_to_time_limit": False,
        "execution": "parallel split/solve/merge",
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
        "va": {
            "engine": "NEC Vector Annealing",
            "manual_reference": solver.VA_MANUAL_REF,
            # Recorded by the solve jobs; merge itself never touches the card.
            **prov,
            # Device memory, kept strictly apart from every host figure above.
            # Carries the analytical prediction always, and an observed number
            # only if this install has any source for one.
            "ve_device": merged_ve_device,
            "objective_scale_enabled": bool(args.enable_objective_scale),
            "min_penalty": float(args.min_penalty),
            "seeded": bool(args.va_seed is not None),
            "seed": (None if args.va_seed is None else int(args.va_seed)),
            "precision": (str(args.va_precision) or "VA default"),
            "vector_mode": str(args.va_vector_mode),
            "num_reads_base": int(args.num_reads),
            "num_sweeps": int(args.num_sweeps),
            "repeats": int(args.va_repeats),
            "adaptive_penalty_mode": str(args.adaptive_penalty_mode),
            "batches_reaching_feasibility": int(
                sum(1 for s in all_va_stats if s.get("adaptive_was_feasible"))
            ),
            "adaptive_exit_reasons": {
                str(s["batch_id"]): str(s.get("adaptive_exit_reason")) for s in all_va_stats
            },
        },
        "precision_audit": precision_summary,
        "performance": {
            "hubs": int(len(data["J"])),
            "zips": int(len(data["zips"])),
            "parts": int(len(data["K"])),
            "active_demand_rows": int(len(data["active"])),
            "batches": int(len(results)),
            "num_z": int(sum(r.num_z for r in results)),
            "num_y": int(sum(r.num_y for r in results)),
            "num_x": int(sum(r.num_x for r in results)),
            "binary_variables": int(sum(r.num_z + r.num_y + r.num_x for r in results)),
            "max_batch_binary_variables": int(max(s["total_vars"] for s in all_va_stats)),
            "qubo_interactions": int(sum(r.interactions for r in results)),
            "matrix_density": float(
                sum(s["matrix_density"] * s["total_vars"] for s in all_va_stats)
                / max(1, sum(s["total_vars"] for s in all_va_stats))
            ),
            "avg_couplings_per_var": float(
                sum(s["avg_couplings_per_var"] * s["total_vars"] for s in all_va_stats)
                / max(1, sum(s["total_vars"] for s in all_va_stats))
            ),
            "max_dense_matrix_bytes": int(max(s["dense_matrix_bytes"] for s in all_va_stats)),
            "qubo_construction_seconds": float(runtime["qubo_build_seconds"]),
            "annealing_seconds": float(runtime["annealing_seconds"]),
            "evaluation_seconds": float(runtime["sample_eval_seconds"]),
            "total_wall_seconds": float(runtime["wall_seconds"]),
            # Host RSS high-water mark, max over batch and merge processes.
            # NOT a concurrent total -- see the runtime dict above.
            "peak_memory_mb": float(runtime["peak_memory_mb"]),
            "rss_peak_mb": float(runtime["max_batch_rss_peak_mb"]),
            "cgroup_peak_mb": float(runtime["max_batch_cgroup_peak_mb"]),
            "raw_cost": float(raw_cost["total_cost"]),
            "raw_structural_violations": int(raw_audit["total_structural_violations"]),
            "raw_c1_violations": int(raw_audit["c1_assignment_violations"]),
            "raw_c2_violations": int(raw_audit["c2_assignment_without_stock"]),
            "raw_c3_violations": int(raw_audit["c3_stock_without_open_hub"]),
            "raw_c4_hubs_over_L": int(raw_audit["c4_hubs_over_L"]),
            "batch_c1_violations": int(sum(r.c1_violations for r in results)),
            "batch_c2_violations": int(sum(r.c2_violations for r in results)),
            "batch_c3_violations": int(sum(r.c3_violations for r in results)),
            "batches_feasible_from_sampler": int(
                sum(1 for s in all_va_stats if s.get("adaptive_was_feasible"))
            ),
            "structurally_feasible_reads": int(
                sum(s["structurally_feasible_reads"] for s in all_va_stats)
            ),
            "total_reads": int(sum(s["va_total_reads"] for s in all_va_stats)),
        },
    }

    summary = solver.write_solution_outputs(
        va_dir,
        solver_name="va",
        data=data,
        assignments=final["assignments"],
        stocked_pairs=final["stocked_pairs"],
        open_hubs=final["open_hubs"],
        runtime=runtime,
        extra=extra,
        assignment_sources=final.get("assignment_sources", {}),
    )
    print(solver.final_results_block(summary, "VA FINAL RESULTS (PARALLEL MERGE)"), flush=True)
    print(
        f"\n  merged {len(results)} batch(es) | solve total {solve_total:,.1f}s "
        f"+ merge {merge_seconds:,.1f}s (postprocess {postprocess_seconds:,.1f}s)",
        flush=True,
    )
    if sampler is not None:
        sampler.print_report("HOST MEMORY BY PHASE - merge")
    print(f"  outputs -> {va_dir}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Split our pipeline flags off, then hand the rest to the solver's parser.

    Delegating keeps this file from drifting: every solver flag, default and
    validation rule applies here automatically, including any added later.
    """
    ap = argparse.ArgumentParser(
        add_help=False,
        description="Split/solve/merge pipeline for the VA solver.",
    )
    ap.add_argument("--mode", choices=["split", "solve", "merge"], required=True)
    ap.add_argument("--batch-id", type=int, default=None,
                    help="Which batch to solve. Required for --mode solve.")
    ap.add_argument("--run-name", default="",
                    help="Run folder name under --output-dir. Must match across all modes.")
    ap.add_argument("--output-dir", default="results/va_parallel")
    ap.add_argument("--work-dir", default="",
                    help="Batch checkpoints. Default <output-dir>/<run-name>/parallel_work.")
    ap.add_argument("-h", "--help", action="store_true",
                    help="Show this help plus the solver's own flags.")
    mine, rest = ap.parse_known_args(argv)

    if mine.help:
        ap.print_help()
        print("\n" + "=" * 76)
        print("Plus every flag of run_va_fsl_solver.py:")
        print("=" * 76)
        solver.parse_args(["--help"])
        raise SystemExit(0)

    # The solver's parser owns everything else, so defaults never diverge.
    solver_args = solver.parse_args(rest)
    return mine, solver_args


def main(argv: list[str] | None = None) -> int:
    mine, args = parse_args(argv)
    args.run_name = mine.run_name
    args.output_dir = mine.output_dir
    args.work_dir = mine.work_dir

    run_root, va_dir, work_dir = resolve_dirs(args)
    # run_root drives the solver's own path logic; keep them consistent.
    args.run_root = str(run_root)

    print(f">>> mode={mine.mode} run_root={run_root}", flush=True)

    if mine.mode == "split":
        return do_split(args, work_dir, run_root)
    if mine.mode == "solve":
        return do_solve(args, mine.batch_id, work_dir, va_dir)
    return do_merge(args, work_dir, va_dir)


if __name__ == "__main__":
    raise SystemExit(main())
