#!/usr/bin/env python3
"""Prove the memory-attribution mechanism works on THIS machine, before trusting it.

Run this on the VE node BEFORE any solve. Everything the phase attribution
claims rests on two kernel behaviours that cannot be assumed across platforms
and kernels, and both are cheap to test directly:

  1. VmHWM (/proc/self/status) is readable, monotonic, and rises when memory is
     allocated. Without it, per-phase attribution degrades to sampled RSS, which
     can miss a spike between two polls.
  2. getrusage(RUSAGE_SELF).ru_maxrss holds its high-water mark across a free.
     This is what peak_rss_mb() returns and the one number that must not read
     low.

It also runs the nested-phase test (previously only exercised against a
simulated /proc on macOS) and the optional clear_refs probe.

    python3 va_memory_selftest.py

Exit status 0 means every REQUIRED check passed. clear_refs is reported but
never required -- it is an optimisation, and the solver ships with it off.
"""
from __future__ import annotations

import gc
import mmap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_va_fsl_solver as S  # noqa: E402

BAR = "=" * 78
MB = 1024 * 1024
failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str, required: bool = True) -> None:
    tag = "PASS" if ok else ("FAIL" if required else "n/a ")
    print(f"  [{tag}] {name}: {detail}")
    if not ok and required:
        failures.append(f"{name}: {detail}")
    elif not ok:
        notes.append(f"{name}: {detail}")


def hold(mb: int) -> mmap.mmap:
    """mmap, not bytearray: mmap is returned to the OS on close on every platform."""
    buf = mmap.mmap(-1, mb * MB)
    buf.write(b"x" * (mb * MB))
    return buf


def main() -> int:
    print(BAR)
    print("VA HOST MEMORY SELF-TEST")
    print(BAR)
    print(f"  platform    {sys.platform}")
    print(f"  python      {sys.version.splitlines()[0]}")
    print(f"  psutil      {'present' if S.psutil is not None else 'ABSENT (optional)'}")

    print("\n1. Readers available on this machine")
    hwm0 = S.vm_hwm_mb()
    check("VmHWM readable", hwm0 > 0,
          f"{hwm0:,.1f} MB" if hwm0 > 0 else
          "0.0 -- /proc/self/status absent; attribution falls back to sampled RSS",
          required=sys.platform.startswith("linux"))
    check("current RSS readable", S.current_rss_mb() > 0, f"{S.current_rss_mb():,.1f} MB")
    check("peak RSS readable", S.peak_rss_mb() > 0, f"{S.peak_rss_mb():,.1f} MB")
    cg = S.cgroup_peak_mb()
    check("cgroup peak readable", cg > 0,
          f"{cg:,.1f} MB" if cg > 0 else "0.0 -- not in a memory cgroup (fine off SLURM)",
          required=False)
    kids = S.peak_rss_children_mb()
    check("children peak", kids == 0.0,
          f"{kids:,.1f} MB (0.0 expected: nothing here forks)", required=False)

    print("\n2. Peak survives a free  (the core invariant)")
    c0, p0 = S.current_rss_mb(), S.peak_rss_mb()
    bufs = [hold(60) for _ in range(5)]
    c1, p1, h1 = S.current_rss_mb(), S.peak_rss_mb(), S.vm_hwm_mb()
    for b in bufs:
        b.close()
    del bufs
    gc.collect()
    c2, p2, h2 = S.current_rss_mb(), S.peak_rss_mb(), S.vm_hwm_mb()
    print(f"       before  cur={c0:8,.1f}  peak={p0:8,.1f}  hwm={hwm0:8,.1f}")
    print(f"       alloc   cur={c1:8,.1f}  peak={p1:8,.1f}  hwm={h1:8,.1f}")
    print(f"       freed   cur={c2:8,.1f}  peak={p2:8,.1f}  hwm={h2:8,.1f}")
    check("allocation registered", c1 - c0 > 250, f"current rose {c1 - c0:,.1f} MB")
    check("current fell after free", c2 < c1 - 200, f"current fell {c1 - c2:,.1f} MB")
    check("ru_maxrss held its peak", p2 >= p1 - 0.5, f"peak held at {p2:,.1f} MB")
    if hwm0 > 0:
        check("VmHWM rose with allocation", h1 > hwm0 + 200, f"{hwm0:,.1f} -> {h1:,.1f} MB")
        check("VmHWM held after free", h2 >= h1 - 0.5, f"held at {h2:,.1f} MB")

    print("\n3. Nested phase attribution  (against the REAL kernel counters)")
    s = S.MemorySampler(interval_s=0.05).start()
    S.set_active_sampler(s)
    with S.phase("preflight"):
        with S.phase("pyqubo_compile"):
            b = hold(100); b.close(); del b; gc.collect()
    with S.phase("solve"):
        with S.phase("pyqubo_compile"):
            b = hold(400); b.close(); del b; gc.collect()
        with S.phase("va_sample"):
            b = hold(50); b.close(); del b; gc.collect()
        with S.phase("va_sample"):
            b = hold(50); b.close(); del b; gc.collect()
    S.stop_memory_sampler(s)
    rows = {r["phase"]: r for r in s.phase_summary()}
    s.print_report("SELF-TEST PHASE REPORT")

    check("labels nest, not collide",
          "preflight.pyqubo_compile" in rows and "solve.pyqubo_compile" in rows,
          "preflight.pyqubo_compile and solve.pyqubo_compile are distinct rows")
    check("re-entry accumulates", rows.get("solve.va_sample", {}).get("entries") == 2,
          f"solve.va_sample entries={rows.get('solve.va_sample', {}).get('entries')}")
    check("sampled peaks ordered",
          rows["solve.pyqubo_compile"]["rss_peak_mb"] > rows["preflight.pyqubo_compile"]["rss_peak_mb"],
          f"solve {rows['solve.pyqubo_compile']['rss_peak_mb']:,.1f} > "
          f"preflight {rows['preflight.pyqubo_compile']['rss_peak_mb']:,.1f} MB")

    if hwm0 > 0:
        biggest = max(s.phase_summary(), key=lambda r: r["vm_hwm_delta_self_mb"])
        check("HWM attribution names the 400 MB phase",
              biggest["phase"] == "solve.pyqubo_compile",
              f"largest self-delta is {biggest['phase']} "
              f"(+{biggest['vm_hwm_delta_self_mb']:,.1f} MB)")
        total_self = sum(r["vm_hwm_delta_self_mb"] for r in s.phase_summary())
        outer = rows["solve"]["vm_hwm_delta_self_mb"]
        check("no double-counting across nesting", outer < 1.0,
              f"outer 'solve' self-delta {outer:,.1f} MB (children already claimed it)")
        check("self-deltas are a partition", total_self <= s.peak_mb + 1.0,
              f"self-deltas sum to {total_self:,.1f} MB, under the {s.peak_mb:,.1f} MB peak")
    else:
        notes.append("VmHWM unavailable: HWM attribution checks skipped "
                     "(sampled RSS only -- a spike between polls can be missed)")

    print("\n4. clear_refs  (optional; exact per-phase peaks if safe)")
    safe, reason = S.clear_refs_resets_hwm()
    check("clear_refs usable", safe, reason, required=False)
    if not safe:
        print("       -> solver stays on VmHWM deltas. This is the shipped default.")

    print("\n" + BAR)
    if failures:
        print(f"FAILED {len(failures)} required check(s):")
        for f in failures:
            print(f"  - {f}")
        print("Do NOT trust per-phase memory attribution until these pass.")
    else:
        print("All required checks PASSED. Phase attribution is trustworthy here.")
    for n in notes:
        print(f"  note: {n}")
    print(BAR)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
