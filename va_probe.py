#!/usr/bin/env python3
"""
Probe the installed NEC Vector Annealing package and report its real API.

Run this on the VA node BEFORE trusting run_va_fsl_solver.py's defaults, which
were written against the 2022 PoC manual (VApoc_0201). This cluster has
V3.0.0, for which no public documentation exists, so the installed module is
the only authority on what parameters it accepts.

    export PYTHONPATH=/opt/va/V3.0.0/libexec/VectorAnnealing/python:$PYTHONPATH
    python3 va_probe.py

Needs no VE card: it only imports and introspects. If a card is present it
optionally runs a 4-variable smoke test (--smoke).

Paste the whole output back and the solver can be adapted to match.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path


BAR = "=" * 78


def section(title: str) -> None:
    print("\n" + BAR)
    print(title)
    print(BAR)


def show(label: str, value: object) -> None:
    print(f"  {label:<28} {value}")


def probe_environment() -> None:
    section("1. ENVIRONMENT")
    show("python executable", sys.executable)
    show("python version", sys.version.splitlines()[0])
    show("PYTHONPATH", os.environ.get("PYTHONPATH", "<unset>"))
    show("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "<unset>"))
    show("VE_NODE_NUMBER", os.environ.get("VE_NODE_NUMBER", "<unset>"))

    print("\n  VA install directories:")
    found = sorted(glob.glob("/opt/va/*/"))
    for path in found or ["<none under /opt/va>"]:
        print(f"    {path}")

    print("\n  VE devices (each entry is one Vector Engine card):")
    devices = sorted(glob.glob("/dev/veslot*")) + sorted(glob.glob("/dev/ve[0-9]*"))
    for dev in devices or ["<none visible from this node>"]:
        print(f"    {dev}")
    print(f"  -> VE card count: {len(devices)}  "
          f"({'parallel batches possible, one per card' if len(devices) > 1 else 'sequential only' if len(devices) == 1 else 'no card visible here'})")

    print("\n  Documentation shipped with the install:")
    docs: list[str] = []
    for root in found:
        for ext in ("pdf", "md", "txt", "html"):
            docs.extend(glob.glob(f"{root}**/*.{ext}", recursive=True))
    for doc in sorted(docs)[:25] or ["<none found>"]:
        print(f"    {doc}")
    if len(docs) > 25:
        print(f"    ... and {len(docs) - 25} more")


def probe_module() -> object | None:
    section("2. MODULE IMPORT")
    try:
        import VectorAnnealing  # type: ignore
    except Exception as exc:
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}")
        print("\n  Try:")
        print("    export PYTHONPATH=/opt/va/V3.0.0/libexec/VectorAnnealing/python:$PYTHONPATH")
        print("    source /opt/nec/ve/nlc/<ver>/bin/nlcvars.sh")
        print("    export PATH=${PATH}:/opt/nec/ve/bin")
        return None

    show("module file", getattr(VectorAnnealing, "__file__", "<unknown>"))
    for attr in ("__version__", "VERSION", "version", "__doc__"):
        value = getattr(VectorAnnealing, attr, None)
        if value is not None and not callable(value):
            text = str(value).splitlines()[0][:120] if attr == "__doc__" else value
            show(attr, text)

    print("\n  Public names exported by the module:")
    names = sorted(n for n in dir(VectorAnnealing) if not n.startswith("_"))
    for i in range(0, len(names), 3):
        print("    " + "".join(f"{n:<26}" for n in names[i:i + 3]))
    return VectorAnnealing


def probe_signatures(VectorAnnealing: object) -> None:
    section("3. API SIGNATURES  (what the solver must match)")

    for name in ("model", "sampler"):
        fn = getattr(VectorAnnealing, name, None)
        if fn is None:
            print(f"  {name}: NOT PRESENT")
            continue
        try:
            print(f"  VectorAnnealing.{name}{inspect.signature(fn)}")
        except (TypeError, ValueError):
            print(f"  VectorAnnealing.{name}(...)  <signature unavailable>")
        doc = inspect.getdoc(fn)
        if doc:
            for line in doc.splitlines()[:12]:
                print(f"      {line}")

    try:
        sampler = VectorAnnealing.sampler()  # type: ignore[attr-defined]
    except Exception as exc:
        print(f"\n  sampler() could not be instantiated: {type(exc).__name__}: {exc}")
        return

    print()
    try:
        print(f"  sampler.sample{inspect.signature(sampler.sample)}")
    except (TypeError, ValueError):
        print("  sampler.sample(...)  <signature unavailable>")
    doc = inspect.getdoc(sampler.sample)
    if doc:
        print("  --- sample() docstring ---")
        for line in doc.splitlines()[:40]:
            print(f"      {line}")


def probe_capabilities(VectorAnnealing: object) -> None:
    section("4. CAPABILITY CHECKS  (the things that differ across versions)")

    names = set(dir(VectorAnnealing))

    groups = {
        "vector_mode constants": sorted(n for n in names if "VECTOR_MODE" in n.upper()),
        "precision constants": sorted(n for n in names if "PRECISION" in n.upper()),
        "other enum-like constants": sorted(
            n for n in names if n.isupper() and "VECTOR_MODE" not in n and "PRECISION" not in n
        ),
    }
    for label, values in groups.items():
        show(label, ", ".join(values) if values else "<none>")

    print()
    print("  Interpretation:")
    if groups["vector_mode constants"]:
        print("    vector_mode expects MODULE CONSTANTS, not the strings 'SPEED'/'ACCURACY'.")
        print("    -> run_va_fsl_solver.py must be updated to pass these.")
    else:
        print("    No vector_mode constants exported; the PoC-style string form is likely correct.")
    if groups["precision constants"]:
        print("    A precision parameter EXISTS -> single vs double can be tested directly,")
        print("    which is a stronger experiment than the float64 recompute audit.")
    else:
        print("    No precision constants; single precision is presumably fixed.")

    print()
    print("  Parameters accepted by sample() (probed from the signature/doc):")
    try:
        sampler = VectorAnnealing.sampler()  # type: ignore[attr-defined]
        params = inspect.signature(sampler.sample).parameters
        for interesting in (
            "num_reads", "num_results", "num_sweeps", "beta_range", "beta_list",
            "init_spin", "vector_mode", "precision", "dense", "seed",
            "num_threads", "timeout", "ve_num", "Ve_num",
        ):
            present = interesting in params
            mark = "YES" if present else " - "
            default = ""
            if present and params[interesting].default is not inspect.Parameter.empty:
                default = f"  (default {params[interesting].default!r})"
            print(f"    [{mark}] {interesting}{default}")
    except Exception as exc:
        print(f"    could not introspect: {type(exc).__name__}: {exc}")
        print("    (C-extension methods often hide signatures; read the docstring above instead)")

    print()
    print("  NOTE: a 'seed' row reading YES would contradict every NEC doc found so far.")
    print("  If it is YES, tell me -- the no-seed design decision would need revisiting.")

    # V3.0.0-specific: NEC's Vector Annealing Service 3.0 reportedly added
    # Constraint Weight Auto-Tuning, Constraint Priority (inequalities), and
    # higher-order term support. If the on-prem V3 package carries the same
    # features, they overlap with the adaptive penalty loop in
    # run_va_fsl_solver.py -- so look for them explicitly rather than by eye.
    section("4b. V3 FEATURE SEARCH  (auto-tuning / priority / higher-order)")
    keywords = (
        "auto", "tune", "tuning", "weight", "priority", "high_order", "higher",
        "order", "inequality", "penalty", "constraint",
    )
    hits: dict[str, list[str]] = {}
    for name in dir(VectorAnnealing):
        low = name.lower()
        for kw in keywords:
            if kw in low:
                hits.setdefault(kw, []).append(name)
    if hits:
        for kw in sorted(hits):
            print(f"  '{kw}' -> {', '.join(sorted(set(hits[kw])))}")
        print()
        print("  If anything above looks like weight auto-tuning, say so -- it may")
        print("  duplicate the adaptive penalty loop we implemented by hand.")
    else:
        print("  No auto-tuning / priority / higher-order names found in the module.")
        print("  -> our adaptive penalty loop is doing work the engine does not.")

    for fn_name in ("model", "sampler"):
        fn = getattr(VectorAnnealing, fn_name, None)
        if fn is None:
            continue
        try:
            params = list(inspect.signature(fn).parameters)
            print(f"\n  {fn_name}() accepts: {', '.join(params)}")
        except (TypeError, ValueError):
            pass


def smoke_test(VectorAnnealing: object) -> None:
    section("5. SMOKE TEST  (needs a VE card)")
    qubo = {("a", "a"): -1.0, ("b", "b"): -1.0, ("a", "b"): 2.0}
    print("  QUBO: {('a','a'):-1, ('b','b'):-1, ('a','b'):2}  -> optimum energy -1")
    try:
        model = VectorAnnealing.model(qubo, 0.0)  # type: ignore[attr-defined]
        sampler = VectorAnnealing.sampler()  # type: ignore[attr-defined]
        results = sampler.sample(model, num_reads=2, num_results=2)
        results = list(results)
        print(f"  returned {len(results)} result(s)")
        for i, r in enumerate(results):
            print(f"    [{i}] spin={getattr(r, 'spin', None)} energy={getattr(r, 'energy', None)} "
                  f"constraint={getattr(r, 'constraint', None)} time={getattr(r, 'time', None)} "
                  f"memory_usage={getattr(r, 'memory_usage', None)}")
        print("\n  Result attributes:")
        if results:
            print("    " + ", ".join(sorted(a for a in dir(results[0]) if not a.startswith("_"))))
    except Exception as exc:
        print(f"  SMOKE TEST FAILED: {type(exc).__name__}: {exc}")
        print("  (expected if no VE card is visible from this node)")


# ---------------------------------------------------------------------------
# 6. VE device memory introspection
# ---------------------------------------------------------------------------

# Directories and tools worth interrogating. NONE of this is confirmed to exist:
# the point of this section is to find out which of them are real on a given
# install, so that run_va_fsl_solver.ve_device_report() can consume the ones
# that are and stay silent about the ones that are not.
# VE-owned directories are listed in full -- everything in them is relevant.
VE_BIN_DIRS = ("/opt/nec/ve/bin", "/opt/nec/ve/veos/bin", "/opt/nec/ve3/bin",
               "/opt/nec/ve/veos/sbin", "/opt/nec/ve/bin/ve")
# Shared directories are filtered to VE-looking names only; listing all of
# /usr/bin buries the finding in a thousand irrelevant lines.
VE_SHARED_BIN_DIRS = ("/usr/bin", "/usr/local/bin", "/opt/nec/bin")
VE_TOOL_HINTS = (
    "ve-info", "veinfo", "venodeinfo", "veosinfo", "vecmd", "veps", "vestat",
    "ve_exec", "veswap", "vhcall", "vemem", "vemap", "lsve", "nodeinfo",
)
VE_SYSFS_ROOTS = ("/sys/class/ve", "/sys/devices/platform/ve", "/sys/module/ve_drv")
MEMORY_NAME_HINTS = (
    "memory", "mem", "capacity", "device", "resource", "info", "node", "stat", "hbm",
)


# Names that plausibly belong to the Vector Engine toolchain. Deliberately
# narrow: "ve" as a bare prefix also matches verify, version, ved... and on a
# shared /usr/bin that noise buries the real finding.
def _looks_ve(name: str) -> bool:
    low = name.lower()
    if any(h in low for h in VE_TOOL_HINTS):
        return True
    return low.startswith(("ve_", "ve-")) or (
        low.startswith("ve") and len(low) > 2 and low[2].isdigit()
    )


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    """Run a discovery command. Never raises; returns a short transcript."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return out.strip()[:4000] or f"<rc={proc.returncode}, no output>"
    except FileNotFoundError:
        return "<not found>"
    except subprocess.TimeoutExpired:
        return f"<timed out after {timeout}s>"
    except Exception as exc:
        return f"<{type(exc).__name__}: {exc}>"


def probe_device_memory(VectorAnnealing: object | None, out_path: str | None) -> dict:
    """Inventory every possible source of VE device memory on this node.

    Answers, with evidence rather than assumption: does anything on this system
    report the card's HBM capacity or its usage? Everything device-side in the
    solver is currently ANALYTICAL (dense_matrix_bytes), and the only honest way
    to add an observed figure is to first establish what can actually be
    observed. Writes the findings to JSON as well as stdout so they can be
    diffed across installs and quoted without retyping a terminal dump.
    """
    section("6. VE DEVICE MEMORY INTROSPECTION")
    found: dict = {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "<unknown>",
        "executables": {}, "sysfs": {}, "proc": {}, "tools": {},
        "module_attrs": [], "result_attrs": {}, "conclusion": "",
    }

    # --- 6a. executables --------------------------------------------------
    print("\n  6a. Executables under the VE bin directories:")
    for d in VE_BIN_DIRS + VE_SHARED_BIN_DIRS:
        if not os.path.isdir(d):
            if d in VE_BIN_DIRS:
                print(f"    {d}  <absent>")
            continue
        try:
            names = sorted(
                n for n in os.listdir(d)
                if os.access(os.path.join(d, n), os.X_OK)
                and not os.path.isdir(os.path.join(d, n))
            )
        except Exception as exc:
            print(f"    {d}  <unreadable: {exc}>")
            continue
        if d in VE_SHARED_BIN_DIRS:
            names = [n for n in names if _looks_ve(n)]
            if not names:
                continue
        found["executables"][d] = names
        print(f"    {d}  ({len(names)} executables)")
        for n in names:
            print(f"      {n}")

    # --- 6b. run the plausible ones --------------------------------------
    print("\n  6b. Output of tools that might report device memory:")
    candidates: list[str] = []
    for d, names in found["executables"].items():
        for n in names:
            if _looks_ve(n):
                candidates.append(os.path.join(d, n))
    for tool in candidates[:25]:
        for flag in (["--help"], ["-h"], []):
            out = _run([tool] + flag)
            if out and not out.startswith("<not found>"):
                found["tools"][f"{tool} {' '.join(flag)}".strip()] = out
                print(f"\n    $ {tool} {' '.join(flag)}")
                for line in out.splitlines()[:20]:
                    print(f"      {line}")
                break
    if not candidates:
        print("    <no VE-looking executables found on this host>")

    # --- 6c. sysfs --------------------------------------------------------
    print("\n  6c. /sys entries:")
    any_sysfs = False
    for root in VE_SYSFS_ROOTS:
        if not os.path.isdir(root):
            print(f"    {root}  <absent>")
            continue
        any_sysfs = True
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                full = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(full) > 8192:
                        continue
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        val = fh.read().strip()[:200]
                except Exception:
                    continue
                found["sysfs"][full] = val
                flag = "  <-- memory-ish" if any(
                    h in name.lower() for h in MEMORY_NAME_HINTS) else ""
                print(f"    {full} = {val!r}{flag}")
    if not any_sysfs:
        print("    <no VE sysfs tree on this host>")

    # --- 6d. /proc and /dev ----------------------------------------------
    print("\n  6d. /proc and /dev VE entries:")
    for pattern in ("/proc/ve*", "/dev/veslot*", "/dev/ve[0-9]*"):
        for hit in sorted(glob.glob(pattern)):
            info = "<device node>"
            if os.path.isfile(hit):
                try:
                    with open(hit, "r", encoding="utf-8", errors="replace") as fh:
                        info = fh.read().strip()[:300]
                except Exception as exc:
                    info = f"<unreadable: {exc}>"
            found["proc"][hit] = info
            print(f"    {hit} = {info!r}")
    if not found["proc"]:
        print("    <none>")

    # --- 6e. the VectorAnnealing module itself ---------------------------
    print("\n  6e. VectorAnnealing attributes that might expose device memory:")
    if VectorAnnealing is None:
        print("    <module not imported>")
    else:
        hits = sorted(
            n for n in dir(VectorAnnealing)
            if any(h in n.lower() for h in MEMORY_NAME_HINTS)
        )
        found["module_attrs"] = hits
        for n in hits or ["<none matched memory/device/capacity/resource/info>"]:
            val = getattr(VectorAnnealing, n, None) if hits else None
            print(f"    {n}" + (f" = {val!r}" if val is not None and not callable(val) else ""))

        # A result object is the likeliest place a per-run device figure lives.
        print("\n  6f. Attributes of a returned result object (2-variable anneal):")
        try:
            model = VectorAnnealing.model({("a", "a"): -1.0, ("b", "b"): -1.0,
                                           ("a", "b"): 2.0}, 0.0)
            sampler = VectorAnnealing.sampler()
            res = sampler.sample(model, num_reads=1, num_results=1)
            items = list(res) if not isinstance(res, list) else res
            for holder, label in ((res, "sample() return"),
                                  (items[0] if items else None, "result[0]")):
                if holder is None:
                    continue
                attrs = {}
                for n in sorted(a for a in dir(holder) if not a.startswith("_")):
                    try:
                        v = getattr(holder, n)
                    except Exception as exc:
                        v = f"<raised {type(exc).__name__}>"
                    if callable(v):
                        continue
                    attrs[n] = repr(v)[:200]
                found["result_attrs"][label] = attrs
                print(f"\n    {label}:")
                for n, v in attrs.items():
                    flag = "  <-- memory-ish" if any(
                        h in n.lower() for h in MEMORY_NAME_HINTS) else ""
                    print(f"      {n} = {v}{flag}")
        except Exception as exc:
            print(f"    <needs a VE card: {type(exc).__name__}: {exc}>")

    # --- conclusion -------------------------------------------------------
    memoryish = (
        [k for k in found["sysfs"] if any(h in k.lower() for h in ("mem", "capacity", "hbm"))]
        + [f"{lbl}.{a}" for lbl, d in found["result_attrs"].items()
           for a in d if "mem" in a.lower()]
    )
    found["conclusion"] = (
        "device memory sources found: " + ", ".join(memoryish)
        if memoryish else
        "NO source of VE device memory found on this host. Device memory can only "
        "be PREDICTED (dense_matrix_bytes), never measured, on this install."
    )
    print(f"\n  CONCLUSION: {found['conclusion']}")

    if out_path:
        try:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(found, indent=2, default=str), encoding="utf-8")
            print(f"\n  written -> {path}")
        except Exception as exc:
            print(f"\n  could not write {out_path}: {type(exc).__name__}: {exc}")
    return found


def main() -> int:
    p = argparse.ArgumentParser(description="Probe the installed NEC Vector Annealing package.")
    p.add_argument("--smoke", action="store_true", help="Also run a 2-variable anneal (needs a VE card).")
    p.add_argument("--device-memory", action="store_true",
                   help="Inventory every source of VE device memory on this node "
                        "(executables, sysfs, /proc, module and result attributes) "
                        "and write the findings to JSON.")
    p.add_argument("--device-probe-out", default="va_device_probe.json",
                   help="Where to write the device inventory JSON "
                        "(default: va_device_probe.json in the run directory).")
    args = p.parse_args()

    print(BAR)
    print("NEC VECTOR ANNEALING INSTALL PROBE")
    print(BAR)

    probe_environment()
    module = probe_module()
    if module is None:
        print("\nStopping: module could not be imported.")
        return 1
    probe_signatures(module)
    probe_capabilities(module)
    if args.smoke:
        smoke_test(module)
    if args.device_memory:
        probe_device_memory(module, args.device_probe_out)

    print("\n" + BAR)
    print("Done. Paste this entire output back to adapt run_va_fsl_solver.py.")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
