#!/usr/bin/env python3
"""
Turn a va_benchmark.py CSV into a readable HTML report.

The CSV is the record; this is how you read it. Every number, chart and verdict
on the page is derived from the CSV at generation time -- nothing is hardcoded --
so the same command works on whatever the SOL run produces.

    # on the VE node, after the benchmark
    python va_benchmark.py --mode full --csv results/va_perf.csv
    python va_report.py results/va_perf.csv -o results/va_perf.html

    # or in one step
    python va_benchmark.py --mode full --csv results/va_perf.csv --html results/va_perf.html

Sections adapt to what the run actually measured: a preflight CSV has no
annealing or solution-quality columns, so those sections are replaced by a note
saying why they are absent. A full CSV from the VE card fills them in.

Open the .html in any browser, or publish it. It is a single self-contained file.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def growth_exponent(x: list[float], y: list[float]) -> float:
    """k in y ~ x^k, by least squares on log-log."""
    pts = [(math.log(a), math.log(b)) for a, b in zip(x, y)
           if a and b and a > 0 and b > 0 and a == a and b == b]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else float("nan")


def classify(k: float) -> tuple[str, str, str]:
    """(css class, short label, plain meaning)."""
    if k != k:
        return "flat", "n/a", "not enough data points"
    if k < -0.3:
        return "down", "Shrinking", f"falls as the problem grows (~1/n^{abs(k):.1f})"
    if k < 0.3:
        return "ok", "Flat", "roughly constant regardless of size"
    if k < 0.75:
        return "ok", "Sub-linear", "grows slower than the hub count"
    if k < 1.3:
        return "ok", "Linear", "doubles when hubs double"
    if k < 1.7:
        return "warn", "Super-linear", "grows faster than hubs, cheaper than squared"
    if k < 2.3:
        return "warn", "Quadratic", "quadruples when hubs double"
    return "crit", "Explosive", "grows faster than squared; this binds first"


def human_bytes(n: float) -> str:
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "-"
    if x != x:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:,.1f} {unit}"
        x /= 1024.0
    return f"{x:,.1f} TiB"


def has(df: pd.DataFrame, col: str) -> bool:
    """True when the column exists and holds at least one real value."""
    return col in df.columns and bool(df[col].notna().any())


def num(v: Any, fmt: str = ",.0f", blank: str = "—") -> str:
    try:
        f = float(v)
        if f != f:
            return blank
        return format(f, fmt)
    except (TypeError, ValueError):
        return blank


# ---------------------------------------------------------------------------
# page assembly
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--ground:#fbfbfd;--surface:#fff;--panel:#f4f5f8;--line:#e0e3ea;
--line-strong:#c6cbd6;--ink:#14161c;--ink-2:#545b6b;--ink-3:#868d9d;
--s-size:#2a78d6;--s-mem:#eb6834;--s-density:#1baf7a;--critical:#c92a2a;
--s-size-soft:#2a78d61f;--s-mem-soft:#eb68341f;--s-density-soft:#1baf7a1f;--crit-soft:#c92a2a1a;
--radius:3px}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--ground:#101216;--surface:#171a20;--panel:#1e222a;--line:#2b303a;--line-strong:#3d4450;
--ink:#eef0f4;--ink-2:#a7aebd;--ink-3:#757c8b;
--s-size:#3987e5;--s-mem:#d95926;--s-density:#199e70;--critical:#f26d6d;
--s-size-soft:#3987e526;--s-mem-soft:#d9592626;--s-density-soft:#199e7026;--crit-soft:#f26d6d26}}
:root[data-theme=dark]{color-scheme:dark;
--ground:#101216;--surface:#171a20;--panel:#1e222a;--line:#2b303a;--line-strong:#3d4450;
--ink:#eef0f4;--ink-2:#a7aebd;--ink-3:#757c8b;
--s-size:#3987e5;--s-mem:#d95926;--s-density:#199e70;--critical:#f26d6d;
--s-size-soft:#3987e526;--s-mem-soft:#d9592626;--s-density-soft:#199e7026;--crit-soft:#f26d6d26}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px}
h1,h2,h3{font-family:Archivo,"IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em;background:var(--panel);padding:1px 5px;border-radius:2px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;letter-spacing:.13em;
text-transform:uppercase;color:var(--ink-3)}
header{border-bottom:1px solid var(--line);background:var(--surface);padding:44px 0 34px}
header h1{font-size:clamp(30px,4.4vw,44px);font-weight:700;letter-spacing:-.021em;line-height:1.08;
margin:12px 0 14px;max-width:22ch}
.standfirst{font-size:17px;color:var(--ink-2);max-width:64ch;margin:0}
.runmeta{display:flex;flex-wrap:wrap;gap:0 30px;margin-top:26px;padding-top:20px;border-top:1px solid var(--line)}
.runmeta div{display:flex;flex-direction:column;gap:2px}
.runmeta dt{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3)}
.runmeta dd{margin:0;font-size:13.5px;font-weight:500;font-family:"IBM Plex Mono",monospace}
section{padding:52px 0;border-bottom:1px solid var(--line)}
section:last-of-type{border-bottom:0}
.sechead{margin-bottom:26px}
.sechead h2{font-size:21px;font-weight:600;letter-spacing:-.012em;margin:7px 0 8px}
.sechead p{margin:0;color:var(--ink-2);max-width:70ch;font-size:14.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(224px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}
.tile{background:var(--surface);padding:20px 22px 22px;display:flex;flex-direction:column;gap:5px}
.tile .label{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.tile .value{font-family:Archivo,sans-serif;font-size:31px;font-weight:700;letter-spacing:-.02em;line-height:1.05;
font-variant-numeric:tabular-nums}
.tile .sub{font-size:13px;color:var(--ink-2)}
.tile.ok .value{color:var(--s-size)}.tile.warn .value{color:var(--s-mem)}
.tile.down .value{color:var(--s-density)}.tile.crit .value{color:var(--critical)}
.chip{display:inline-flex;align-items:center;gap:6px;align-self:flex-start;margin-top:3px;
font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
padding:3px 8px;border-radius:2px;border:1px solid}
.chip.ok{color:var(--s-size);border-color:var(--s-size);background:var(--s-size-soft)}
.chip.warn{color:var(--s-mem);border-color:var(--s-mem);background:var(--s-mem-soft)}
.chip.down{color:var(--s-density);border-color:var(--s-density);background:var(--s-density-soft)}
.chip.crit{color:var(--critical);border-color:var(--critical);background:var(--crit-soft)}
.chip.flat{color:var(--ink-3);border-color:var(--line-strong);background:var(--panel)}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:22px 24px 18px}
figcaption{font-size:13px;color:var(--ink-2);margin-top:14px;padding-top:13px;border-top:1px solid var(--line)}
.figtitle{font-family:Archivo,sans-serif;font-size:15.5px;font-weight:600;margin:0 0 3px}
.figsub{font-size:13px;color:var(--ink-3);margin:0 0 16px}
.legend{display:flex;flex-wrap:wrap;gap:18px;margin-bottom:14px}
.legend span{display:inline-flex;align-items:center;gap:7px;font-size:13px;color:var(--ink-2)}
.swatch{width:11px;height:11px;border-radius:2px;flex:none}
.chartbox{position:relative;overflow-x:auto}
svg{display:block;max-width:100%;height:auto}
.grid-line{stroke:var(--line);stroke-width:1}.axis-line{stroke:var(--line-strong);stroke-width:1}
.tick{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--ink-3);font-variant-numeric:tabular-nums}
.axis-title{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--ink-3);letter-spacing:.09em;text-transform:uppercase}
.dlabel{font-family:"IBM Plex Mono",monospace;font-size:11.5px;font-weight:600;font-variant-numeric:tabular-nums}
.hit{fill:transparent;cursor:pointer}
.tooltip{position:absolute;pointer-events:none;opacity:0;transition:opacity .12s ease;background:var(--surface);
color:var(--ink);border:1px solid var(--line-strong);border-radius:var(--radius);padding:9px 11px;font-size:12.5px;
line-height:1.5;box-shadow:0 6px 22px rgba(0,0,0,.13);z-index:5;min-width:150px}
.tooltip .tt-h{font-family:Archivo,sans-serif;font-weight:600;font-size:13px;margin-bottom:5px}
.tooltip .tt-r{display:flex;justify-content:space-between;gap:16px;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.tooltip .tt-r span:first-child{color:var(--ink-2)}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media (max-width:860px){.twocol{grid-template-columns:1fr}}
.tablebox{overflow-x:auto;border:1px solid var(--line);border-radius:var(--radius);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 13px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:600;letter-spacing:.07em;
text-transform:uppercase;color:var(--ink-3);background:var(--panel);position:sticky;top:0}
tbody td{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tbody td:first-child{font-family:"IBM Plex Sans",sans-serif;font-weight:500}
tbody tr:last-child td{border-bottom:0}
tfoot td{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-2);background:var(--panel);
border-top:1px solid var(--line-strong)}
.proj{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}
.prow{display:grid;grid-template-columns:96px 116px 1fr 130px;align-items:center;gap:16px;background:var(--surface);padding:11px 18px}
.prow.head{background:var(--panel)}
.prow.head>*{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.prow .hubs{font-family:Archivo,sans-serif;font-weight:600;font-size:15px}
.prow .vars{font-family:"IBM Plex Mono",monospace;font-size:13px;font-variant-numeric:tabular-nums}
.measured{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--ink-3);letter-spacing:.06em;font-weight:400}
.bar{position:relative;height:22px;background:var(--panel);border-radius:2px;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;display:block;border-radius:2px}
.bar .ceil{position:absolute;top:-3px;bottom:-3px;width:2px}
.verdict{font-family:"IBM Plex Mono",monospace;font-size:11.5px;font-weight:600;text-align:right}
.verdict.fits{color:var(--s-size)}.verdict.over{color:var(--critical)}
.callout{border:1px solid var(--line);border-left:3px solid var(--s-mem);background:var(--surface);
border-radius:var(--radius);padding:20px 24px}
.callout.pending{border-left-color:var(--ink-3)}
.callout.good{border-left-color:var(--s-size)}
.callout h3{font-size:16px;font-weight:600;margin-bottom:8px}
.callout p{margin:0 0 10px;color:var(--ink-2);max-width:74ch}
.callout p:last-child{margin-bottom:0}
ul.plain{margin:0;padding-left:18px;color:var(--ink-2)}
ul.plain li{margin-bottom:7px}ul.plain li::marker{color:var(--ink-3)}
strong{color:var(--ink);font-weight:600}
footer{padding:30px 0 46px;color:var(--ink-3);font-size:12.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

CHART_JS = r"""
const NS="http://www.w3.org/2000/svg";
const el=(n,a={})=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const fmt=n=>Number(n).toLocaleString("en-US");
const bytes=b=>{const u=["B","KiB","MiB","GiB","TiB"];let x=Number(b),i=0;
  while(x>=1024&&i<u.length-1){x/=1024;i++;}return x.toFixed(1)+" "+u[i];};
function tipOn(box,tip,node,html){
  node.addEventListener("mouseenter",()=>{tip.innerHTML=html;tip.style.opacity="1";});
  node.addEventListener("mousemove",ev=>{const r=box.getBoundingClientRect();
    let x=ev.clientX-r.left+14,y=ev.clientY-r.top-12;
    if(x+tip.offsetWidth>r.width)x=ev.clientX-r.left-tip.offsetWidth-14;
    tip.style.left=Math.max(0,x)+"px";tip.style.top=Math.max(0,y)+"px";});
  node.addEventListener("mouseleave",()=>{tip.style.opacity="0";});
}

/* indexed multi-series growth on a log y axis */
function growthChart(svgId,boxId,tipId,series){
  const svg=document.getElementById(svgId);if(!svg)return;
  const box=document.getElementById(boxId),tip=document.getElementById(tipId);
  const W=780,H=360,M={t:18,r:80,b:46,l:60},iw=W-M.l-M.r,ih=H-M.t-M.b;
  const hubs=DATA.map(d=>d.hubs);
  const lo=Math.min(...hubs),hi=Math.max(...hubs);
  const x0=Math.log10(lo*0.9),x1=Math.log10(hi*1.15);
  const X=v=>M.l+(Math.log10(v)-x0)/(x1-x0)*iw;
  let maxMult=1;
  series.forEach(s=>{const b=DATA[0][s.key];
    DATA.forEach(d=>{if(b&&d[s.key])maxMult=Math.max(maxMult,d[s.key]/b);});});
  const topTick=Math.pow(10,Math.ceil(Math.log10(maxMult*1.35)));
  const y0=Math.log10(0.85),y1=Math.log10(topTick);
  const Y=v=>M.t+ih-(Math.log10(Math.max(v,0.86))-y0)/(y1-y0)*ih;
  const ticks=[];for(let e=0;Math.pow(10,e)<=topTick+1e-9;e++){
    const p=Math.pow(10,e);[1,2,5].forEach(m=>{if(p*m<=topTick+1e-9&&p*m>=1)ticks.push(p*m);});}
  ticks.forEach(t=>{svg.appendChild(el("line",{x1:M.l,x2:M.l+iw,y1:Y(t),y2:Y(t),class:"grid-line"}));
    const lb=el("text",{x:M.l-10,y:Y(t)+3.5,class:"tick","text-anchor":"end"});
    lb.textContent=fmt(t)+"×";svg.appendChild(lb);});
  svg.appendChild(el("line",{x1:M.l,x2:M.l+iw,y1:M.t+ih,y2:M.t+ih,class:"axis-line"}));
  DATA.forEach(d=>{const lb=el("text",{x:X(d.hubs),y:M.t+ih+20,class:"tick","text-anchor":"middle"});
    lb.textContent=d.hubs;svg.appendChild(lb);});
  const xt=el("text",{x:M.l+iw/2,y:H-8,class:"axis-title","text-anchor":"middle"});
  xt.textContent="Hubs";svg.appendChild(xt);
  const yt=el("text",{x:15,y:M.t+ih/2,class:"axis-title","text-anchor":"middle",
    transform:"rotate(-90 15 "+(M.t+ih/2)+")"});
  yt.textContent="Growth vs smallest run";svg.appendChild(yt);
  series.forEach(s=>{const b=DATA[0][s.key];if(!b)return;
    const pts=DATA.map(d=>[X(d.hubs),Y(d[s.key]/b)]);
    svg.appendChild(el("polyline",{points:pts.map(p=>p.join(",")).join(" "),fill:"none",
      stroke:s.color,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
    pts.forEach((p,i)=>{
      svg.appendChild(el("circle",{cx:p[0],cy:p[1],r:4.5,fill:s.color,stroke:"var(--surface)","stroke-width":2}));
      const hit=el("circle",{cx:p[0],cy:p[1],r:16,class:"hit"});svg.appendChild(hit);
      const d=DATA[i];
      tipOn(box,tip,hit,'<div class="tt-h">'+d.hubs+' hubs · '+s.label+'</div>'+
        '<div class="tt-r"><span>Value</span><span>'+s.fmtv(d)+'</span></div>'+
        '<div class="tt-r"><span>Growth</span><span>'+(d[s.key]/b).toFixed(1)+'×</span></div>');});
    const last=pts[pts.length-1],d=DATA[DATA.length-1];
    const lab=el("text",{x:last[0]+11,y:last[1]+4,class:"dlabel",fill:s.color});
    lab.textContent=(d[s.key]/b).toFixed(1)+"×";svg.appendChild(lab);});
}

/* single-series bars with a value label above each */
function barChart(svgId,boxId,tipId,key,color,fmtLabel,tipRows,W){
  const svg=document.getElementById(svgId);if(!svg)return;
  const box=document.getElementById(boxId),tip=document.getElementById(tipId);
  const H=300,M={t:32,r:20,b:44,l:66},iw=(W||480)-M.l-M.r,ih=H-M.t-M.b;
  const bw=iw/DATA.length;
  const mx=Math.max(...DATA.map(d=>d[key]||0))||1;
  [0,.25,.5,.75,1].forEach(f=>{const y=M.t+ih-f*ih;
    svg.appendChild(el("line",{x1:M.l,x2:M.l+iw,y1:y,y2:y,class:"grid-line"}));
    const lb=el("text",{x:M.l-9,y:y+3.5,class:"tick","text-anchor":"end"});
    lb.textContent=fmtLabel(mx*f);svg.appendChild(lb);});
  svg.appendChild(el("line",{x1:M.l,x2:M.l+iw,y1:M.t+ih,y2:M.t+ih,class:"axis-line"}));
  DATA.forEach((d,i)=>{const v=d[key]||0,h=Math.max(2,(v/mx)*ih);
    const x=M.l+i*bw+7,w=Math.max(8,bw-16);
    svg.appendChild(el("rect",{x,y:M.t+ih-h,width:w,height:h,rx:3,fill:color}));
    const t=el("text",{x:x+w/2,y:M.t+ih-h-7,class:"dlabel",fill:"var(--ink-2)","text-anchor":"middle"});
    t.textContent=fmtLabel(v);svg.appendChild(t);
    const lb=el("text",{x:x+w/2,y:M.t+ih+19,class:"tick","text-anchor":"middle"});
    lb.textContent=d.hubs;svg.appendChild(lb);
    const hit=el("rect",{x,y:M.t,width:w,height:ih,class:"hit"});svg.appendChild(hit);
    tipOn(box,tip,hit,'<div class="tt-h">'+d.hubs+' hubs</div>'+tipRows(d));});
  const xt=el("text",{x:M.l+iw/2,y:H-8,class:"axis-title","text-anchor":"middle"});
  xt.textContent="Hubs";svg.appendChild(xt);
}
"""


def build_report(df: pd.DataFrame, source: str) -> str:
    df = df[df.get("status", pd.Series(["OK"] * len(df))) == "OK"].copy()
    if df.empty:
        raise SystemExit("ERROR: no rows with status OK in the CSV; nothing to report.")
    df = df.sort_values("hubs").reset_index(drop=True)
    mode = str(df["mode"].iloc[0]) if "mode" in df.columns else "unknown"
    full = mode == "full"
    hubs = [float(v) for v in df["hubs"]]

    # ---- chart data, straight from the CSV -------------------------------
    chart_cols = ["instance", "hubs", "binary_vars", "interactions", "matrix_density",
                  "dense_bytes", "sparse_bytes", "dense_waste_factor",
                  "construction_seconds", "annealing_seconds", "total_wall_seconds",
                  "rss_peak_mb", "raw_cost", "final_cost", "raw_structural_violations",
                  "feasible_read_share", "demand_rows"]
    data_js = []
    for _, r in df.iterrows():
        rec: dict[str, Any] = {}
        for c in chart_cols:
            v = r.get(c)
            if c == "instance":
                rec[c] = str(v)
            else:
                try:
                    f = float(v)
                    rec[c] = None if f != f else f
                except (TypeError, ValueError):
                    rec[c] = None
        data_js.append(rec)

    # ---- verdict tiles ---------------------------------------------------
    tile_specs = [
        ("binary_vars", "Binary variables", lambda a, b: f"{num(a)} → {num(b)}"),
        ("dense_bytes", "Device memory", lambda a, b: f"{human_bytes(a)} → {human_bytes(b)}"),
        ("matrix_density", "Matrix density", lambda a, b: f"{num(a, '.4%')} → {num(b, '.4%')}"),
    ]
    if full and has(df, "annealing_seconds"):
        tile_specs.append(("annealing_seconds", "Annealing time",
                           lambda a, b: f"{num(a, ',.2f')}s → {num(b, ',.2f')}s"))
    else:
        tile_specs.append(("construction_seconds", "QUBO construction",
                           lambda a, b: f"{num(a, ',.2f')}s → {num(b, ',.2f')}s"))

    tiles = []
    exponents: dict[str, float] = {}
    for col, label, span in tile_specs:
        if not has(df, col):
            continue
        ys = [float(v) for v in df[col]]
        k = growth_exponent(hubs, ys)
        exponents[col] = k
        cls, short, meaning = classify(k)
        tiles.append(
            f'<div class="tile {cls}"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">k = {num(k, "+.2f")}</div>'
            f'<div class="chip {cls}">{html.escape(short)}</div>'
            f'<div class="sub">{html.escape(span(ys[0], ys[-1]))}. {html.escape(meaning.capitalize())}.</div></div>'
        )

    # ---- trend table -----------------------------------------------------
    trend_metrics = [
        ("demand_rows", "Demand rows", ",.0f"),
        ("binary_vars", "Binary variables", ",.0f"),
        ("interactions", "QUBO interactions", ",.0f"),
        ("matrix_density", "Matrix density", ".5%"),
        ("couplings_per_var", "Couplings per variable", ".2f"),
        ("dense_bytes", "Device memory (dense)", None),
        ("construction_seconds", "QUBO construction (s)", ",.2f"),
        ("annealing_seconds", "Annealing time (s)", ",.2f"),
        ("total_wall_seconds", "Total runtime (s)", ",.2f"),
        ("rss_peak_mb", "Host peak RSS (MB)", ",.1f"),
        ("raw_cost", "Raw solution cost", ",.0f"),
        ("raw_structural_violations", "Raw violations", ",.0f"),
    ]
    trend_rows = []
    for col, label, fmt in trend_metrics:
        if not has(df, col):
            continue
        ys = [float(v) for v in df[col] if float(v) == float(v)]
        if len(ys) < 2:
            continue
        k = growth_exponent(hubs[: len(ys)], ys)
        cls, short, meaning = classify(k)
        f0 = human_bytes(ys[0]) if fmt is None else num(ys[0], fmt)
        f1 = human_bytes(ys[-1]) if fmt is None else num(ys[-1], fmt)
        mult = (ys[-1] / ys[0]) if ys[0] else float("nan")
        trend_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{f0}</td><td>{f1}</td>"
            f"<td>{num(mult, ',.1f')}×</td><td>{num(k, '+.2f')}</td>"
            f'<td style="text-align:left"><span class="chip {cls}">{html.escape(short)}</span></td>'
            f'<td style="text-align:left;font-family:\'IBM Plex Sans\',sans-serif">{html.escape(meaning)}</td></tr>'
        )

    # ---- full metrics table ---------------------------------------------
    table_cols = [
        ("instance", "Instance", None), ("hubs", "Hubs", ",.0f"),
        ("demand_rows", "Demand rows", ",.0f"), ("batches", "Batches", ",.0f"),
        ("num_z", "Z", ",.0f"), ("num_y", "Y", ",.0f"), ("num_x", "X", ",.0f"),
        ("binary_vars", "Binary vars", ",.0f"), ("interactions", "Interactions", ",.0f"),
        ("matrix_density", "Density", ".4%"), ("couplings_per_var", "Cpl/var", ".2f"),
        ("dense_bytes", "Dense (device)", "bytes"), ("dense_waste_factor", "Waste", ",.0f"),
        ("construction_seconds", "Build s", ",.2f"), ("annealing_seconds", "Anneal s", ",.2f"),
        ("total_wall_seconds", "Wall s", ",.2f"), ("rss_peak_mb", "Host RSS MB", ",.0f"),
        ("raw_cost", "Raw cost", ",.0f"), ("final_cost", "Final cost", ",.0f"),
        ("raw_structural_violations", "Raw viol", ",.0f"),
        ("final_structural_violations", "Final viol", ",.0f"),
    ]
    live = [(c, lbl, f) for c, lbl, f in table_cols if c == "instance" or has(df, c)]
    thead = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl, _ in live)
    body_rows = []
    for _, r in df.iterrows():
        cells = []
        for col, _, fmt in live:
            v = r.get(col)
            if col == "instance":
                cells.append(f"<td>{html.escape(str(v))}</td>")
            elif fmt == "bytes":
                cells.append(f"<td>{human_bytes(v)}</td>")
            else:
                cells.append(f"<td>{num(v, fmt or ',.0f')}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    foot_cells = []
    for col, _, _ in live:
        if col == "instance":
            foot_cells.append("<td>Growth exponent k</td>")
        elif has(df, col) and col != "hubs":
            ys = [float(v) for v in df[col] if float(v) == float(v)]
            k = growth_exponent(hubs[: len(ys)], ys) if len(ys) > 1 else float("nan")
            foot_cells.append(f"<td>{num(k, '+.2f')}</td>")
        else:
            foot_cells.append("<td>—</td>")

    # ---- projection ------------------------------------------------------
    var_k = growth_exponent(hubs, [float(v) for v in df["binary_vars"]])
    base_h, base_v = hubs[0], float(df["binary_vars"].iloc[0])
    max_h = int(max(hubs))
    targets = sorted({max_h, int(max_h * 1.5), max_h * 2, int(max_h * 2.5), max_h * 3})
    CEIL, HARD = 60000, 100000
    proj_rows = []
    for t in targets:
        v = base_v * (t / base_h) ** var_k
        pct = min(100.0, v / HARD * 100.0)
        measured = ' <span class="measured">measured</span>' if t == max_h else ""
        if v <= CEIL:
            verdict, vcls, color = "Fits", "fits", "var(--s-size)"
        elif v <= HARD:
            verdict, vcls, color = "Over 60k ceiling", "over", "var(--s-mem)"
        else:
            verdict, vcls, color = "Over hard max", "over", "var(--critical)"
        proj_rows.append(
            f'<div class="prow"><div class="hubs">{t:,}{measured}</div>'
            f'<div class="vars">{v:,.0f}</div>'
            f'<div class="bar"><i style="width:{pct:.1f}%;background:{color}"></i>'
            f'<span class="ceil" style="left:60%;background:var(--line-strong)"></span>'
            f'<span class="ceil" style="left:99.6%;background:var(--critical)"></span></div>'
            f'<div class="verdict {vcls}">{verdict}</div></div>'
        )
    single_batch_limit = base_h * (CEIL / base_v) ** (1.0 / var_k) if var_k else float("nan")
    hard_limit = base_h * (HARD / base_v) ** (1.0 / var_k) if var_k else float("nan")

    # ---- quality / pending ----------------------------------------------
    if full:
        quality_note = (
            '<div class="callout"><h3>Reading the violation columns</h3>'
            '<p>Judge Vector Annealing on the <strong>raw</strong> counts. Those are the samples the card '
            'returned. The <strong>final</strong> counts come after the repair and hub-prune post-pass, '
            'which drives structural violations to zero almost regardless of what was sampled — so the '
            'final figure measures the post-pass, not the annealer.</p></div>'
        )
    else:
        quality_note = (
            '<div class="callout pending"><h3>Annealing time and solution quality are not in this run</h3>'
            '<p>This CSV came from <code>--mode preflight</code>, which formulates and compiles the QUBO '
            'but never samples it. Those columns are absent by construction, not missing by oversight.</p>'
            '<p>Run <code>va_benchmark.py --mode full</code> on the VE node (<code>sfpga01n</code>) and '
            'regenerate this page from the resulting CSV — the annealing, cost and violation sections '
            'appear automatically once the columns exist.</p></div>'
        )

    # ---- figures ---------------------------------------------------------
    series_js = [
        '{key:"binary_vars",color:"var(--s-size)",label:"Binary variables",'
        'fmtv:d=>fmt(d.binary_vars)+" vars"}',
        '{key:"dense_bytes",color:"var(--s-mem)",label:"Device memory",fmtv:d=>bytes(d.dense_bytes)}',
    ]
    legend = (
        '<span><i class="swatch" style="background:var(--s-size)"></i> Binary variables — what the model needs</span>'
        '<span><i class="swatch" style="background:var(--s-mem)"></i> Device memory — what the card must hold</span>'
    )
    if full and has(df, "annealing_seconds"):
        series_js.append('{key:"annealing_seconds",color:"var(--s-density)",label:"Annealing time",'
                         'fmtv:d=>d.annealing_seconds.toFixed(2)+" s"}')
        legend += ('<span><i class="swatch" style="background:var(--s-density)"></i> '
                   'Annealing time — what the card spends</span>')

    extra_fig = ""
    if full and has(df, "raw_structural_violations"):
        extra_fig = """
    <figure>
      <p class="figtitle">Raw constraint violations returned by the card</p>
      <p class="figsub">Before the repair post-pass. Lower is better.</p>
      <div class="chartbox" id="box-viol">
        <svg id="chart-viol" viewBox="0 0 480 300" role="img" aria-label="Raw structural violations per instance."></svg>
        <div class="tooltip" id="tt-viol"></div>
      </div>
      <figcaption>This is the annealer's own output quality. If it climbs steeply with size, the sampling
      budget is not keeping up with the problem.</figcaption>
    </figure>"""

    meta = (
        f'<div><dt>Instances</dt><dd>{len(df)} · {int(min(hubs))}→{int(max(hubs))} hubs</dd></div>'
        f'<div><dt>Mode</dt><dd>{html.escape(mode)}</dd></div>'
        f'<div><dt>Source</dt><dd>{html.escape(Path(source).name)}</dd></div>'
    )
    if has(df, "parts"):
        meta += f'<div><dt>Part catalog</dt><dd>{num(df["parts"].iloc[0])}</dd></div>'
    if full and has(df, "total_reads"):
        meta += f'<div><dt>Total reads</dt><dd>{num(df["total_reads"].sum())}</dd></div>'

    headline = (
        "Vector Annealing on the FSL model"
        if full else
        "Formulation profile, ahead of the card"
    )
    standfirst = (
        f"{len(df)} instances from {int(min(hubs))} to {int(max(hubs))} hubs, measured end to end on the "
        "vector engine. Every figure below is computed from the benchmark CSV, so this page says what "
        "the run actually did."
        if full else
        f"{len(df)} instances from {int(min(hubs))} to {int(max(hubs))} hubs, profiled through the solver's "
        "formulation path. Annealing and quality still need the card — the page marks where."
    )

    return f"""<title>VA Scaling Profile</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>

<header><div class="wrap">
  <div class="eyebrow">NEC Vector Annealing · FSL digital twin</div>
  <h1>{html.escape(headline)}</h1>
  <p class="standfirst">{html.escape(standfirst)}</p>
  <dl class="runmeta">{meta}</dl>
</div></header>

<div class="wrap">

<section>
  <div class="sechead">
    <div class="eyebrow">At a glance</div>
    <h2>How each metric grows with the network</h2>
    <p>Each figure is the growth exponent <em>k</em>, fitted across the runs: the metric scales as
    hubs<sup>k</sup>. k=1 means it doubles when hubs double; k=2 means it quadruples.</p>
  </div>
  <div class="tiles">{''.join(tiles)}</div>
</section>

<section>
  <div class="sechead">
    <div class="eyebrow">Figure 1</div>
    <h2>Growth curves</h2>
    <p>Each series is indexed to its own value at the smallest run, so the axis reads as growth multiple
    rather than raw units. Log scale — a straight line means clean power-law growth, and steeper is worse.</p>
  </div>
  <figure>
    <p class="figtitle">Growth relative to the {int(min(hubs))}-hub run</p>
    <p class="figsub">Hover any point for its underlying value.</p>
    <div class="legend">{legend}</div>
    <div class="chartbox" id="box-growth">
      <svg id="chart-growth" viewBox="0 0 780 360" role="img" aria-label="Log-scale growth chart across the benchmark runs."></svg>
      <div class="tooltip" id="tt-growth"></div>
    </div>
    <figcaption>The gap between the lines is the cost of dense storage: the card's allocation grows as the
    square of a variable count that itself only grows linearly.</figcaption>
  </figure>
</section>

<section>
  <div class="sechead">
    <div class="eyebrow">Figure 2</div>
    <h2>Why the memory curve is steep</h2>
    <p>Vector Annealing allocates a full <code>n × n</code> matrix at 4 bytes per cell whatever the coupling
    structure looks like. These couplings are local, so the matrix is overwhelmingly zeros.</p>
  </div>
  <div class="twocol">
    <figure>
      <p class="figtitle">Share of the matrix that is non-zero</p>
      <p class="figsub">Density falls as the problem grows.</p>
      <div class="chartbox" id="box-density">
        <svg id="chart-density" viewBox="0 0 480 300" role="img" aria-label="Matrix density per instance."></svg>
        <div class="tooltip" id="tt-density"></div>
      </div>
      <figcaption>Local couplings: Z–Z only within a demand row, Z–Y and Y–X only along the hub and part
      they belong to.</figcaption>
    </figure>
    <figure>
      <p class="figtitle">Allocated vs. needed</p>
      <p class="figsub">The same couplings, stored densely and sparsely.</p>
      <div class="chartbox" id="box-waste">
        <svg id="chart-waste" viewBox="0 0 480 300" role="img" aria-label="Dense storage waste factor per instance."></svg>
        <div class="tooltip" id="tt-waste"></div>
      </div>
      <figcaption>The waste factor is how many times more memory the dense layout needs than the couplings
      themselves do.</figcaption>
    </figure>
  </div>
</section>

{f'<section><div class="sechead"><div class="eyebrow">Figure 3</div><h2>What the card returned</h2>'
 f'<p>Sampler output quality across the suite, before any repair.</p></div>'
 f'<div class="twocol">{extra_fig}</div></section>' if extra_fig else ''}

<section>
  <div class="sechead">
    <div class="eyebrow">Trend table</div>
    <h2>Every metric, with its growth law</h2>
    <p>Fitted across all {len(df)} runs. The class column is the plain reading of the exponent.</p>
  </div>
  <div class="tablebox"><table>
    <thead><tr><th>Metric</th><th>Smallest run</th><th>Largest run</th><th>Growth</th><th>k</th>
    <th style="text-align:left">Class</th><th style="text-align:left">Meaning</th></tr></thead>
    <tbody>{''.join(trend_rows)}</tbody>
  </table></div>
</section>

<section>
  <div class="sechead">
    <div class="eyebrow">Headroom</div>
    <h2>How far this scales before it stops</h2>
    <p>Extrapolating the fitted law (variables ≈ hubs<sup>{num(var_k, '.2f')}</sup>) against the configured
    per-batch ceiling of 60,000 variables and Vector Annealing's hard maximum of 100,000. Bars are scaled
    to the hard maximum; the grey mark is the 60k ceiling, the red mark the 100k limit.</p>
  </div>
  <div class="proj">
    <div class="prow head"><div>Hubs</div><div>Binary vars</div>
    <div>Against the 60k ceiling and 100k hard max</div><div style="text-align:right">Single batch</div></div>
    {''.join(proj_rows)}
  </div>
  <div class="callout good" style="margin-top:22px">
    <h3>Single-batch headroom ends near {num(single_batch_limit, ',.0f')} hubs</h3>
    <p>At the configured ceiling that is roughly <strong>{num(single_batch_limit, ',.0f')} hubs</strong>;
    against the 100,000-variable hard maximum, roughly <strong>{num(hard_limit, ',.0f')}</strong>. Past that,
    batching stops being optional — lowering <code>--max-z-vars-per-batch</code> splits the problem and cuts
    the largest dense allocation proportionally.</p>
  </div>
</section>

<section>
  <div class="sechead">
    <div class="eyebrow">Recorded metrics</div>
    <h2>The full table</h2>
    <p>Device-side and host-side memory are separate resources and should not be added together. Dense is
    what the VE card must hold; host RSS is Python building the QUBO before transfer.</p>
  </div>
  <div class="tablebox"><table>
    <thead><tr>{thead}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
    <tfoot><tr>{''.join(foot_cells)}</tr></tfoot>
  </table></div>
</section>

<section>
  <div class="sechead"><div class="eyebrow">Notes</div><h2>Reading this run</h2></div>
  <div style="display:grid;gap:16px">{quality_note}
  <div class="callout"><h3>Worth raising with NEC</h3>
  <p>A density of {num(df['matrix_density'].iloc[-1], '.4%')} at
  {num(df['couplings_per_var'].iloc[-1], '.1f') if has(df, 'couplings_per_var') else '~2'} couplings per
  variable is a poor fit for dense storage. If Vector Annealing V3.0.0 exposes a sparse or block ingest
  path, it is worth more on this workload than any amount of sampling-parameter tuning — it would turn a
  quadratic memory wall back into a linear one.</p></div></div>
</section>

<footer>Generated by <code>va_report.py</code> from <code>{html.escape(source)}</code>.
Growth exponents are log–log least-squares fits across the runs in that file. Device memory is
<code>binary_vars² × 4 B</code>, the allocation Vector Annealing makes regardless of density.</footer>
</div>

<script>
const DATA = {json.dumps(data_js)};
{CHART_JS}
growthChart("chart-growth","box-growth","tt-growth",[{','.join(series_js)}]);
barChart("chart-density","box-density","tt-density","matrix_density","var(--s-density)",
  v=>(v*100).toFixed(3)+"%",
  d=>'<div class="tt-r"><span>Density</span><span>'+(d.matrix_density*100).toFixed(4)+'%</span></div>'+
     '<div class="tt-r"><span>Non-zero</span><span>'+fmt(d.interactions)+'</span></div>'+
     '<div class="tt-r"><span>Cells</span><span>'+fmt(Math.round(d.binary_vars*d.binary_vars))+'</span></div>',480);
barChart("chart-waste","box-waste","tt-waste","dense_waste_factor","var(--s-mem)",
  v=>fmt(Math.round(v))+"×",
  d=>'<div class="tt-r"><span>Allocated</span><span>'+bytes(d.dense_bytes)+'</span></div>'+
     (d.sparse_bytes?'<div class="tt-r"><span>Needed</span><span>'+bytes(d.sparse_bytes)+'</span></div>':'')+
     '<div class="tt-r"><span>Waste</span><span>'+fmt(Math.round(d.dense_waste_factor))+'×</span></div>',480);
barChart("chart-viol","box-viol","tt-viol","raw_structural_violations","var(--critical)",
  v=>fmt(Math.round(v)),
  d=>'<div class="tt-r"><span>Raw violations</span><span>'+fmt(Math.round(d.raw_structural_violations))+'</span></div>'+
     (d.raw_cost?'<div class="tt-r"><span>Raw cost</span><span>'+fmt(Math.round(d.raw_cost))+'</span></div>':''),480);
</script>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="CSV written by va_benchmark.py --csv")
    p.add_argument("-o", "--out", default="", help="Output .html (default: alongside the CSV)")
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"ERROR: CSV not found: {csv_path}")
    out = Path(args.out) if args.out else csv_path.with_suffix(".html")

    df = pd.read_csv(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(df, str(csv_path)), encoding="utf-8")
    print(f"report written to {out}")
    print(f"  open it with:  open {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
