"""Summary table and charts for the SPARKNLP-1455 notebook. Downloaded to /content by the setup cell.

Figures are plain matplotlib Figure objects (no pyplot state), so the notebook shows each one once with display().
Colors come from the reference data-viz palette, light mode; the two runtime colors pass the CVD validator
(worst ΔE 24.7). Spark NLP is always blue and HF always orange, in every chart.
"""
from __future__ import annotations

import os

import bench_utils

SURFACE, INK, INK_2, GRID, AXIS, CORE_INK = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c7c1", "#a8a6a0"
RUNTIMES = ("spark", "hf")
RUNTIME_NAME = {"spark": "Spark NLP", "hf": "HF Transformers"}
RUNTIME_COLOR = {"spark": "#2a78d6", "hf": "#eb6834"}
MODEL_NAME = {"m2m100": "M2M100 418M", "marian": "Marian opus-mt-pl-de", "nllb": "NLLB 600M"}
METRICS = ["bleu", "chrf", "load_s", "disk_mb", "latency_median_s", "latency_p95_s", "throughput_sps", "peak_rss_mb"]
COLUMNS = ["model", "runtime", "bleu", "chrf", "spark_hf_match_pct", "load_s", "disk_mb",
           "latency_median_s", "latency_p95_s", "throughput_sps", "peak_rss_mb"]


# ---- tables -----------------------------------------------------------------------------------------------

def comparison_table(results):
    import pandas as pd

    rows = []
    for key, r in results.items():
        m = r["metrics"]
        for rt in RUNTIMES:
            rows.append({"model": key, "runtime": rt, "spark_hf_match_pct": m.get("spark_hf_exact_match_pct"),
                         **{c: m[rt].get(c) for c in METRICS}})
    return pd.DataFrame(rows, columns=COLUMNS)


def qualitative_table(results):
    table = None
    for key, r in results.items():
        q = r["qualitative"].rename(columns={"spark": f"{key}_spark", "hf": f"{key}_hf"})
        table = q if table is None else table.merge(q.drop(columns="pl"), on="id", how="left")
    return table


# ---- shared chart pieces ----------------------------------------------------------------------------------

def _figure(n_rows, n_cols=1, row_height=2.6, width=10):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, row_height * max(n_rows, 1) + 0.6), facecolor=SURFACE, layout="constrained")
    axes = list(fig.subplots(max(n_rows, 1), n_cols, squeeze=False).flat)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    return fig, axes


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=10, color=INK, pad=16)  # room for the phase names above the plot


def _label(ax, text, axis):
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)(text, fontsize=8.5, color=INK_2)


def _legend(fig, handles):
    fig.legend(handles=handles, loc="outside upper right", ncol=len(handles), frameon=False, fontsize=8.5,
               labelcolor=INK_2)


def _runtime_handles(kind="patch"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if kind == "dot":
        return [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=RUNTIME_COLOR[rt],
                       markeredgecolor=SURFACE, label=RUNTIME_NAME[rt]) for rt in RUNTIMES]
    return [Patch(color=RUNTIME_COLOR[rt], label=RUNTIME_NAME[rt]) for rt in RUNTIMES]


def _phase_spans(cpu):
    """[start_t, end_t, phase] for each run of consecutive samples in the same phase."""
    spans = []
    for phase, t in zip(cpu["phase"], cpu["t"]):
        if spans and spans[-1][2] == phase:
            spans[-1][1] = t
        else:
            spans.append([t, t, phase])
    return spans


def _shade_phases(ax, cpu):
    """Tint each phase in its runtime's color; name the phases wide enough to hold a label."""
    width = max(float(cpu["t"].max() - cpu["t"].min()), 1e-9) if len(cpu) else 1.0
    for start, end, phase in _phase_spans(cpu):
        runtime, _, step = phase.partition("_")
        if runtime not in RUNTIME_COLOR:
            continue
        ax.axvspan(start, end, color=RUNTIME_COLOR[runtime], alpha=0.10, linewidth=0)
        if (end - start) / width > 0.035:
            ax.text((start + end) / 2, 1.01, step, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7.5, color=INK_2)


def _phase_handles():
    from matplotlib.patches import Patch

    return [Patch(color=RUNTIME_COLOR[rt], alpha=0.25, label=f"{RUNTIME_NAME[rt]} phases") for rt in RUNTIMES]


def _grouped_bars(ax, results, metric, fmt=None):
    keys = list(results)
    width = 0.36
    for j, rt in enumerate(RUNTIMES):
        values = [results[k]["metrics"][rt][metric] for k in keys]
        xs = [i + (j - 0.5) * width for i in range(len(keys))]
        bars = ax.bar(xs, values, width, color=RUNTIME_COLOR[rt], edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values) if fmt else ():
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([MODEL_NAME.get(k, k) for k in keys])
    return keys


# ---- charts -----------------------------------------------------------------------------------------------

def plot_cpu(results):
    """Is every core busy? Thin gray line per core, bold line for their mean, phases tinted by runtime."""
    from matplotlib.lines import Line2D

    fig, axes = _figure(len(results))
    for ax, (key, r) in zip(axes, results.items()):
        cpu = r["cpu"]
        cores = [c for c in cpu.columns if c.startswith("cpu_core")]
        _shade_phases(ax, cpu)
        for c in cores:
            ax.plot(cpu["t"], cpu[c], color=CORE_INK, linewidth=1)
        ax.plot(cpu["t"], cpu[cores].mean(axis=1), color=INK, linewidth=2)
        ax.set_ylim(0, 105)
        _title(ax, MODEL_NAME.get(key, key))
        _label(ax, "CPU %", "y")
    _label(axes[-1], "seconds since the section started", "x")
    _legend(fig, [Line2D([], [], color=CORE_INK, linewidth=1, label="each core"),
                  Line2D([], [], color=INK, linewidth=2, label="mean of cores"), *_phase_handles()])
    return fig


def plot_memory(results):
    from matplotlib.lines import Line2D

    fig, axes = _figure(len(results))
    for ax, (key, r) in zip(axes, results.items()):
        cpu = r["cpu"]
        _shade_phases(ax, cpu)
        ax.plot(cpu["t"], cpu["java_rss_mb"], color=RUNTIME_COLOR["spark"], linewidth=2)
        ax.plot(cpu["t"], cpu["py_rss_mb"], color=RUNTIME_COLOR["hf"], linewidth=2)
        ax.set_ylim(bottom=0)
        _title(ax, MODEL_NAME.get(key, key))
        _label(ax, "memory in use (MB)", "y")
    _label(axes[-1], "seconds since the section started", "x")
    _legend(fig, [Line2D([], [], color=RUNTIME_COLOR["spark"], linewidth=2, label="Java process (Spark NLP)"),
                  Line2D([], [], color=RUNTIME_COLOR["hf"], linewidth=2, label="Python process (HF)"),
                  *_phase_handles()])
    return fig


def plot_speed(results):
    fig, (throughput, latency) = _figure(1, 2, row_height=3.4)
    _grouped_bars(throughput, results, "throughput_sps", "{:.2f}")
    _title(throughput, "Batch throughput (higher is better)")
    _label(throughput, "sentences per second", "y")
    keys = _grouped_bars(latency, results, "latency_median_s")  # values in the table; labels would sit on the whiskers
    for j, rt in enumerate(RUNTIMES):  # whisker from the median up to p95
        for i, k in enumerate(keys):
            m = results[k]["metrics"][rt]
            x = i + (j - 0.5) * 0.36
            latency.vlines(x, m["latency_median_s"], m["latency_p95_s"], color=INK, linewidth=1.5)
            latency.hlines(m["latency_p95_s"], x - 0.06, x + 0.06, color=INK, linewidth=1.5)
    _title(latency, "Single-sentence latency (lower is better)")
    _label(latency, "seconds (bar = median, whisker = p95)", "y")
    _legend(fig, _runtime_handles())
    return fig


def plot_quality(results):
    fig, (bleu, chrf) = _figure(1, 2, row_height=3.4)
    _grouped_bars(bleu, results, "bleu", "{:.1f}")
    _title(bleu, "BLEU vs FLORES reference")
    _grouped_bars(chrf, results, "chrf", "{:.1f}")
    _title(chrf, "chrF vs FLORES reference")
    _legend(fig, _runtime_handles())
    return fig


def plot_latency_vs_length(results):
    """Without a KV cache, latency grows faster than output length; with one it stays close to linear."""
    fig, axes = _figure(1, max(len(results), 1), row_height=3.2)
    for ax, (key, r) in zip(axes, results.items()):
        single = r["single"]
        for rt in RUNTIMES:
            part = single[single["runtime"] == rt]
            ax.scatter(part["out_tokens"], part["latency_s"], s=40, color=RUNTIME_COLOR[rt], edgecolors=SURFACE,
                       linewidths=1)
        ax.set_ylim(bottom=0)
        _title(ax, MODEL_NAME.get(key, key))
        _label(ax, "output tokens", "x")
    _label(axes[0], "seconds per sentence", "y")
    _legend(fig, _runtime_handles("dot"))
    return fig


def save_figures(figs):
    folder = os.path.join(bench_utils.RESULTS_DIR, "charts")  # read at call time: ROOT can change on reload
    os.makedirs(folder, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        path = os.path.join(folder, f"{name}.png")
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        paths.append(path)
    return paths
