#!/usr/bin/env python3
"""Write speedup charts in resultados/ from the published CSVs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "resultados" / "dados"
OUT = ROOT / "resultados"

MODES = ("outer", "reduction", "simd", "outer_simd")
LABELS = {
    "outer": "outer",
    "reduction": "reduction",
    "simd": "simd",
    "outer_simd": "outer + SIMD",
}
COLORS = {
    "outer": "#0072B2",
    "reduction": "#E69F00",
    "simd": "#009E73",
    "outer_simd": "#CC79A7",
}


def load(path):
    rows = list(csv.DictReader(path.open()))
    by_mode = {mode: [] for mode in MODES}
    for row in rows:
        by_mode[row["mode"]].append(row)
    for mode in MODES:
        by_mode[mode].sort(key=lambda r: int(r["threads"]))
    return by_mode


def plot(by_mode, title, outfile):
    threads = [int(r["threads"]) for r in by_mode["outer"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=140)
    ax.plot(
        threads,
        threads,
        linestyle="--",
        color="#444444",
        linewidth=1.4,
        label="ideal",
        zorder=1,
    )
    for mode in MODES:
        ys = [float(r["speedup"]) for r in by_mode[mode]]
        ax.plot(
            threads,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.1,
            color=COLORS[mode],
            label=LABELS[mode],
            zorder=2,
        )
    ax.axvline(20, color="#888888", linestyle=":", linewidth=1.1, zorder=0)
    ax.text(20.35, 30.2, "20 physical cores", color="#666666", fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Speedup versus 1 thread of the same mode")
    ax.set_xticks(threads)
    ax.set_ylim(0, 34)
    ax.set_xlim(0, 34)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    print(outfile)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot(
        load(DATA / "8camadas.csv"),
        "Speedup — 8 layers (1000 → 10000 ×6 → 1000)",
        OUT / "speedup_8camadas.png",
    )
    plot(
        load(DATA / "5camadas.csv"),
        "Speedup — 5 layers (8000 → 52000 ×3 → 8000)",
        OUT / "speedup_5camadas.png",
    )


if __name__ == "__main__":
    main()
