#!/bin/bash
# One VTune run per mode and thread count.
# Usage: bash run_vtune_sweep.sh standard|wide5

BENCH="${1:-}"
case "$BENCH" in
    standard) OUT_DIR="vtune_8layers" ;;
    wide5)    OUT_DIR="vtune_5layers" ;;
    *)
        echo "Usage: bash run_vtune_sweep.sh standard|wide5"
        echo "  standard  8-layer network"
        echo "  wide5     5 wider layers, 1 inference"
        exit 1
        ;;
esac

BENCHMARKS=("$BENCH")
MODES=(outer reduction simd outer_simd)
THREADS_LIST=(1 2 4 8 16 20 21 32)
CSV_FILE="$OUT_DIR/resultados.csv"
REPORT_DIR="$OUT_DIR/reports"
MD_FILE="$OUT_DIR/relatorio.md"

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

pin_clock() {
    local cpu0="/sys/devices/system/cpu/cpu0/cpufreq"
    local target=2590000
    local freq="$target"
    local avail chosen

    if [ ! -d "$cpu0" ]; then
        echo "cpufreq is not available on this node; leaving the clock unchanged."
        return 0
    fi

    if [ -f "$cpu0/scaling_available_frequencies" ]; then
        avail=$(cat "$cpu0/scaling_available_frequencies")
        chosen=$(printf '%s\n' $avail | awk -v t="$target" '
            BEGIN { best = -1 }
            {
                d = ($1 > t) ? ($1 - t) : (t - $1)
                if (best < 0 || d < bestd) { best = $1; bestd = d }
            }
            END { print best }
        ')
        if [ -n "$chosen" ] && [ "$chosen" != "-1" ]; then
            freq="$chosen"
        fi
    fi

    echo "Pinning every core to ${freq} kHz ($(awk -v f="$freq" 'BEGIN{printf "%.3f GHz", f/1000000}'))"
    local i
    for i in $(seq 0 $(($(nproc) - 1))); do
        if command -v cpufreq-set >/dev/null 2>&1; then
            cpufreq-set -c "$i" -g performance -d "$freq" -u "$freq" || true
        fi
        echo "$freq" > "/sys/devices/system/cpu/cpu${i}/cpufreq/scaling_max_freq" 2>/dev/null || true
        echo "$freq" > "/sys/devices/system/cpu/cpu${i}/cpufreq/scaling_min_freq" 2>/dev/null || true
    done
}

mkdir -p "$REPORT_DIR"

if ! command -v vtune &> /dev/null; then
    echo "vtune is not on PATH. Load the profiler environment, then run this script again."
    exit 1
fi

pin_clock

echo "Compilando nn_sweep..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target nn_sweep

echo "benchmark,topology,mode,threads,time_s,speedup,efficiency_pct,cpi,cores_util,memory_bound_pct,numa_remote_pct,dram_bw_gbs,checksum" > "$CSV_FILE"

declare -A BASE_TIME

for BENCH in "${BENCHMARKS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for T in "${THREADS_LIST[@]}"; do
            RESULT_DIR="$OUT_DIR/r_${BENCH}_${MODE}_${T}t"
            LOG_FILE="$OUT_DIR/run_${BENCH}_${MODE}_${T}t.log"
            REPORT_FILE="$REPORT_DIR/${BENCH}_${MODE}_t${T}.txt"

            echo "------------------------------------------------------------------------"
            echo ">>> VTune  benchmark=$BENCH  modo=$MODE  threads=$T"
            echo "------------------------------------------------------------------------"

            rm -rf "$RESULT_DIR" "$LOG_FILE" "$REPORT_FILE"

            set +e
            vtune -collect hpc-performance -result-dir "$RESULT_DIR" -- \
                ./build/nn_sweep "$MODE" "$T" "$BENCH" 2>&1 | tee "$LOG_FILE"
            RUN_STATUS=${PIPESTATUS[0]}
            set -e

            BATCH_N=$(grep "^BATCH:" "$LOG_FILE" | head -n 1 | cut -d':' -f2 | tr -d ' ' || true)
            if [ "$BATCH_N" != "1" ]; then
                echo "Timed BATCH=$BATCH_N, expected 1 inference."
                echo "The binary is stale. Rebuild and run the script again."
                exit 1
            fi

            TOPOLOGY=$(grep "^TOPOLOGY:" "$LOG_FILE" | head -n 1 | cut -d':' -f2- | tr -d ' ' || true)
            if [ -z "$TOPOLOGY" ]; then
                TOPOLOGY="N/A"
            fi

            RESULT_LINE=$(grep "RESULT_CSV:" "$LOG_FILE" | head -n 1 | cut -d':' -f2- || true)
            TIME_S=$(echo "$RESULT_LINE" | cut -d',' -f4 || true)
            CHECKSUM=$(grep "CHECKSUM:" "$LOG_FILE" | head -n 1 | cut -d':' -f2 || true)
            if [ -z "$TIME_S" ]; then
                TIME_S="nan"
            fi
            if [ -z "$CHECKSUM" ]; then
                CHECKSUM="nan"
            fi

            BASE_KEY="${BENCH}_${MODE}"
            if [ "$T" -eq 1 ] && [ "$TIME_S" != "nan" ]; then
                BASE_TIME[$BASE_KEY]="$TIME_S"
            fi

            STATS=$(python3 -c "
base = float('${BASE_TIME[$BASE_KEY]:-0}' or 0)
tp = float('$TIME_S') if '$TIME_S' != 'nan' else 0.0
threads = int('$T')
sp = (base / tp) if base > 0 and tp > 0 else float('nan')
eff = (sp / threads) * 100.0 if sp == sp and threads > 0 else float('nan')
print(f'{sp:.2f},{eff:.1f}')
")
            SPEEDUP=$(echo "$STATS" | cut -d',' -f1)
            EFF=$(echo "$STATS" | cut -d',' -f2)

            vtune -report summary -report-knob show-issues=false -result-dir "$RESULT_DIR" > "$REPORT_FILE" 2>/dev/null || true

            CPI=$(grep "CPI Rate:" "$REPORT_FILE" | head -n 1 | awk '{print $3}' || true)
            CORES=$(grep "Effective Physical Core Utilization:" "$REPORT_FILE" | head -n 1 | grep -o '([0-9.]*' | tr -d '(' || true)
            MEM_BOUND=$(grep "Memory Bound:" "$REPORT_FILE" | head -n 1 | awk '{print $3}' | tr -d '%' || true)
            NUMA=$(grep "NUMA: % of Remote Accesses:" "$REPORT_FILE" | head -n 1 | awk '{print $6}' | tr -d '%' || true)
            DRAM_BW=$(grep "DRAM, GB/sec" "$REPORT_FILE" | head -n 1 | awk '{print $4}' || true)

            CPI=${CPI:-N/A}
            CORES=${CORES:-N/A}
            MEM_BOUND=${MEM_BOUND:-N/A}
            NUMA=${NUMA:-N/A}
            DRAM_BW=${DRAM_BW:-N/A}

            echo "$BENCH,$TOPOLOGY,$MODE,$T,$TIME_S,$SPEEDUP,$EFF,$CPI,$CORES,$MEM_BOUND,$NUMA,$DRAM_BW,$CHECKSUM" >> "$CSV_FILE"
            echo "status=$RUN_STATUS benchmark=$BENCH tempo=${TIME_S}s speedup=${SPEEDUP} cpi=${CPI}"

            rm -rf "$RESULT_DIR"
        done
    done
done

python3 - "$CSV_FILE" "$MD_FILE" << 'PY'
import csv
import sys
from collections import defaultdict

csv_path, md_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
by_bench_mode = defaultdict(list)
for row in rows:
    by_bench_mode[(row["benchmark"], row["mode"])].append(row)

descriptions = {
    "outer": "Parallel neuron loop; scalar dot product.",
    "reduction": "Sequential neuron loop; reduction on the dot product.",
    "simd": "Sequential neuron loop; SIMD reduction on the dot product.",
    "outer_simd": "Parallel neuron loop; SIMD dot product, no reduction.",
}
bench_desc = {
    "standard": "8 layers: 1000 → 10000 (×6) → 1000, 1 inference.",
    "wide5": "5 layers: 8000 → 52000 (×3) → 8000, 1 inference.",
}

lines = []
lines.append("# Varredura VTune — dois benchmarks")
lines.append("")
lines.append("Each row is one `vtune -collect hpc-performance` process.")
lines.append("Speedup and efficiency use the **1-thread** time of the same benchmark and mode.")
lines.append("")
lines.append("Colunas: benchmark, topology, mode, threads, time_s, speedup, efficiency_pct, cpi, cores_util, memory_bound_pct, numa_remote_pct, dram_bw_gbs, checksum.")
lines.append("")

header = "| Threads | Time (s) | Speedup | Efficiency | CPI | Cores | Memory bound | Remote NUMA | DRAM (GB/s) |"
sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

seen = []
for row in rows:
    if row["benchmark"] not in seen:
        seen.append(row["benchmark"])
for bench in seen:
    lines.append(f"## Benchmark `{bench}`")
    lines.append("")
    lines.append(bench_desc.get(bench, ""))
    lines.append("")
    for mode in ("outer", "reduction", "simd", "outer_simd"):
        lines.append(f"### {mode}")
        lines.append("")
        lines.append(descriptions.get(mode, ""))
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for row in sorted(by_bench_mode.get((bench, mode), []), key=lambda r: int(r["threads"])):
            def cell(key, suffix=""):
                value = row.get(key, "N/A")
                if value in ("", "nan", "N/A"):
                    return "N/A"
                return f"{value}{suffix}"
            lines.append(
                "| {t} | {time} | {sp} | {eff} | {cpi} | {cores} | {mem} | {numa} | {dram} |".format(
                    t=row["threads"],
                    time=cell("time_s"),
                    sp=cell("speedup", "×"),
                    eff=cell("efficiency_pct", "%"),
                    cpi=cell("cpi"),
                    cores=cell("cores_util"),
                    mem=cell("memory_bound_pct", "%"),
                    numa=cell("numa_remote_pct", "%"),
                    dram=cell("dram_bw_gbs"),
                )
            )
        lines.append("")

text = "\n".join(lines) + "\n"
open(md_path, "w").write(text)
print(text)
PY

echo ""
echo "Done. Files are in $OUT_DIR/"
