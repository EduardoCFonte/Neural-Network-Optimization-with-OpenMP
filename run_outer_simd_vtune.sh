#!/bin/bash
# VTune runs for outer_simd only.
# Usage: bash run_outer_simd_vtune.sh standard|wide5|both

BENCH_ARG="${1:-}"
case "$BENCH_ARG" in
    standard) BENCHMARKS=(standard) ;;
    wide5)    BENCHMARKS=(wide5) ;;
    both)     BENCHMARKS=(standard wide5) ;;
    *)
        echo "Usage: bash run_outer_simd_vtune.sh standard|wide5|both"
        echo "  standard  8 layers, 1 inference"
        echo "  wide5     5 wider layers, 1 inference"
        echo "  both      both benchmarks"
        exit 1
        ;;
esac

MODE=outer_simd
THREADS_LIST=(1 2 4 8 16 20 21 32)

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

if ! command -v vtune &> /dev/null; then
    echo "vtune is not on PATH. Load the profiler environment, then run this script again."
    exit 1
fi

pin_clock

echo "Compilando nn_sweep..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target nn_sweep

for BENCH in "${BENCHMARKS[@]}"; do
    case "$BENCH" in
        standard) OUT_DIR="vtune_8layers_outer_simd" ;;
        wide5)    OUT_DIR="vtune_5layers_outer_simd" ;;
    esac
    REPORT_DIR="$OUT_DIR/reports"
    CSV_FILE="$OUT_DIR/resultados_outer_simd.csv"
    MD_FILE="$OUT_DIR/relatorio_outer_simd.md"

    mkdir -p "$REPORT_DIR"
    echo "benchmark,topology,mode,threads,time_s,speedup,efficiency_pct,cpi,cores_util,memory_bound_pct,numa_remote_pct,checksum" > "$CSV_FILE"

    declare -A BASE_TIME
    BASE_TIME=()

    for T in "${THREADS_LIST[@]}"; do
        RESULT_DIR="$OUT_DIR/r_${BENCH}_${MODE}_${T}t"
        LOG_FILE="$OUT_DIR/run_${MODE}_${T}t.log"
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

        if [ "$T" -eq 1 ] && [ "$TIME_S" != "nan" ]; then
            BASE_TIME[$BENCH]="$TIME_S"
        fi

        STATS=$(python3 -c "
base = float('${BASE_TIME[$BENCH]:-0}' or 0)
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

        CPI=${CPI:-N/A}
        CORES=${CORES:-N/A}
        MEM_BOUND=${MEM_BOUND:-N/A}
        NUMA=${NUMA:-N/A}

        echo "$BENCH,$TOPOLOGY,$MODE,$T,$TIME_S,$SPEEDUP,$EFF,$CPI,$CORES,$MEM_BOUND,$NUMA,$CHECKSUM" >> "$CSV_FILE"
        echo "status=$RUN_STATUS benchmark=$BENCH tempo=${TIME_S}s speedup=${SPEEDUP} cpi=${CPI}"

        rm -rf "$RESULT_DIR"
    done

    python3 - "$CSV_FILE" "$MD_FILE" "$BENCH" << 'PY'
import csv
import sys

csv_path, md_path, bench = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(csv_path)))
rows.sort(key=lambda r: int(r["threads"]))

bench_desc = {
    "standard": "8 layers: 1000 → 10000 (×6) → 1000, **1 inference**.",
    "wide5": "5 layers: 8000 → 52000 (×3) → 8000, **1 inference**.",
}

lines = [
    f"# VTune — apenas `outer_simd` — benchmark `{bench}`",
    "",
    "Parallel neuron loop (`#pragma omp parallel for`); dot product with `#pragma omp simd`, no `reduction`.",
    "",
    bench_desc.get(bench, ""),
    "",
    "Speedup and efficiency are relative to **1 thread** on this benchmark.",
    "",
    "| Threads | Time (s) | Speedup | Efficiency | CPI | Cores | Memory bound | Remote NUMA |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

def cell(row, key, suffix=""):
    value = row.get(key, "N/A")
    if value in ("", "nan", "N/A"):
        return "N/A"
    return f"{value}{suffix}"

for row in rows:
    lines.append(
        "| {t} | {time} | {sp} | {eff} | {cpi} | {cores} | {mem} | {numa} |".format(
            t=row["threads"],
            time=cell(row, "time_s"),
            sp=cell(row, "speedup", "×"),
            eff=cell(row, "efficiency_pct", "%"),
            cpi=cell(row, "cpi"),
            cores=cell(row, "cores_util"),
            mem=cell(row, "memory_bound_pct", "%"),
            numa=cell(row, "numa_remote_pct", "%"),
        )
    )

open(md_path, "w").write("\n".join(lines) + "\n")
print(f"Escrito: {csv_path} e {md_path}")
PY

    echo ""
    echo "Done ($BENCH). Output: $OUT_DIR/"
done

echo ""
echo "Done."
