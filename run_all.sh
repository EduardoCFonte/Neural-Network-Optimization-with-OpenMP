#!/bin/bash
# Profile nn_inference at 1, 2, 4, 8, 16, 20, 21, and 32 threads.

set -e

# Pin threads to nearby cores.
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

echo "Building nn_inference..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# VTune must already be on PATH.
if ! command -v vtune &> /dev/null; then
    echo "vtune is not on PATH. Load the profiler environment, then run this script again."
    exit 1
fi

THREADS_LIST=(1 2 4 8 16 20 21 32)
CSV_FILE="benchmark_results.csv"
SUMMARY_FILE="benchmark_summary.txt"

echo "Threads,Time_s,Speedup,Efficiency_pct,CPI,Cores_Util,Memory_Bound_pct,NUMA_Remote_pct,DRAM_BW_GBs" > "$CSV_FILE"

echo "========================================================================"
echo "          INICIANDO BATERIA DE BENCHMARKS AUTOMATIZADA"
echo "  Threads a testar: 1, 2, 4, 8, 16, 20, 21, 32"
echo "  One inference per thread count"
echo "========================================================================"
echo ""

T1_TIME=0

for T in "${THREADS_LIST[@]}"; do
    RESULT_DIR="r_buffed_${T}threads"
    LOG_FILE="run_${T}t.log"
    REPORT_FILE="report_${T}t.txt"

    echo "------------------------------------------------------------------------"
    echo ">>> [ETAPA] Executando benchmark com $T thread(s)..."
    echo "------------------------------------------------------------------------"

    rm -rf "$RESULT_DIR" "$LOG_FILE" "$REPORT_FILE"

    # Executa a coleta do VTune
    vtune -collect hpc-performance -result-dir "$RESULT_DIR" -- ./build/nn_inference "$T" 2>&1 | tee "$LOG_FILE"

    # Extrai o tempo medido no C++
    TIME_S=$(grep "RESULT_CSV:" "$LOG_FILE" | head -n 1 | cut -d',' -f2 || true)
    if [ -z "$TIME_S" ]; then
        TIME_S=$(grep "Parallel Time" "$LOG_FILE" | head -n 1 | awk '{print $(NF-3)}' || true)
    fi
    if [ -z "$TIME_S" ]; then
        TIME_S="0.00"
    fi

    # 1 thread is the sequential baseline.
    if [ "$T" -eq 1 ]; then
        T1_TIME="$TIME_S"
    fi

    # Speedup and efficiency from the 1-thread time.
    STATS=$(python3 -c "
t1 = float('$T1_TIME') if float('$T1_TIME') > 0 else float('$TIME_S')
tp = float('$TIME_S')
threads = int('$T')
sp = (t1 / tp) if tp > 0 else 1.0
eff = (sp / threads) * 100.0 if threads > 0 else 100.0
print(f'{sp:.2f},{eff:.1f}')
")
    SPEEDUP=$(echo "$STATS" | cut -d',' -f1)
    EFF=$(echo "$STATS" | cut -d',' -f2)

    # VTune summary used for the metrics below.
    vtune -report summary -report-knob show-issues=false -result-dir "$RESULT_DIR" > "$REPORT_FILE" 2>/dev/null || true

    # Microarchitecture counters from the VTune summary.
    CPI=$(grep "CPI Rate:" "$REPORT_FILE" | head -n 1 | awk '{print $3}' || echo "N/A")
    CORES=$(grep "Effective Physical Core Utilization:" "$REPORT_FILE" | head -n 1 | grep -o '([0-9.]*' | tr -d '(' || echo "N/A")
    MEM_BOUND=$(grep "Memory Bound:" "$REPORT_FILE" | head -n 1 | awk '{print $3}' | tr -d '%' || echo "N/A")
    NUMA=$(grep "NUMA: % of Remote Accesses:" "$REPORT_FILE" | head -n 1 | awk '{print $6}' | tr -d '%' || echo "0.0")
    DRAM_BW=$(grep "DRAM, GB/sec" "$REPORT_FILE" | head -n 1 | awk '{print $4}' || echo "N/A")

    echo "$T,$TIME_S,$SPEEDUP,$EFF,$CPI,$CORES,$MEM_BOUND,$NUMA,$DRAM_BW" >> "$CSV_FILE"

    echo ""
    echo "done $T threads  time=${TIME_S}s  speedup=${SPEEDUP}x  eff=${EFF}%  cpi=${CPI}  cores=${CORES}"
    echo ""
done

# Text summary of the CSV.
python3 -c "
import csv

lines = []
with open('$CSV_FILE', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lines.append(row)

output = []
output.append('=' * 88)
output.append('            RESULT TABLE (BUFFED_MATRIX)')
output.append('=' * 88)
output.append(f'{\"Threads\":<8} | {\"Time (s)\":<10} | {\"Speedup\":<8} | {\"Efficiency\":<11} | {\"CPI\":<6} | {\"Cores\":<8} | {\"NUMA rem\":<9} | {\"DRAM BW\":<8}')
output.append('-' * 88)

for r in lines:
    t = r['Threads']
    ts = f\"{float(r['Time_s']):.2f} s\"
    sp = f\"{float(r['Speedup']):.2f}x\"
    eff = f\"{float(r['Efficiency_pct']):.1f}%\"
    cpi = r['CPI']
    cores = r['Cores_Util']
    numa = f\"{r['NUMA_Remote_pct']}%\"
    bw = f\"{r['DRAM_BW_GBs']} GB/s\" if r['DRAM_BW_GBs'] != 'N/A' else 'N/A'
    output.append(f'{t:<8} | {ts:<10} | {sp:<8} | {eff:<11} | {cpi:<6} | {cores:<8} | {numa:<9} | {bw:<8}')

output.append('=' * 88)
output.append('Notes:')
output.append('  • 20 threads: one thread per physical core.')
output.append('  • 21 threads: the first thread that shares a physical core.')
output.append('  • 32 threads: 20 physical cores plus 12 extra logical threads.')
output.append('=' * 88)

text = '\n'.join(output)
print(text)

with open('$SUMMARY_FILE', 'w') as f_out:
    f_out.write(text + '\n')
"

echo ""
echo "========================================================================"
echo "  All runs finished."
echo "  Results:"
echo "    • $CSV_FILE"
echo "    • $SUMMARY_FILE"
echo "========================================================================"
echo ""
