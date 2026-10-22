#!/bin/bash
# Delete regenerable profiler output.
# Usage: bash clean_project.sh [vtune_raw|logs|old|all]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-vtune_raw}"

remove_vtune_raw() {
    local n=0
    while IFS= read -r d; do
        echo "rm -rf $d"
        rm -rf "$d"
        n=$((n + 1))
    done < <(find . -maxdepth 3 -type d -name 'r_*' 2>/dev/null | grep -E './vtune_' || true)
    echo "Pastas brutas do VTune (r_*): $n removida(s)."
}

remove_logs() {
    local n
    n=$(find vtune_* -maxdepth 1 -name 'run_*.log' 2>/dev/null | wc -l)
    find vtune_* -maxdepth 1 -name 'run_*.log' -delete 2>/dev/null || true
    echo "Logs run_*.log em vtune_*: $n removido(s)."
}

remove_old_dupes() {
    rm -rf vtune_sweep
    rm -rf slides_imagens
    rm -f benchmark_results.csv benchmark_summary.txt
    rm -f grafico_cpi.svg grafico_eficiencia.svg grafico_speedup.svg grafico_tempo_execucao.svg
    rm -f painel_completo_4graficos.svg
    echo "Removidos: vtune_sweep/, slides_imagens/, CSV/SVG antigos na raiz."
    echo "(Published charts stay in resultados/.)"
}

case "$MODE" in
    vtune_raw|"")
        remove_vtune_raw
        ;;
    logs)
        remove_vtune_raw
        remove_logs
        ;;
    old)
        remove_old_dupes
        ;;
    all)
        remove_vtune_raw
        remove_logs
        remove_old_dupes
        ;;
    *)
        echo "Usage: bash clean_project.sh [vtune_raw|logs|old|all]"
        exit 1
        ;;
esac

echo ""
echo "Kept:"
echo "  Code: main.cpp, sweep_bench.cpp, matrix/, OpenMP_matrix.cpp/, CMakeLists.txt"
echo "  Scripts: run_*vtune*.sh, run_benchmark_*.sh"
echo "  Summaries: vtune_*/resultados*.csv, relatorio*.md, vtune_*/reports/"
echo "  Slides: apresentacao/"
