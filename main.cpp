#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include "matrix.h"
#include "buffed_matrix.h"

int main() {
    int input_dim = 500;
    int first_layer = 3000;
    int second_layer = 3000;
    int output_dim = 300;
    const int NUM_THREADS = 16;

    std::cout << "==========================================================" << std::endl;
    std::cout << "     BENCHMARK: SEQUENTIAL VS OPENMP (" << NUM_THREADS << " THREADS)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Topology: " << input_dim << " -> " << first_layer 
              << " -> " << second_layer << " -> " << output_dim << std::endl;
    std::cout << "Max Hardware Threads Available: " << omp_get_max_threads() << std::endl;

    // 1. Initializing Sequential Layers
    Matrix seq_l1(input_dim, first_layer);
    Matrix seq_l2(first_layer, second_layer);
    Matrix seq_l3(second_layer, output_dim);

    // 2. Initializing Parallel Layers (Buffed_Matrix with OpenMP)
    Buffed_Matrix par_l1(input_dim, first_layer);
    Buffed_Matrix par_l2(first_layer, second_layer);
    Buffed_Matrix par_l3(second_layer, output_dim);

    // Simulated input vector
    std::vector<float> input(input_dim, 0.5f);

    // ----------------------------------------------------
    // [1] SEQUENTIAL INFERENCE
    // ----------------------------------------------------
    std::cout << "\n[1] Running Sequential Inference..." << std::endl;
    
    // Warmup
    seq_l1.forward_pass(input);

    double t_seq_start = omp_get_wtime();
    std::vector<float> seq_out1 = seq_l1.forward_pass(input);
    std::vector<float> seq_out2 = seq_l2.forward_pass(seq_out1);
    std::vector<float> seq_out3 = seq_l3.forward_pass(seq_out2);
    double t_seq_end = omp_get_wtime();

    double t_seq_ms = (t_seq_end - t_seq_start) * 1000.0;
    std::cout << "Sequential Time: " << std::fixed << std::setprecision(2) 
              << t_seq_ms << " ms" << std::endl;

    // ----------------------------------------------------
    // [2] PARALLEL INFERENCE (16 Threads)
    // ----------------------------------------------------
    std::cout << "\n[2] Running Parallel Inference with " << NUM_THREADS << " Threads..." << std::endl;
    omp_set_num_threads(NUM_THREADS);

    // Warmup
    par_l1.forward_pass(input);

    double t_par_start = omp_get_wtime();
    std::vector<float> par_out1 = par_l1.forward_pass(input);
    std::vector<float> par_out2 = par_l2.forward_pass(par_out1);
    std::vector<float> par_out3 = par_l3.forward_pass(par_out2);
    double t_par_end = omp_get_wtime();

    double t_par_ms = (t_par_end - t_par_start) * 1000.0;
    double speedup = t_seq_ms / t_par_ms;
    double efficiency = (speedup / NUM_THREADS) * 100.0;

    std::cout << "Parallel Time:   " << std::fixed << std::setprecision(2) 
              << t_par_ms << " ms" << std::endl;

    // ----------------------------------------------------
    // [3] BENCHMARK RESULTS & VERIFICATION
    // ----------------------------------------------------
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "                    SUMMARY RESULTS" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Sequential Time: " << t_seq_ms << " ms" << std::endl;
    std::cout << "Parallel Time:   " << t_par_ms << " ms (" << NUM_THREADS << " threads)" << std::endl;
    std::cout << "Speedup:         " << std::setprecision(2) << speedup << "x faster" << std::endl;
    std::cout << "Efficiency:      " << std::setprecision(2) << efficiency << "%" << std::endl;

    // Verify numerical correctness
    bool match = true;
    for (size_t i = 0; i < seq_out3.size(); i++) {
        if (std::abs(seq_out3[i] - par_out3[i]) > 1e-4f) {
            match = false;
            break;
        }
    }

    std::cout << "Correctness:     " 
              << (match ? "PASSED (Outputs match exactly!)" : "FAILED (Output mismatch!)") 
              << std::endl;

    return 0;
}
