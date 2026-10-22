#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <omp.h>
#include "buffed_matrix.h"

int main(int argc, char* argv[]) {
    int NUM_THREADS = 1;
    if (argc > 1) {
        NUM_THREADS = std::atoi(argv[1]);
        if (NUM_THREADS <= 0) NUM_THREADS = 1;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

    const int l0 = 1000;
    const int l1 = 10000;
    const int l2 = 10000;
    const int l3 = 10000;
    const int l4 = 10000;
    const int l5 = 10000;
    const int l6 = 10000;
    const int l7 = 10000;
    const int l8 = 1000;

    const int BATCH_SIZE = 20;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  BUFFED_MATRIX (OUTER LOOP - " << NUM_THREADS << " THREADS)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Topology (8 Layers):" << std::endl;
    std::cout << "  " << l0 << " -> " << l1 << " -> " << l2 << " -> " << l3 
              << " -> " << l4 << " -> " << l5 << " -> " << l6 << " -> " << l7 
              << " -> " << l8 << std::endl;
    std::cout << "OpenMP Threads Active: " << NUM_THREADS << std::endl;

    std::cout << "\nAllocating 8 layers with Buffed_Matrix..." << std::endl;
    Buffed_Matrix w1(l0, l1);
    Buffed_Matrix w2(l1, l2);
    Buffed_Matrix w3(l2, l3);
    Buffed_Matrix w4(l3, l4);
    Buffed_Matrix w5(l4, l5);
    Buffed_Matrix w6(l5, l6);
    Buffed_Matrix w7(l6, l7);
    Buffed_Matrix w8(l7, l8);

    size_t total_weights = w1.weight_count() + w2.weight_count() + w3.weight_count() +
                           w4.weight_count() + w5.weight_count() + w6.weight_count() +
                           w7.weight_count() + w8.weight_count();
    double total_mb = (total_weights * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "Total parameters: " << total_weights << " (" << std::fixed << std::setprecision(1) << total_mb << " MB ~ " << (total_mb / 1024.0) << " GB)" << std::endl;
    std::cout << "Batch size (Inferences): " << BATCH_SIZE << std::endl;

    std::vector<float> input(l0, 0.5f);

    std::cout << "\nStarting parallel inference on outer loop over " << BATCH_SIZE << " samples..." << std::endl;

    // Warm up the thread pool before timing.
    {
        std::vector<float> a1 = w1.forward_pass(input);
        std::vector<float> a2 = w2.forward_pass(a1);
        std::vector<float> a3 = w3.forward_pass(a2);
        std::vector<float> a4 = w4.forward_pass(a3);
        std::vector<float> a5 = w5.forward_pass(a4);
        std::vector<float> a6 = w6.forward_pass(a5);
        std::vector<float> a7 = w7.forward_pass(a6);
        w8.forward_pass(a7);
    }

    double t_start = omp_get_wtime();

    std::vector<float> final_output;
    for (int b = 0; b < BATCH_SIZE; b++) {
        std::vector<float> a1 = w1.forward_pass(input);
        std::vector<float> a2 = w2.forward_pass(a1);
        std::vector<float> a3 = w3.forward_pass(a2);
        std::vector<float> a4 = w4.forward_pass(a3);
        std::vector<float> a5 = w5.forward_pass(a4);
        std::vector<float> a6 = w6.forward_pass(a5);
        std::vector<float> a7 = w7.forward_pass(a6);
        final_output = w8.forward_pass(a7);
    }

    double t_end = omp_get_wtime();
    double total_ms = (t_end - t_start) * 1000.0;
    double total_sec = total_ms / 1000.0;

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "                    EXECUTION SUMMARY" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Parallel Time (Buffed " << NUM_THREADS << " Threads): " << std::fixed << std::setprecision(2) << total_sec << " seconds (" << total_ms << " ms)" << std::endl;
    std::cout << "Average Time / Pass: " << std::fixed << std::setprecision(2) << (total_ms / BATCH_SIZE) << " ms" << std::endl;
    std::cout << "RESULT_CSV:" << NUM_THREADS << "," << total_sec << "," << (total_ms / BATCH_SIZE) << std::endl;

    return 0;
}
