#include <iostream>
#include <vector>
#include <omp.h>
#include "matrix.h"

int main() {
    int input_dim = 500;
    int first_layer = 3000;
    int second_layer = 3000;
    int output_dim = 300;

    std::cout << "========================================" << std::endl;
    std::cout << "     NEURAL NETWORK INITIALIZATION" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Topology: " << input_dim << " -> " << first_layer 
              << " -> " << second_layer << " -> " << output_dim << std::endl;

    // Creating weight matrices for each layer
    Matrix input_l1(input_dim, first_layer);
    std::cout << "Matrix W1 created: " << input_l1.rows << "x" << input_l1.cols 
              << " (" << input_l1.data.size() << " weights)" << std::endl;
    
    Matrix l1_l2(first_layer, second_layer);
    std::cout << "Matrix W2 created: " << l1_l2.rows << "x" << l1_l2.cols 
              << " (" << l1_l2.data.size() << " weights)" << std::endl;

    Matrix l2_output(second_layer, output_dim);
    std::cout << "Matrix W3 created: " << l2_output.rows << "x" << l2_output.cols 
              << " (" << l2_output.data.size() << " weights)" << std::endl;

    // Simulated input vector (500 elements initialized to 0.5)
    std::vector<float> Input(input_dim, 0.5f);

    std::cout << "\n========================================" << std::endl;
    std::cout << "     STARTING SEQUENTIAL INFERENCE" << std::endl;
    std::cout << "========================================" << std::endl;

    // Measuring total time and time per layer
    double t_total_start = omp_get_wtime();

    // 1. Layer 1 forward pass
    double t1_start = omp_get_wtime();
    std::vector<float> Input_L1 = input_l1.forward_pass(Input);
    double t1_end = omp_get_wtime();
    std::cout << "Layer 1 (500 -> 3000):   " << (t1_end - t1_start) * 1000.0 << " ms" << std::endl;

    // 2. Layer 2 forward pass (computational bottleneck: 3000 x 3000)
    double t2_start = omp_get_wtime();
    std::vector<float> Input_L2 = l1_l2.forward_pass(Input_L1);
    double t2_end = omp_get_wtime();
    std::cout << "Layer 2 (3000 -> 3000): " << (t2_end - t2_start) * 1000.0 << " ms" << std::endl;

    // 3. Output layer forward pass
    double t3_start = omp_get_wtime();
    std::vector<float> Output = l2_output.forward_pass(Input_L2);
    double t3_end = omp_get_wtime();
    std::cout << "Layer 3 (3000 -> 300):  " << (t3_end - t3_start) * 1000.0 << " ms" << std::endl;

    double t_total_end = omp_get_wtime();
    double total_time = (t_total_end - t_total_start) * 1000.0;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total Inference Time: " << total_time << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Printing a sample of the output values
    std::cout << "\nSample of first 5 outputs (out of " << Output.size() << "):" << std::endl;
    for (int i = 0; i < 5 && i < (int)Output.size(); i++) {
        std::cout << "  Output[" << i << "] = " << Output[i] << std::endl;
    }

    return 0;
}
