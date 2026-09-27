#include <iostream>
#include <omp.h>
#include "matrix.h"

int main() {
    int input_dim = 500;
    int first_layer = 3000;
    int second_layer = 3000;
    int output_dim = 300;

    std::cout << "=== Teste de Inicialização da Rede ===" << std::endl;
    std::cout << "Camadas: " << input_dim << " -> " << first_layer 
              << " -> " << second_layer << " -> " << output_dim << std::endl;

    Matrix input_l1(input_dim, first_layer);
    std::cout << "Matriz W1 criada com sucesso: " << input_l1.rows << "x" << input_l1.cols 
              << " (" << input_l1.data.size() << " elementos)" << std::endl;
    
    Matrix l1_l2(first_layer, second_layer);
    std::cout << "Matriz W1 criada com sucesso: " << l1_l2.rows << "x" << l1_l2.cols 
    << " (" << l1_l2.data.size() << " elementos)" << std::endl;

    Matrix l2_output(first_layer, output_dim);
    std::cout << "Matriz W1 criada com sucesso: " << l2_output.rows << "x" << l2_output.cols 
    << " (" << l2_output.data.size() << " elementos)" << std::endl;

    std::vector<float> Input(input_dim, 0.5f);

    std::vector<float> Input_L1 = input_l1.forward_pass(Input);
    std::vector<float> Input_L2 = l1_l2.forward_pass(Input_L1);
    std::vector<float> Output   = l2_output.forward_pass(Input_L2);



    return 0;
}
