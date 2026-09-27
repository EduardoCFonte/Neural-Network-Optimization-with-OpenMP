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

    // Teste instanciando uma matriz de pesos da primeira camada (500 x 3000)
    Matrix w1(input_dim, first_layer);
    std::cout << "Matriz W1 criada com sucesso: " << w1.rows << "x" << w1.cols 
              << " (" << w1.data.size() << " elementos)" << std::endl;

    // Teste se o OpenMP está habilitado
    #ifdef _OPENMP
        std::cout << "OpenMP ativo! Threads maximas disponiveis: " 
                  << omp_get_max_threads() << std::endl;
    #else
        std::cout << "OpenMP NAO habilitado!" << std::endl;
    #endif

    return 0;
}
