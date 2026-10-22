#include "omega_matrix.h"
#include <omp.h>

Omega_Matrix::Omega_Matrix(int r, int c) 
    : rows(r), cols(c), data(r * c, 0.1f), bias(c, 0.4f) {}

std::vector<float> Omega_Matrix::forward_pass(const std::vector<float>& input) const {
    std::vector<float> output(cols);

    for (int i = 0; i < cols; i++) {
        float sum = bias[i];

        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int j = 0; j < rows; j++) {
            sum += input[j] * (*this)(j, i);
        }

        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    return output;
}
