#include "nested_buffed_matrix.h"
#include <omp.h>

Nested_Buffed_Matrix::Nested_Buffed_Matrix(int r, int c, int outer_th, int inner_th) 
    : rows(r), cols(c), outer_threads(outer_th), inner_threads(inner_th),
      data(r * c, 0.1f), bias(c, 0.4f) {}

std::vector<float> Nested_Buffed_Matrix::forward_pass(const std::vector<float>& input) const {
    std::vector<float> output(cols);

    #pragma omp parallel for num_threads(outer_threads) schedule(static)
    for (int i = 0; i < cols; i++) {
        float sum = bias[i];

        #pragma omp parallel for num_threads(inner_threads) reduction(+:sum) schedule(static)
        for (int j = 0; j < rows; j++) {
            sum += input[j] * (*this)(j, i);
        }

        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    return output;
}
