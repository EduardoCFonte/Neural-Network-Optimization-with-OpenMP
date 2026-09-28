#include "buffed_matrix.h"
#include <omp.h>

Buffed_Matrix::Buffed_Matrix(int r, int c) 
    : rows(r), cols(c), data(r * c, 0.1f), bias(c, 0.4f) {}

std::vector<float> Buffed_Matrix::forward_pass(const std::vector<float>& input) const {
    std::vector<float> output(cols);

    // Parallelize the outer loop across available OpenMP threads
    #pragma omp parallel for
    for (int i = 0; i < cols; i++) {
        float sum = bias[i];

        // Dot product: multiply each input by its corresponding weight
        for (int j = 0; j < rows; j++) {
            sum += input[j] * (*this)(j, i);
        }

        // ReLU activation function: max(0, sum)
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    return output;
}