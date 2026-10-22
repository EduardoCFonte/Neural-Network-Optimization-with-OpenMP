#include "buffed_matrix.h"

#include <cstdlib>
#include <new>
#include <omp.h>

Buffed_Matrix::Buffed_Matrix(int r, int c)
    : rows(r), cols(c), bias(c, 0.4f), weights(nullptr) {
    const size_t n = weight_count();
    const size_t bytes = (n * sizeof(float) + 63u) & ~size_t{63};
    weights = static_cast<float*>(std::aligned_alloc(64, bytes));
    if (!weights) {
        throw std::bad_alloc();
    }

    // First-touch each column on the thread that will own it.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cols; ++i) {
        float* column = weights + static_cast<size_t>(i) * static_cast<size_t>(rows);
        for (int j = 0; j < rows; ++j) {
            column[j] = 0.1f;
        }
    }
}

Buffed_Matrix::~Buffed_Matrix() {
    std::free(weights);
}

std::vector<float> Buffed_Matrix::forward_pass(const std::vector<float>& input) const {
    std::vector<float> output(cols);
    const float* in = input.data();
    const int height = rows;

    // One neuron per iteration. The weight column is contiguous.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cols; ++i) {
        const float* column = weights + static_cast<size_t>(i) * static_cast<size_t>(height);
        float sum = bias[i];
        for (int j = 0; j < height; ++j) {
            sum += in[j] * column[j];
        }
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    return output;
}
