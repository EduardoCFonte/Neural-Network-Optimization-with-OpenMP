#include "matrix.h"

Matrix::Matrix(int r, int c) 
    : rows(r), cols(c), data(r * c, 0.1f), bias(c, 0.4f) {}

std::vector<float> Matrix::forward_pass(const std::vector<float>& input) const {
    std::vector<float> output(cols);

    // Loop through each output neuron
    for (int i = 0; i < cols; i++) {
        // Start with the neuron's bias
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