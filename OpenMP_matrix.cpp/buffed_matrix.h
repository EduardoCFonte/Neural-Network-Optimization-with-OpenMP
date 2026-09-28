#pragma once

#include <vector>
#include <cstddef>

class Buffed_Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;
    std::vector<float> bias;

    Buffed_Matrix(int r, int c);
    ~Buffed_Matrix() = default;

    // Fast element access (r: row, c: column)
    inline float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    // Forward pass: Y = ReLU(X * W + b)
    std::vector<float> forward_pass(const std::vector<float>& input) const;
};