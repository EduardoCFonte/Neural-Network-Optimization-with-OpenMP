#pragma once

#include <vector>
#include <cstddef>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;
    std::vector<float> bias;

    Matrix(int r, int c);
    ~Matrix() = default;

    // Element at row r, column c.
    inline float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    // Y = ReLU(X * W + b)
    std::vector<float> forward_pass(const std::vector<float>& input) const;
};