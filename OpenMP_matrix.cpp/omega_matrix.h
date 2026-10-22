#pragma once

#include <vector>
#include <cstddef>

class Omega_Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;
    std::vector<float> bias;

    Omega_Matrix(int r, int c);
    ~Omega_Matrix() = default;

    // Element at row r, column c.
    inline float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    // Y = ReLU(X * W + b), reduced over the inner loop.
    std::vector<float> forward_pass(const std::vector<float>& input) const;
};
