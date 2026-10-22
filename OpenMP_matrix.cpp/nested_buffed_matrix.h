#pragma once

#include <vector>
#include <cstddef>

class Nested_Buffed_Matrix {
public:
    int rows;
    int cols;
    int outer_threads;
    int inner_threads;
    std::vector<float> data;
    std::vector<float> bias;

    Nested_Buffed_Matrix(int r, int c, int outer_th = 2, int inner_th = 2);
    ~Nested_Buffed_Matrix() = default;

    // Element at row r, column c.
    inline float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    // Y = ReLU(X * W + b), nested over neurons and the dot product.
    std::vector<float> forward_pass(const std::vector<float>& input) const;
};
