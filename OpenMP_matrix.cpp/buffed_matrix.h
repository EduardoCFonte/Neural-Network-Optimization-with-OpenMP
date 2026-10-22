#pragma once

#include <cstddef>
#include <vector>

class Buffed_Matrix {
public:
    int rows;
    int cols;
    std::vector<float> bias;

    Buffed_Matrix(int r, int c);
    ~Buffed_Matrix();

    Buffed_Matrix(const Buffed_Matrix&) = delete;
    Buffed_Matrix& operator=(const Buffed_Matrix&) = delete;

    size_t weight_count() const {
        return static_cast<size_t>(rows) * static_cast<size_t>(cols);
    }

    // Column c is contiguous.
    inline float& operator()(int r, int c) {
        return weights[static_cast<size_t>(c) * static_cast<size_t>(rows) + r];
    }

    inline const float& operator()(int r, int c) const {
        return weights[static_cast<size_t>(c) * static_cast<size_t>(rows) + r];
    }

    // Y = ReLU(X * W + b)
    std::vector<float> forward_pass(const std::vector<float>& input) const;

private:
    float* weights;
};
