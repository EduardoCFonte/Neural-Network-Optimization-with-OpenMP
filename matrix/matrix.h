#pragma once

#include <vector>
#include <cstddef>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r, int c);
    ~Matrix() = default;

    inline float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }
};