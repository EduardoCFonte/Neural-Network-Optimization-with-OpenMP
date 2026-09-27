#include "matrix.h"

Matrix::Matrix(int r, int c) 
    : rows(r), cols(c), data(r * c, 0.1f) {
}