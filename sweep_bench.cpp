#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

enum class Mode { Outer, Reduction, Simd, OuterSimd };

struct Layer {
    int rows = 0;
    int cols = 0;
    float* w = nullptr;
    std::vector<float> bias;

    Layer() = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    Layer(Layer&& other) noexcept
        : rows(other.rows), cols(other.cols), w(other.w), bias(std::move(other.bias)) {
        other.w = nullptr;
    }

    Layer& operator=(Layer&& other) noexcept {
        if (this != &other) {
            std::free(w);
            rows = other.rows;
            cols = other.cols;
            w = other.w;
            bias = std::move(other.bias);
            other.w = nullptr;
        }
        return *this;
    }

    ~Layer() { std::free(w); }
};

static const char* mode_name(Mode mode) {
    switch (mode) {
    case Mode::Outer: return "outer";
    case Mode::Reduction: return "reduction";
    case Mode::Simd: return "simd";
    case Mode::OuterSimd: return "outer_simd";
    }
    return "unknown";
}

static bool parse_mode(const std::string& text, Mode& mode) {
    if (text == "outer") mode = Mode::Outer;
    else if (text == "reduction") mode = Mode::Reduction;
    else if (text == "simd") mode = Mode::Simd;
    else if (text == "outer_simd") mode = Mode::OuterSimd;
    else return false;
    return true;
}

struct BenchProfile {
    const char* id;
    const char* description;
    std::vector<int> dims;
    int batch;
};

static bool parse_benchmark(const std::string& text, BenchProfile& profile) {
    if (text == "standard") {
        profile = {
            "standard",
            "8 layers 1000->10000 (x6)->1000, 1 inference",
            {1000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000},
            1,
        };
        return true;
    }
    if (text == "wide5") {
        profile = {
            "wide5",
            "5 layers 8000->52000 (x3)->8000, 1 inference",
            {8000, 52000, 52000, 52000, 52000, 8000},
            1,
        };
        return true;
    }
    return false;
}

static int static_chunk(int n, int threads) {
    return (n + threads - 1) / threads;
}

static void touch_columns(Layer& layer, int threads) {
    const int chunk = static_chunk(layer.cols, threads);
    #pragma omp parallel for schedule(static, chunk) num_threads(threads)
    for (int i = 0; i < layer.cols; ++i) {
        float* column = layer.w + static_cast<size_t>(i) * static_cast<size_t>(layer.rows);
        for (int j = 0; j < layer.rows; ++j) {
            column[j] = 0.1f;
        }
    }
}

static void touch_row_chunks(Layer& layer, int threads) {
    const int chunk = static_chunk(layer.rows, threads);
    #pragma omp parallel for schedule(static, chunk) num_threads(threads)
    for (int j = 0; j < layer.rows; ++j) {
        for (int i = 0; i < layer.cols; ++i) {
            layer.w[static_cast<size_t>(i) * static_cast<size_t>(layer.rows) + j] = 0.1f;
        }
    }
}

static Layer make_layer(int rows, int cols, Mode mode, int threads) {
    Layer layer;
    layer.rows = rows;
    layer.cols = cols;
    layer.bias.assign(cols, 0.4f);

    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    size_t bytes = (n * sizeof(float) + 63u) & ~size_t{63};
    if (bytes == 0) bytes = 64;
    layer.w = static_cast<float*>(std::aligned_alloc(64, bytes));
    if (!layer.w) throw std::bad_alloc();

    if (mode == Mode::Outer || mode == Mode::OuterSimd) {
        touch_columns(layer, threads);
    } else {
        touch_row_chunks(layer, threads);
    }
    return layer;
}

__attribute__((optimize("no-tree-vectorize")))
static std::vector<float> forward_outer(const Layer& layer, const float* in, int threads) {
    std::vector<float> output(layer.cols);
    const int height = layer.rows;
    const int chunk = static_chunk(layer.cols, threads);

    #pragma omp parallel for schedule(static, chunk) num_threads(threads)
    for (int i = 0; i < layer.cols; ++i) {
        const float* column = layer.w + static_cast<size_t>(i) * static_cast<size_t>(height);
        float sum = layer.bias[i];
        for (int j = 0; j < height; ++j) {
            sum += in[j] * column[j];
        }
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    return output;
}

__attribute__((optimize("no-tree-vectorize")))
static std::vector<float> forward_reduction(const Layer& layer, const float* in, int threads) {
    std::vector<float> output(layer.cols);
    const int height = layer.rows;
    const int chunk = static_chunk(height, threads);

    for (int i = 0; i < layer.cols; ++i) {
        const float* column = layer.w + static_cast<size_t>(i) * static_cast<size_t>(height);
        float sum = layer.bias[i];
        #pragma omp parallel for reduction(+:sum) schedule(static, chunk) num_threads(threads)
        for (int j = 0; j < height; ++j) {
            sum += in[j] * column[j];
        }
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    return output;
}

static std::vector<float> forward_simd(const Layer& layer, const float* in, int threads) {
    std::vector<float> output(layer.cols);
    const int height = layer.rows;
    const int chunk = static_chunk(height, threads);

    for (int i = 0; i < layer.cols; ++i) {
        const float* column = layer.w + static_cast<size_t>(i) * static_cast<size_t>(height);
        float sum = layer.bias[i];
        #pragma omp parallel for simd reduction(+:sum) schedule(static, chunk) num_threads(threads)
        for (int j = 0; j < height; ++j) {
            sum += in[j] * column[j];
        }
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    return output;
}

static std::vector<float> forward_outer_simd(const Layer& layer, const float* in, int threads) {
    std::vector<float> output(layer.cols);
    const int height = layer.rows;
    const int chunk = static_chunk(layer.cols, threads);

    #pragma omp parallel for schedule(static, chunk) num_threads(threads)
    for (int i = 0; i < layer.cols; ++i) {
        const float* column = layer.w + static_cast<size_t>(i) * static_cast<size_t>(height);
        float sum = layer.bias[i];
        #pragma omp simd
        for (int j = 0; j < height; ++j) {
            sum += in[j] * column[j];
        }
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    return output;
}

static std::vector<float> forward(const Layer& layer, const std::vector<float>& input, Mode mode, int threads) {
    const float* in = input.data();
    switch (mode) {
    case Mode::Outer: return forward_outer(layer, in, threads);
    case Mode::Reduction: return forward_reduction(layer, in, threads);
    case Mode::Simd: return forward_simd(layer, in, threads);
    case Mode::OuterSimd: return forward_outer_simd(layer, in, threads);
    }
    return {};
}

static void usage() {
    std::cerr << "Usage: nn_sweep <outer|reduction|simd|outer_simd> <threads> [standard|wide5]\n"
              << "  standard    8 layers, 1000->10000 (x6)->1000, 1 inference\n"
              << "  wide5       5 layers, 8000->52000 (x3)->8000, 1 inference\n"
              << "  outer       parallel neuron loop, scalar dot product\n"
              << "  reduction   OpenMP reduction on the dot product\n"
              << "  simd        reduction plus SIMD on the dot product\n"
              << "  outer_simd  parallel neuron loop, SIMD dot product, no reduction\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        usage();
        return 1;
    }

    Mode mode;
    if (!parse_mode(argv[1], mode)) {
        usage();
        return 1;
    }

    const int threads = std::atoi(argv[2]);
    if (threads <= 0) {
        usage();
        return 1;
    }

    std::string bench_id = "standard";
    if (argc == 4) {
        bench_id = argv[3];
    } else if (const char* env = std::getenv("SWEEP_BENCH")) {
        bench_id = env;
    }

    BenchProfile profile;
    if (!parse_benchmark(bench_id, profile)) {
        usage();
        return 1;
    }

    if (bench_id == "standard") {
        if (const char* env = std::getenv("SWEEP_DIM")) {
            const int hidden = std::atoi(env);
            if (hidden > 0) {
                const int edge = hidden < 1000 ? hidden : 1000;
                profile.dims.assign(9, hidden);
                profile.dims.front() = edge;
                profile.dims.back() = edge;
            }
        }
    }

    const std::vector<int>& shape = profile.dims;
    const int batch = profile.batch;
    const int num_layers = static_cast<int>(shape.size()) - 1;

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    std::cout << "BENCHMARK:" << profile.id << "\n";
    std::cout << "BENCH_DESC:" << profile.description << "\n";
    std::cout << "MODE:" << mode_name(mode) << "\n";
    std::cout << "THREADS:" << threads << "\n";
    std::cout << "TOPOLOGY:" << shape[0];
    for (size_t i = 1; i < shape.size(); ++i) std::cout << "->" << shape[i];
    std::cout << "\nBATCH:" << batch << std::endl;

    std::vector<Layer> layers;
    layers.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(make_layer(shape[i], shape[i + 1], mode, threads));
    }

    std::vector<float> input(shape[0], 0.5f);
    double t0 = omp_get_wtime();
    std::vector<float> output = input;
    for (const Layer& layer : layers) {
        output = forward(layer, output, mode, threads);
    }
    double t1 = omp_get_wtime();

    double checksum = 0.0;
    for (float value : output) checksum += static_cast<double>(value);

    const double seconds = t1 - t0;
    const double ms_per_pass = (seconds * 1000.0) / static_cast<double>(batch);

    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << "CHECKSUM:" << checksum << "\n";
    std::cout.precision(6);
    std::cout << "RESULT_CSV:" << profile.id << "," << mode_name(mode) << "," << threads << ","
              << seconds << "," << ms_per_pass << std::endl;

    if (!std::isfinite(checksum)) return 2;
    return 0;
}
