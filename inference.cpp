#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
        labels.push_back(line);
    return labels;
}

// Run inference N times and return average latency in milliseconds
double benchmark(Ort::Session& session,
                 Ort::Value& input_tensor,
                 const char* input_names[],
                 const char* output_names[],
                 int runs = 50) {
    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        session.Run(Ort::RunOptions{nullptr},
                    input_names, &input_tensor, 1,
                    output_names, 1);

        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total / runs;
}

Ort::Session create_session(Ort::Env& env,
                             const char* model_path,
                             GraphOptimizationLevel opt_level,
                             int threads = 1) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetInterOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(opt_level);
    if (opt_level == ORT_ENABLE_ALL)
        opts.EnableCpuMemArena();  // reuse memory allocations
    return Ort::Session(env, model_path, opts);
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inference");

    const char* model_path = "/Users/I760158/ml_inference_env/resnet18.onnx";

    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    size_t input_size = 1 * 3 * 224 * 224;
    std::vector<float> input_data(input_size, 0.5f);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size,
        input_shape.data(), input_shape.size()
    );

    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    // --- Baseline: no optimization, 1 thread ---
    std::cout << "Benchmarking with NO optimization...\n";
    auto session_base = create_session(env, model_path, ORT_DISABLE_ALL, 1);
    double baseline_ms = benchmark(session_base, input_tensor, input_names, output_names);
    std::cout << "Baseline avg latency: " << baseline_ms << " ms\n\n";

    // --- Optimized: full operator fusion + memory arena + multithreading ---
    std::cout << "Benchmarking with FULL optimization (operator fusion + memory arena)...\n";
    auto session_opt = create_session(env, model_path, ORT_ENABLE_ALL, 4);
    double optimized_ms = benchmark(session_opt, input_tensor, input_names, output_names);
    std::cout << "Optimized avg latency: " << optimized_ms << " ms\n\n";

    // --- Report improvement ---
    double improvement = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0;
    std::cout << "Latency improvement: " << improvement << "%\n";

    // --- Post-processing on optimized session ---
    auto labels = load_labels("/Users/I760158/ml_inference_env/imagenet_classes.txt");
    auto output_tensors = session_opt.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int> indices(1000);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return output_data[a] > output_data[b];
    });

    std::cout << "\nTop 5 Predictions (optimized session):\n";
    for (int i = 0; i < 5; i++) {
        int idx = indices[i];
        std::cout << i+1 << ". [" << idx << "] " << labels[idx]
                  << "  (score: " << output_data[idx] << ")\n";
    }

    return 0;
}
