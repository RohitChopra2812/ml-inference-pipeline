cmake_minimum_required(VERSION 3.14)
project(ml_inference)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)

# Path to ONNX Runtime SDK we downloaded
set(ORT_ROOT /Users/I760158/ml_inference_env/onnxruntime-osx-arm64-1.18.0)

# Tell compiler where to find ONNX Runtime headers
include_directories(${ORT_ROOT}/include)

# Create executable from inference.cpp
add_executable(inference inference.cpp)

# Link against ONNX Runtime library
target_link_libraries(inference ${ORT_ROOT}/lib/libonnxruntime.dylib)
