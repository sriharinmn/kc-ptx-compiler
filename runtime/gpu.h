#pragma once

// A tiny wrapper around the CUDA Driver API (cuda.h).
//
// Why the Driver API and not the usual CUDA Runtime (cudaMalloc, <<<>>>)?
// The runtime API expects kernels compiled by nvcc. The driver API lets us
// hand PTX text to the driver ourselves (cuModuleLoadDataEx): the driver
// JIT-compiles it with its built-in ptxas for the GPU in this machine.
// That is exactly the path PTX from non-NVIDIA compilers takes.

#include <cuda.h>

#include <string>
#include <vector>

// Throws std::runtime_error with the CUDA error name if r is not CUDA_SUCCESS.
void cuCheck(CUresult r, const char* what);

class Gpu {
public:
    Gpu();  // initializes the driver and uses GPU 0
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    std::string name() const;
    std::string arch() const;  // e.g. "sm_89"

private:
    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
};

// GPU code loaded onto the GPU.
class Module {
public:
    // JIT-compiles; throws with ptxas' error log on failure. ptxasOptLevel 0-4 (4 = full
    // optimization, the default); 0 turns ptxas' optimizer off, which helps tell whether a
    // wrong result comes from the PTX or from ptxas.
    explicit Module(const std::string& ptx, int ptxasOptLevel = 4);
    explicit Module(const std::vector<char>& cubin);  // already compiled machine code (see ptx_compiler.h)
    ~Module();
    Module(Module&& other) noexcept;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    CUfunction kernel(const std::string& name) const;
    const std::string& jitLog() const { return log_; }  // ptxas info: registers used, etc.

private:
    CUmodule mod_ = nullptr;
    std::string log_;
};

// An array in GPU memory.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        cuCheck(cuMemAlloc(&ptr_, bytes()), "cuMemAlloc");
        cuCheck(cuMemsetD8(ptr_, 0, bytes()), "cuMemsetD8");
    }
    explicit DeviceBuffer(const std::vector<T>& host) : count_(host.size()) {
        cuCheck(cuMemAlloc(&ptr_, bytes()), "cuMemAlloc");
        cuCheck(cuMemcpyHtoD(ptr_, host.data(), bytes()), "cuMemcpyHtoD");
    }
    ~DeviceBuffer() { cuMemFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const std::vector<T>& host) { cuCheck(cuMemcpyHtoD(ptr_, host.data(), bytes()), "cuMemcpyHtoD"); }
    void clear() { cuCheck(cuMemsetD8(ptr_, 0, bytes()), "cuMemsetD8"); }

    std::vector<T> download() const {
        std::vector<T> host(count_);
        cuCheck(cuMemcpyDtoH(host.data(), ptr_, bytes()), "cuMemcpyDtoH");
        return host;
    }

    // Kernel arguments are passed as pointers to the argument values.
    void* arg() { return &ptr_; }

private:
    size_t bytes() const { return count_ * sizeof(T); }
    CUdeviceptr ptr_ = 0;
    size_t count_;
};

// Runs kernel f on a 1-D grid of 'blocks' blocks with 'threads' threads each, then waits for it.
// args[i] points to the value of the i-th kernel parameter.
void launch(CUfunction f, unsigned blocks, unsigned threads, std::vector<void*> args);

// Average time of one launch in milliseconds, measured on the GPU with CUDA events
// (after one warm-up launch). The launches run back to back, without waiting in between.
float timeKernelMs(CUfunction f, unsigned blocks, unsigned threads, std::vector<void*> args, int reps);
