#include "gpu.h"

#include <cstdint>
#include <stdexcept>

void cuCheck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    throw std::runtime_error(std::string(what) + " failed: " + (name ? name : "?") + " (" + (desc ? desc : "") + ")");
}

Gpu::Gpu() {
    cuCheck(cuInit(0), "cuInit");
    cuCheck(cuDeviceGet(&dev_, 0), "cuDeviceGet");
    // The "primary context" is the same context the CUDA runtime API would use.
    cuCheck(cuDevicePrimaryCtxRetain(&ctx_, dev_), "cuDevicePrimaryCtxRetain");
    cuCheck(cuCtxSetCurrent(ctx_), "cuCtxSetCurrent");
}

Gpu::~Gpu() { cuDevicePrimaryCtxRelease(dev_); }

std::string Gpu::name() const {
    char buf[256];
    cuCheck(cuDeviceGetName(buf, sizeof buf, dev_), "cuDeviceGetName");
    return buf;
}

std::string Gpu::arch() const {
    int major = 0, minor = 0;
    cuCheck(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_), "cuDeviceGetAttribute");
    cuCheck(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev_), "cuDeviceGetAttribute");
    return "sm_" + std::to_string(major) + std::to_string(minor);
}

Module::Module(const std::string& ptx) {
    // Ask the JIT for its logs. The info log (with LOG_VERBOSE) reports registers used per kernel.
    char infoLog[8192] = {};
    char errorLog[8192] = {};
    CUjit_option options[] = {
        CU_JIT_INFO_LOG_BUFFER,  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE,
    };
    void* values[] = {
        infoLog,  reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof infoLog)),
        errorLog, reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof errorLog)),
        reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
    };
    CUresult r = cuModuleLoadDataEx(&mod_, ptx.c_str(), 5, options, values);
    log_ = infoLog;
    if (r != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(r, &name);
        throw std::runtime_error(std::string("PTX JIT compilation failed (") + (name ? name : "?") + "):\n" + errorLog);
    }
}

Module::~Module() {
    if (mod_) cuModuleUnload(mod_);
}

Module::Module(Module&& other) noexcept : mod_(other.mod_), log_(std::move(other.log_)) { other.mod_ = nullptr; }

CUfunction Module::kernel(const std::string& name) const {
    CUfunction f = nullptr;
    cuCheck(cuModuleGetFunction(&f, mod_, name.c_str()), ("cuModuleGetFunction(" + name + ")").c_str());
    return f;
}

void launch(CUfunction f, unsigned blocks, unsigned threads, std::vector<void*> args) {
    cuCheck(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, nullptr, args.data(), nullptr), "cuLaunchKernel");
    cuCheck(cuCtxSynchronize(), "kernel execution");
}
