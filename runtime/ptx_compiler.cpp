#include "ptx_compiler.h"

#include <nvPTXCompiler.h>

#include <regex>
#include <stdexcept>

namespace {

void check(nvPTXCompileResult r, const char* what) {
    if (r != NVPTXCOMPILE_SUCCESS) throw std::runtime_error(std::string(what) + " failed (code " + std::to_string(r) + ")");
}

}  // namespace

CompiledPtx compilePtx(const std::string& ptx, const std::string& arch) {
    nvPTXCompilerHandle compiler = nullptr;
    check(nvPTXCompilerCreate(&compiler, ptx.size(), ptx.c_str()), "nvPTXCompilerCreate");

    // The same options ptxas takes on the command line. Optimization is -O3 by default.
    std::string gpuName = "--gpu-name=" + arch;
    const char* options[] = {gpuName.c_str(), "--verbose"};
    nvPTXCompileResult status = nvPTXCompilerCompile(compiler, 2, options);

    CompiledPtx out;
    size_t size = 0;
    if (status != NVPTXCOMPILE_SUCCESS) {
        nvPTXCompilerGetErrorLogSize(compiler, &size);
        std::string log(size, '\0');
        if (size) nvPTXCompilerGetErrorLog(compiler, log.data());
        nvPTXCompilerDestroy(&compiler);
        throw std::runtime_error("ptxas (nvPTXCompiler) failed:\n" + log);
    }
    check(nvPTXCompilerGetCompiledProgramSize(compiler, &size), "nvPTXCompilerGetCompiledProgramSize");
    out.cubin.resize(size);
    check(nvPTXCompilerGetCompiledProgram(compiler, out.cubin.data()), "nvPTXCompilerGetCompiledProgram");
    check(nvPTXCompilerGetInfoLogSize(compiler, &size), "nvPTXCompilerGetInfoLogSize");
    out.infoLog.assign(size, '\0');
    if (size) check(nvPTXCompilerGetInfoLog(compiler, out.infoLog.data()), "nvPTXCompilerGetInfoLog");
    nvPTXCompilerDestroy(&compiler);
    return out;
}

std::string ptxCompilerVersion() {
    unsigned major = 0, minor = 0;
    check(nvPTXCompilerGetVersion(&major, &minor), "nvPTXCompilerGetVersion");
    return std::to_string(major) + "." + std::to_string(minor);
}

int registersUsed(const std::string& infoLog, const std::string& kernel) {
    // "Compiling entry function 'saxpy' for 'sm_89'" ... "Used 10 registers"
    std::smatch m;
    std::regex pattern("entry function '" + kernel + "'[\\s\\S]*?Used (\\d+) registers");
    return std::regex_search(infoLog, m, pattern) ? std::stoi(m[1]) : -1;
}
