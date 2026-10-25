#pragma once

// Compiling PTX with the PTX Compiler API (nvPTXCompiler): ptxas as a library,
// running inside our own process.
//
// There are two ways to turn PTX into GPU machine code:
//   - the driver's built-in ptxas (Module(ptx) in gpu.h): its version is tied to
//     the installed driver, so it only accepts PTX versions that driver knows
//   - this library, which ships with the CUDA toolkit
// The benchmark uses the library for both kc's and nvcc's PTX, so the same ptxas
// compiles both and the comparison is fair.

#include <string>
#include <vector>

struct CompiledPtx {
    std::vector<char> cubin;  // the machine code, as an ELF file (load it with Module(cubin))
    std::string infoLog;      // ptxas' report: registers used per function, etc.
};

// Throws std::runtime_error with ptxas' error log if the PTX is invalid.
CompiledPtx compilePtx(const std::string& ptx, const std::string& arch);

std::string ptxCompilerVersion();  // e.g. "13.3"

// Registers used by 'kernel', read from an info log ("Used N registers"); -1 if not found.
int registersUsed(const std::string& infoLog, const std::string& kernel);
