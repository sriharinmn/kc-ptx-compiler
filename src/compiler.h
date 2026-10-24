#pragma once

// The whole compiler pipeline in one call:
//   source -> lex -> parse -> sema -> IR (SSA) -> PTX

#include <string>

struct CompileOptions {
    std::string arch = "sm_89";  // GPU architecture to target (RTX 40xx = sm_89)
};

struct CompileResult {
    std::string ir;   // printed SSA IR, for --emit-ir and debugging
    std::string ptx;  // the PTX assembly
};

// Throws CompileError (see diag.h) if the program is invalid.
CompileResult compile(const std::string& source, const CompileOptions& opts = {});
