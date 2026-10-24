#pragma once

// The whole compiler pipeline in one call:
//   source -> lex -> parse -> sema -> IR (SSA) -> PTX

#include <string>

struct CompileOptions {
    std::string arch = "sm_89";  // GPU architecture to target (RTX 40xx = sm_89)
};

// The output of every stage, as text.
struct CompileResult {
    std::string tokens;  // the token list
    std::string ast;     // the syntax tree (with types once sema has run)
    std::string ir;      // the SSA IR
    std::string ptx;     // the PTX assembly
};

// Throws CompileError (see diag.h) if the program is invalid.
CompileResult compile(const std::string& source, const CompileOptions& opts = {});

// Like compile(), but never throws. The stages that ran are filled in, and the
// error (if any) is returned too. The web playground uses this to show partial results.
struct ExploreResult {
    CompileResult stages;
    std::string error;  // empty if compilation succeeded
    int errorLine = 0;
    int errorCol = 0;
};
ExploreResult explore(const std::string& source, const CompileOptions& opts = {});
