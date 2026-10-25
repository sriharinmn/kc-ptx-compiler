#pragma once

// The whole compiler pipeline in one call:
//   source -> lex -> parse -> sema -> IR (SSA) -> divergence warnings -> optimizer -> PTX

#include <string>

struct CompileOptions {
    std::string arch = "sm_89";  // GPU architecture to target (RTX 40xx = sm_89)
    int optLevel = 1;            // 0 = no optimization, 1 = run the optimizer (src/opt/)
};

// The output of every stage, as text.
struct CompileResult {
    std::string tokens;  // the token list
    std::string ast;     // the syntax tree (with types once sema has run)
    std::string ir;      // the SSA IR, as IR generation built it
    std::string optIr;   // the SSA IR after the optimizer (empty at -O0)
    std::string passes;  // what each optimizer pass changed (empty at -O0)
    std::string ptx;     // the PTX assembly
    std::string warnings;  // "L:C: warning: ..." lines (the program still compiles)
};

// Throws CompileError (see diag.h) if the program is invalid, and
// InternalError (verify.h) if the compiler itself has a bug.
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
