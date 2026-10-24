#pragma once

#include <stdexcept>
#include <string>

// A position in the source file (1-based), used in error messages.
struct Loc {
    int line = 1;
    int col = 1;
};

// Every user-facing compile error is thrown as a CompileError.
// The message looks like "3:12: error: unknown variable 'x'".
struct CompileError : std::runtime_error {
    Loc loc;
    CompileError(Loc l, const std::string& msg)
        : std::runtime_error(std::to_string(l.line) + ":" + std::to_string(l.col) + ": error: " + msg),
          loc(l) {}
};
