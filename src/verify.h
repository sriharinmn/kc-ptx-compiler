#pragma once

// The IR verifier: checks that a function still follows the SSA rules.
// It runs after IR generation and after every optimization pass, so a buggy
// pass is caught right where it broke something, not later as wrong GPU output.
//
// Rules checked:
//   - every block ends with exactly one jump/return, and phis come first
//   - each block's predecessor list matches the jumps that really go there
//   - a phi has one operand per predecessor
//   - every operand is an instruction that is still in the function, and its
//     definition dominates the use (so the value is always computed first)
//   - operand counts and types fit the opcode

#include <stdexcept>
#include <string>

#include "ir.h"

// A bug in the compiler itself (not in the user's program).
struct InternalError : std::logic_error {
    using std::logic_error::logic_error;
};

// Throws InternalError naming the pass and the broken rule.
void verifyFunction(const IrFunction& f, const std::string& afterPass);
void verifyModule(const IrModule& m, const std::string& afterPass);
