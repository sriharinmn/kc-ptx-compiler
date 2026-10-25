#pragma once

// The optimizer: passes that rewrite the SSA IR into smaller IR that computes
// exactly the same results. Each pass is in its own file in src/opt/.
//
// -O0 runs nothing. -O1 runs:
//   1. inline     copy small device functions into their callers
//   then, for each function, repeated until nothing changes:
//   2. simplify   fold constants, simplify x+0 etc., remove branches on constants
//   3. cse        reuse a value that was already computed (common subexpressions)
//   4. licm       move loop-invariant work out of loops
//   5. dce        delete instructions whose result nobody uses
//   finally: delete device functions that are no longer called.
// The verifier (verify.h) checks the IR after every pass.

#include <string>

#include "ir.h"

// Returns a short report of what each pass did (for 'kc --passes' and the playground).
std::string optimize(IrModule& m, int level);

// The passes. Each returns how many changes it made (0 = nothing to do).
int inlineCalls(IrModule& m);
int simplify(IrFunction& f);
int eliminateCommonSubexpressions(IrFunction& f);
int hoistLoopInvariants(IrFunction& f);
int removeDeadCode(IrFunction& f);
int removeUncalledFunctions(IrModule& m);
