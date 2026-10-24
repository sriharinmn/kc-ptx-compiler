#pragma once

// Stage 4: IR generation. Walks the (type-checked) AST and builds SSA IR.
// Uses the algorithm from Braun et al., "Simple and Efficient Construction of
// Static Single Assignment Form" (2013), which builds SSA in one pass directly
// from the AST: no dominator trees needed.

#include "ast.h"
#include "ir.h"

IrModule generateIr(const Program& prog);
