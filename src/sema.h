#pragma once

// Stage 3: semantic analysis ("sema"), i.e. the type checker.
// - connects every variable use to its declaration (scopes)
// - gives every expression a type and rejects invalid programs
// After this stage, later stages can assume the program is valid.

#include "ast.h"

void checkProgram(Program& prog);
