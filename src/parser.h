#pragma once

// Stage 2: the parser turns tokens into an AST (see ast.h).
// It is a hand-written recursive-descent parser: one function per grammar rule.

#include <vector>

#include "ast.h"
#include "lexer.h"

Program parse(const std::vector<Token>& tokens);
