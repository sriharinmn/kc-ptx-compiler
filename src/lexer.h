#pragma once

// Stage 1: the lexer turns source text into a list of tokens.
//   "let i = 0;"  ->  [let] [ident i] [=] [int 0] [;]

#include <string>
#include <vector>

#include "diag.h"

enum class Tok {
    End, Ident, IntLit, FloatLit,
    // keywords
    KwKernel, KwLet, KwIf, KwElse, KwWhile, KwFor, KwReturn, KwTrue, KwFalse,
    KwI32, KwF32, KwBool,
    // punctuation and operators
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semi, Colon, Dot,
    Plus, Minus, Star, Slash, Percent,
    Amp, Pipe, Caret, Shl, Shr, Bang, AmpAmp, PipePipe,
    Lt, Le, Gt, Ge, EqEq, NotEq,
    Assign, PlusAssign, MinusAssign, StarAssign, SlashAssign,
};

struct Token {
    Tok kind;
    std::string text;  // exactly as written in the source
    Loc loc;
};

// Always ends with a Tok::End token. Throws CompileError on bad characters.
std::vector<Token> lex(const std::string& source);

// Human-readable token name for error messages, e.g. "';'".
const char* tokName(Tok t);
