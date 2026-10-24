#pragma once

// The AST (abstract syntax tree): the program as a tree of statements and
// expressions. The parser builds it; sema fills in types and resolves names.

#include <memory>
#include <string>
#include <vector>

#include "diag.h"

// ---- Types ----------------------------------------------------------------

enum class Base { Void, Bool, I32, F32 };

// A scalar (i32, f32, bool) or a pointer to i32/f32 in GPU global memory.
struct Type {
    Base base = Base::Void;
    bool ptr = false;

    bool operator==(const Type& o) const { return base == o.base && ptr == o.ptr; }
    bool operator!=(const Type& o) const { return !(*this == o); }
};

inline Type scalar(Base b) { return Type{b, false}; }
std::string typeName(Type t);  // "i32", "*f32", ...

// ---- Expressions ----------------------------------------------------------

enum class Op {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor, Shl, Shr,
    Lt, Le, Gt, Ge, Eq, Ne,
    LogAnd, LogOr,
    Neg, Not,  // unary
};
const char* opName(Op op);  // "+", "&&", ...

// A kernel parameter or a local variable declared with 'let'.
struct Var {
    std::string name;
    Type type;
    Loc loc;
};

enum class ExprKind {
    IntLit,    // 42
    FloatLit,  // 4.2
    BoolLit,   // true
    VarRef,    // x
    Builtin,   // threadIdx.x, blockIdx.x, blockDim.x, gridDim.x (and .y/.z)
    Index,     // p[i]          kids = {p, i}
    Unary,     // -x, !b        kids = {x}
    Binary,    // a + b         kids = {a, b}
    Cast,      // f32(i)        kids = {i}
};

struct Expr {
    ExprKind kind;
    Loc loc;
    Type type;  // filled in by sema

    long long intVal = 0;
    double floatVal = 0;
    bool boolVal = false;
    std::string name;    // VarRef: variable name; Builtin: e.g. "threadIdx.x"
    Var* var = nullptr;  // VarRef: the variable it refers to (set by sema)
    Op op = Op::Add;     // Unary / Binary
    Type castTo;         // Cast
    std::vector<std::unique_ptr<Expr>> kids;
};
using ExprPtr = std::unique_ptr<Expr>;

// ---- Statements -----------------------------------------------------------

enum class StmtKind {
    Let,     // let x: T = value;
    Assign,  // target = value;   target += value;
    If,      // if (cond) { body } else { elseBody }
    While,   // while (cond) { body }
    For,     // for (init; cond; step) { body }
    Return,  // return;
    Block,   // { body }
};

struct Stmt;
using StmtPtr = std::unique_ptr<Stmt>;

struct Stmt {
    StmtKind kind;
    Loc loc;

    Var* var = nullptr;  // Let: the new variable
    Type declType;       // Let: written type, or Void if it should be inferred
    ExprPtr target;      // Assign: a VarRef or an Index
    bool compound = false;
    Op compoundOp = Op::Add;  // Assign: '+=' is compound with Op::Add
    ExprPtr value;            // Let / Assign
    ExprPtr cond;             // If / While / For
    StmtPtr init, step;       // For (both optional)
    std::vector<StmtPtr> body;
    std::vector<StmtPtr> elseBody;
};

// ---- Top level ------------------------------------------------------------

struct Function {
    std::string name;
    Loc loc;
    std::vector<Var*> params;
    std::vector<StmtPtr> body;
    std::vector<std::unique_ptr<Var>> vars;  // owns every Var (params and locals)
};

struct Program {
    std::vector<std::unique_ptr<Function>> kernels;
};
