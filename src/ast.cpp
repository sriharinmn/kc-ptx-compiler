#include "ast.h"

std::string typeName(Type t) {
    std::string base;
    switch (t.base) {
        case Base::Void: base = "void"; break;
        case Base::Bool: base = "bool"; break;
        case Base::I32: base = "i32"; break;
        case Base::F32: base = "f32"; break;
    }
    return t.ptr ? "*" + base : base;
}

const char* opName(Op op) {
    switch (op) {
        case Op::Add: return "+";
        case Op::Sub: return "-";
        case Op::Mul: return "*";
        case Op::Div: return "/";
        case Op::Rem: return "%";
        case Op::BitAnd: return "&";
        case Op::BitOr: return "|";
        case Op::BitXor: return "^";
        case Op::Shl: return "<<";
        case Op::Shr: return ">>";
        case Op::Lt: return "<";
        case Op::Le: return "<=";
        case Op::Gt: return ">";
        case Op::Ge: return ">=";
        case Op::Eq: return "==";
        case Op::Ne: return "!=";
        case Op::LogAnd: return "&&";
        case Op::LogOr: return "||";
        case Op::Neg: return "-";
        case Op::Not: return "!";
    }
    return "?";
}
