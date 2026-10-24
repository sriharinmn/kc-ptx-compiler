#include "ast.h"

#include <cstdio>

std::string typeName(Type t) {
    std::string base;
    switch (t.base) {
        case Base::Void: base = "void"; break;
        case Base::Bool: base = "bool"; break;
        case Base::I32: base = "i32"; break;
        case Base::F32: base = "f32"; break;
    }
    if (!t.ptr) return base;
    return t.shared ? "shared *" + base : "*" + base;
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

// ---- printing the tree ------------------------------------------------------

namespace {

// The AST is first turned into simple labelled nodes, then drawn.
struct Node {
    std::string label;
    std::vector<Node> kids;
};

std::string withType(const std::string& label, Type t) {
    return t.base == Base::Void ? label : label + "  : " + typeName(t);
}

Node exprNode(const Expr& e) {
    Node n;
    switch (e.kind) {
        case ExprKind::IntLit: n.label = std::to_string(e.intVal); break;
        case ExprKind::FloatLit: {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%g", e.floatVal);
            n.label = buf;
            if (n.label.find_first_of(".en") == std::string::npos) n.label += ".0";
            break;
        }
        case ExprKind::BoolLit: n.label = e.boolVal ? "true" : "false"; break;
        case ExprKind::VarRef:
        case ExprKind::Builtin: n.label = e.name; break;
        case ExprKind::Index: n.label = "index []"; break;
        case ExprKind::Unary:
        case ExprKind::Binary: n.label = opName(e.op); break;
        case ExprKind::Cast: n.label = "convert to " + typeName(e.castTo); break;
        case ExprKind::Call: n.label = "call " + e.name; break;
    }
    n.label = withType(n.label, e.type);
    for (const auto& k : e.kids) n.kids.push_back(exprNode(*k));
    return n;
}

Node stmtNode(const Stmt& s);

Node group(const std::string& label, const std::vector<StmtPtr>& stmts) {
    Node n{label, {}};
    for (const auto& s : stmts) n.kids.push_back(stmtNode(*s));
    return n;
}

Node stmtNode(const Stmt& s) {
    Node n;
    switch (s.kind) {
        case StmtKind::Let:
            n.label = withType("let " + s.var->name, s.var->type);
            n.kids.push_back(exprNode(*s.value));
            break;
        case StmtKind::Shared:
            n.label = "shared " + s.var->name + "  : " + typeName(scalar(s.var->type.base)) + "[" +
                      std::to_string(s.var->arraySize) + "]";
            break;
        case StmtKind::Assign:
            n.label = s.compound ? std::string("assign ") + opName(s.compoundOp) + "=" : "assign =";
            n.kids.push_back(exprNode(*s.target));
            n.kids.push_back(exprNode(*s.value));
            break;
        case StmtKind::ExprStmt: n = exprNode(*s.value); break;
        case StmtKind::Break: n.label = "break"; break;
        case StmtKind::Continue: n.label = "continue"; break;
        case StmtKind::If:
            n.label = "if";
            n.kids.push_back({"cond", {exprNode(*s.cond)}});
            n.kids.push_back(group("then", s.body));
            if (!s.elseBody.empty()) n.kids.push_back(group("else", s.elseBody));
            break;
        case StmtKind::While:
            n.label = "while";
            n.kids.push_back({"cond", {exprNode(*s.cond)}});
            n.kids.push_back(group("body", s.body));
            break;
        case StmtKind::For:
            n.label = "for";
            if (s.init) n.kids.push_back({"init", {stmtNode(*s.init)}});
            n.kids.push_back({"cond", {exprNode(*s.cond)}});
            if (s.step) n.kids.push_back({"step", {stmtNode(*s.step)}});
            n.kids.push_back(group("body", s.body));
            break;
        case StmtKind::Return:
            n.label = "return";
            if (s.value) n.kids.push_back(exprNode(*s.value));
            break;
        case StmtKind::Block: n = group("block", s.body); break;
    }
    return n;
}

void render(const Node& n, const std::string& prefix, bool last, bool root, std::string& out) {
    out += root ? "" : prefix + (last ? "└─ " : "├─ ");
    out += n.label + "\n";
    std::string childPrefix = root ? "" : prefix + (last ? "   " : "│  ");
    for (size_t i = 0; i < n.kids.size(); ++i) render(n.kids[i], childPrefix, i + 1 == n.kids.size(), false, out);
}

}  // namespace

std::string printAst(const Program& prog) {
    std::string out;
    for (const auto& f : prog.functions) {
        Node n{(f->isKernel ? "kernel " : "fn ") + f->name, {}};
        if (!f->isKernel && f->retType.base != Base::Void) n.label += "  -> " + typeName(f->retType);
        for (const Var* p : f->params) n.kids.push_back({"param " + p->name + "  : " + typeName(p->type), {}});
        n.kids.push_back(group("body", f->body));
        render(n, "", true, true, out);
        out += "\n";
    }
    return out;
}
