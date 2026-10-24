#include "sema.h"

#include <set>
#include <unordered_map>

// Type rules (deliberately strict, like Rust/Go):
// - no implicit conversions: 1 + 2.0 is an error, write f32(1) + 2.0
// - arithmetic (+ - * /) needs two i32 or two f32
// - % & | ^ << >> need two i32
// - comparisons give bool; && || ! need bool; if/while/for conditions must be bool
// - only pointers can be indexed, and the index must be i32

namespace {

bool isNumeric(Type t) { return !t.ptr && (t.base == Base::I32 || t.base == Base::F32); }
bool isI32(Type t) { return t == scalar(Base::I32); }
bool isBool(Type t) { return t == scalar(Base::Bool); }

class Sema {
public:
    void program(Program& prog) {
        std::set<std::string> names;
        for (auto& k : prog.kernels) {
            if (!names.insert(k->name).second)
                throw CompileError(k->loc, "kernel '" + k->name + "' is defined twice");
            kernel(*k);
        }
    }

private:
    // Innermost scope is at the back. A name is looked up from inside out.
    std::vector<std::unordered_map<std::string, Var*>> scopes_;

    void kernel(Function& f) {
        scopes_.clear();
        scopes_.emplace_back();
        for (Var* p : f.params) {
            if (p->type == scalar(Base::Bool))
                throw CompileError(p->loc, "kernel parameters must be i32, f32 or a pointer");
            declare(p);
        }
        stmts(f.body);  // same scope as the parameters, so 'let n' cannot hide parameter 'n'
    }

    void declare(Var* v) {
        auto& scope = scopes_.back();
        if (!scope.emplace(v->name, v).second)
            throw CompileError(v->loc, "'" + v->name + "' is already declared in this scope");
    }

    Var* lookup(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        return nullptr;
    }

    void scoped(std::vector<StmtPtr>& body) {
        scopes_.emplace_back();
        stmts(body);
        scopes_.pop_back();
    }

    void stmts(std::vector<StmtPtr>& list) {
        for (auto& s : list) stmt(*s);
    }

    void stmt(Stmt& s) {
        switch (s.kind) {
            case StmtKind::Let: {
                expr(*s.value);
                Type t = s.value->type;
                if (s.declType.base != Base::Void && s.declType != t)
                    throw CompileError(s.value->loc, "'" + s.var->name + "' is declared as " +
                                                         typeName(s.declType) + " but the value has type " +
                                                         typeName(t));
                s.var->type = t;
                declare(s.var);  // declared after the value, so 'let x = x + 1;' uses an outer x
                break;
            }
            case StmtKind::Assign: assign(s); break;
            case StmtKind::If:
                condition(*s.cond, "if");
                scoped(s.body);
                scoped(s.elseBody);
                break;
            case StmtKind::While:
                condition(*s.cond, "while");
                scoped(s.body);
                break;
            case StmtKind::For:
                scopes_.emplace_back();  // the loop variable lives only inside the loop
                if (s.init) stmt(*s.init);
                condition(*s.cond, "for");
                if (s.step) stmt(*s.step);
                scoped(s.body);
                scopes_.pop_back();
                break;
            case StmtKind::Return: break;
            case StmtKind::Block: scoped(s.body); break;
        }
    }

    void assign(Stmt& s) {
        Expr& target = *s.target;
        if (target.kind == ExprKind::Builtin)
            throw CompileError(target.loc, "cannot assign to " + target.name + " (it is read-only)");
        if (target.kind != ExprKind::VarRef && target.kind != ExprKind::Index)
            throw CompileError(target.loc, "the left side of an assignment must be a variable or an element like p[i]");
        expr(target);
        expr(*s.value);
        if (s.compound) {
            binaryType(s.compoundOp, target.type, s.value->type, s.loc);
        } else if (target.type != s.value->type) {
            throw CompileError(s.value->loc, "cannot assign a value of type " + typeName(s.value->type) +
                                                 " to something of type " + typeName(target.type));
        }
    }

    void condition(Expr& e, const char* what) {
        expr(e);
        if (!isBool(e.type))
            throw CompileError(e.loc, std::string(what) + " condition must be bool, but it has type " +
                                          typeName(e.type) + " (try a comparison like x != 0)");
    }

    void expr(Expr& e) {
        switch (e.kind) {
            case ExprKind::IntLit: e.type = scalar(Base::I32); break;
            case ExprKind::FloatLit: e.type = scalar(Base::F32); break;
            case ExprKind::BoolLit: e.type = scalar(Base::Bool); break;
            case ExprKind::Builtin: e.type = scalar(Base::I32); break;
            case ExprKind::VarRef:
                e.var = lookup(e.name);
                if (!e.var) throw CompileError(e.loc, "unknown variable '" + e.name + "'");
                e.type = e.var->type;
                break;
            case ExprKind::Index: {
                Expr& base = *e.kids[0];
                Expr& index = *e.kids[1];
                expr(base);
                expr(index);
                if (!base.type.ptr)
                    throw CompileError(e.loc, "cannot index a value of type " + typeName(base.type) +
                                                  " (only pointers can be indexed)");
                if (!isI32(index.type))
                    throw CompileError(index.loc, "an index must be i32, not " + typeName(index.type));
                e.type = scalar(base.type.base);
                break;
            }
            case ExprKind::Unary: {
                expr(*e.kids[0]);
                Type t = e.kids[0]->type;
                if (e.op == Op::Neg && !isNumeric(t))
                    throw CompileError(e.loc, "cannot negate a value of type " + typeName(t));
                if (e.op == Op::Not && !isBool(t))
                    throw CompileError(e.loc, "'!' needs a bool, but got " + typeName(t));
                e.type = t;
                break;
            }
            case ExprKind::Binary:
                expr(*e.kids[0]);
                expr(*e.kids[1]);
                e.type = binaryType(e.op, e.kids[0]->type, e.kids[1]->type, e.loc);
                break;
            case ExprKind::Cast:
                expr(*e.kids[0]);
                if (!isNumeric(e.kids[0]->type))
                    throw CompileError(e.loc, "can only convert between i32 and f32, not from " +
                                                  typeName(e.kids[0]->type));
                e.type = e.castTo;
                break;
        }
    }

    // Returns the result type of 'a op b', or throws if the operands are not allowed.
    Type binaryType(Op op, Type a, Type b, Loc loc) {
        std::string o = std::string("'") + opName(op) + "'";
        if (op == Op::LogAnd || op == Op::LogOr) {
            if (!isBool(a) || !isBool(b))
                throw CompileError(loc, o + " needs bool operands, but got " + typeName(a) + " and " + typeName(b));
            return a;
        }
        if (a != b)
            throw CompileError(loc, "operands of " + o + " must have the same type, but got " + typeName(a) +
                                        " and " + typeName(b) + " (use i32(...) or f32(...) to convert)");
        switch (op) {
            case Op::Add:
            case Op::Sub:
            case Op::Mul:
            case Op::Div:
                if (!isNumeric(a)) throw CompileError(loc, o + " needs i32 or f32 operands, not " + typeName(a));
                return a;
            case Op::Rem:
            case Op::BitAnd:
            case Op::BitOr:
            case Op::BitXor:
            case Op::Shl:
            case Op::Shr:
                if (!isI32(a)) throw CompileError(loc, o + " needs i32 operands, not " + typeName(a));
                return a;
            case Op::Lt:
            case Op::Le:
            case Op::Gt:
            case Op::Ge:
                if (!isNumeric(a)) throw CompileError(loc, o + " needs i32 or f32 operands, not " + typeName(a));
                return scalar(Base::Bool);
            case Op::Eq:
            case Op::Ne:
                if (a.ptr) throw CompileError(loc, "pointers cannot be compared");
                return scalar(Base::Bool);
            default: throw CompileError(loc, "internal error: unexpected operator " + o);
        }
    }
};

}  // namespace

void checkProgram(Program& prog) { Sema().program(prog); }
