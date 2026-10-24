#include "sema.h"

#include <map>
#include <unordered_map>

// Type rules (deliberately strict, like Rust/Go):
// - no implicit conversions: 1 + 2.0 is an error, write f32(1) + 2.0
// - arithmetic (+ - * /) needs two i32 or two f32
// - % & | ^ << >> need two i32
// - comparisons give bool; && || ! need bool; if/while/for conditions must be bool
// - only pointers can be indexed, and the index must be i32
// - calls must match the function's parameter types; a function with '-> T' must
//   end with 'return <value>;' on every path
// - 'break' / 'continue' only inside loops; 'shared' arrays only in kernels (max 48 KB)

namespace {

bool isNumeric(Type t) { return !t.ptr && (t.base == Base::I32 || t.base == Base::F32); }
bool isI32(Type t) { return t == scalar(Base::I32); }
bool isBool(Type t) { return t == scalar(Base::Bool); }

const std::map<std::string, BuiltinFn> kBuiltins = {
    {"sqrt", BuiltinFn::Sqrt}, {"abs", BuiltinFn::Abs},
    {"min", BuiltinFn::Min},   {"max", BuiltinFn::Max},
    {"atomic_add", BuiltinFn::AtomicAdd}, {"barrier", BuiltinFn::Barrier},
};

constexpr int kMaxSharedBytes = 48 * 1024;  // what every CUDA GPU allows per block by default

// Does this statement list end in a return on every path? (Simple check: we
// only look at the last statement, following if/else and { } blocks.)
bool alwaysReturns(const std::vector<StmtPtr>& stmts) {
    if (stmts.empty()) return false;
    const Stmt& last = *stmts.back();
    switch (last.kind) {
        case StmtKind::Return: return true;
        case StmtKind::Block: return alwaysReturns(last.body);
        case StmtKind::If: return !last.elseBody.empty() && alwaysReturns(last.body) && alwaysReturns(last.elseBody);
        default: return false;
    }
}

class Sema {
public:
    void program(Program& prog) {
        // First collect every function, so a function can be called before it is defined.
        for (auto& f : prog.functions) {
            if (kBuiltins.count(f->name))
                throw CompileError(f->loc, "'" + f->name + "' is a built-in function and cannot be redefined");
            if (!functions_.emplace(f->name, f.get()).second)
                throw CompileError(f->loc, "'" + f->name + "' is defined twice");
        }
        for (auto& f : prog.functions) function(*f);
    }

private:
    std::map<std::string, Function*> functions_;
    // Innermost scope is at the back. A name is looked up from inside out.
    std::vector<std::unordered_map<std::string, Var*>> scopes_;
    Function* fn_ = nullptr;  // function being checked
    int loopDepth_ = 0;
    int sharedBytes_ = 0;

    void function(Function& f) {
        fn_ = &f;
        loopDepth_ = 0;
        sharedBytes_ = 0;
        scopes_.clear();
        scopes_.emplace_back();
        for (Var* p : f.params) {
            if (f.isKernel && p->type == scalar(Base::Bool))
                throw CompileError(p->loc, "kernel parameters must be i32, f32 or a pointer");
            declare(p);
        }
        stmts(f.body);  // same scope as the parameters, so 'let n' cannot hide parameter 'n'
        if (f.retType.base != Base::Void && !alwaysReturns(f.body))
            throw CompileError(f.loc, "function '" + f.name + "' must end with 'return <value>;' on every path");
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

    void loopBody(std::vector<StmtPtr>& body) {
        loopDepth_++;
        scoped(body);
        loopDepth_--;
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
            case StmtKind::Shared:
                if (!fn_->isKernel)
                    throw CompileError(s.loc, "shared arrays can only be declared in kernels");
                if (s.var->arraySize <= 0) throw CompileError(s.var->loc, "a shared array needs at least 1 element");
                sharedBytes_ += 4 * s.var->arraySize;
                if (sharedBytes_ > kMaxSharedBytes)
                    throw CompileError(s.var->loc, "kernel '" + fn_->name + "' uses more than 48 KB of shared memory");
                declare(s.var);
                break;
            case StmtKind::Assign: assign(s); break;
            case StmtKind::ExprStmt:
                if (s.value->kind != ExprKind::Call)
                    throw CompileError(s.loc, "this expression does nothing; only function calls can be used as statements");
                expr(*s.value, /*allowVoid=*/true);
                break;
            case StmtKind::If:
                condition(*s.cond, "if");
                scoped(s.body);
                scoped(s.elseBody);
                break;
            case StmtKind::While:
                condition(*s.cond, "while");
                loopBody(s.body);
                break;
            case StmtKind::For:
                scopes_.emplace_back();  // the loop variable lives only inside the loop
                if (s.init) stmt(*s.init);
                condition(*s.cond, "for");
                if (s.step) stmt(*s.step);
                loopBody(s.body);
                scopes_.pop_back();
                break;
            case StmtKind::Break:
            case StmtKind::Continue:
                if (loopDepth_ == 0)
                    throw CompileError(s.loc, std::string("'") + (s.kind == StmtKind::Break ? "break" : "continue") +
                                                  "' can only be used inside a loop");
                break;
            case StmtKind::Return: returnStmt(s); break;
            case StmtKind::Block: scoped(s.body); break;
        }
    }

    void returnStmt(Stmt& s) {
        Type want = fn_->retType;
        if (want.base == Base::Void) {
            if (s.value)
                throw CompileError(s.value->loc, std::string(fn_->isKernel ? "kernels" : "function '" + fn_->name + "'") +
                                                     " cannot return a value");
            return;
        }
        if (!s.value)
            throw CompileError(s.loc, "function '" + fn_->name + "' must return a value of type " + typeName(want));
        expr(*s.value);
        if (s.value->type != want)
            throw CompileError(s.value->loc, "function '" + fn_->name + "' returns " + typeName(want) +
                                                 ", but this value has type " + typeName(s.value->type));
    }

    void assign(Stmt& s) {
        Expr& target = *s.target;
        if (target.kind == ExprKind::Builtin)
            throw CompileError(target.loc, "cannot assign to " + target.name + " (it is read-only)");
        if (target.kind != ExprKind::VarRef && target.kind != ExprKind::Index)
            throw CompileError(target.loc, "the left side of an assignment must be a variable or an element like p[i]");
        expr(target);
        if (target.kind == ExprKind::VarRef && target.var->arraySize > 0)
            throw CompileError(target.loc, "cannot assign to shared array '" + target.name + "' (assign to its elements instead)");
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

    void expr(Expr& e, bool allowVoid = false) {
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
            case ExprKind::Call:
                call(e);
                if (!allowVoid && e.type.base == Base::Void)
                    throw CompileError(e.loc, "'" + e.name + "' does not return a value");
                break;
        }
    }

    void call(Expr& e) {
        for (auto& arg : e.kids) expr(*arg);

        auto builtin = kBuiltins.find(e.name);
        if (builtin != kBuiltins.end()) {
            e.builtin = builtin->second;
            e.type = builtinType(e);
            return;
        }

        auto found = functions_.find(e.name);
        if (found == functions_.end()) throw CompileError(e.loc, "unknown function '" + e.name + "'");
        Function* callee = found->second;
        if (callee->isKernel)
            throw CompileError(e.loc, "'" + e.name + "' is a kernel; kernels are launched from the CPU, not called");
        if (e.kids.size() != callee->params.size())
            throw CompileError(e.loc, "'" + e.name + "' takes " + std::to_string(callee->params.size()) +
                                          " argument(s), but " + std::to_string(e.kids.size()) + " were given");
        for (size_t i = 0; i < e.kids.size(); ++i) {
            if (e.kids[i]->type != callee->params[i]->type)
                throw CompileError(e.kids[i]->loc, "argument " + std::to_string(i + 1) + " of '" + e.name +
                                                       "' must be " + typeName(callee->params[i]->type) + ", not " +
                                                       typeName(e.kids[i]->type));
        }
        e.callee = callee;
        e.type = callee->retType;
    }

    Type builtinType(const Expr& e) {
        auto args = [&](size_t n) {
            if (e.kids.size() != n)
                throw CompileError(e.loc, "'" + e.name + "' takes " + std::to_string(n) + " argument(s), but " +
                                              std::to_string(e.kids.size()) + " were given");
        };
        switch (e.builtin) {
            case BuiltinFn::Sqrt:
                args(1);
                if (e.kids[0]->type != scalar(Base::F32))
                    throw CompileError(e.loc, "sqrt needs an f32, not " + typeName(e.kids[0]->type));
                return scalar(Base::F32);
            case BuiltinFn::Abs:
                args(1);
                if (!isNumeric(e.kids[0]->type))
                    throw CompileError(e.loc, "abs needs an i32 or f32, not " + typeName(e.kids[0]->type));
                return e.kids[0]->type;
            case BuiltinFn::Min:
            case BuiltinFn::Max:
                args(2);
                if (!isNumeric(e.kids[0]->type) || e.kids[0]->type != e.kids[1]->type)
                    throw CompileError(e.loc, "'" + e.name + "' needs two i32 or two f32 values, but got " +
                                                  typeName(e.kids[0]->type) + " and " + typeName(e.kids[1]->type));
                return e.kids[0]->type;
            case BuiltinFn::Barrier:
                args(0);
                return scalar(Base::Void);
            case BuiltinFn::AtomicAdd:
                args(2);
                if (e.kids[0]->kind != ExprKind::Index)
                    throw CompileError(e.kids[0]->loc, "the first argument of atomic_add must be an array element, "
                                                       "like atomic_add(p[i], 1)");
                if (e.kids[1]->type != e.kids[0]->type)
                    throw CompileError(e.kids[1]->loc, "atomic_add on an " + typeName(e.kids[0]->type) +
                                                           " element needs an " + typeName(e.kids[0]->type) + " value");
                return e.kids[0]->type;  // the old value, before the add
            case BuiltinFn::None: break;
        }
        return scalar(Base::Void);
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
