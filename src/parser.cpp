#include "parser.h"

// Grammar (informal):
//   program    = (kernel | function)*
//   kernel     = "kernel" NAME "(" [param ("," param)*] ")" block
//   function   = ["export"] "fn" NAME "(" [param ("," param)*] ")" ["->" type] block
//              | "extern" "fn" NAME "(" [param ("," param)*] ")" ["->" type] ";"
//   param      = NAME ":" type
//   type       = "i32" | "f32" | "bool" | "*" type
//   block      = "{" statement* "}"
//   statement  = let ";" | shared ";" | if | while | for | block
//              | "break" ";" | "continue" ";" | "return" [expr] ";"
//              | assignment ";" | call ";"
//   let        = "let" NAME [":" type] "=" expr
//   shared     = "shared" NAME ":" ("i32" | "f32") "[" INT "]"
//   assignment = expr ("=" | "+=" | "-=" | "*=" | "/=") expr
//   expr       = binary operators with C precedence, then unary - and !,
//                then postfix p[i], then literals / names / calls f(a, b) / (expr) / i32(e) / f32(e)

namespace {

// Precedence of binary operators (higher binds tighter). Same order as C.
bool binaryOp(Tok t, int& prec, Op& op) {
    switch (t) {
        case Tok::PipePipe: prec = 1; op = Op::LogOr; return true;
        case Tok::AmpAmp: prec = 2; op = Op::LogAnd; return true;
        case Tok::Pipe: prec = 3; op = Op::BitOr; return true;
        case Tok::Caret: prec = 4; op = Op::BitXor; return true;
        case Tok::Amp: prec = 5; op = Op::BitAnd; return true;
        case Tok::EqEq: prec = 6; op = Op::Eq; return true;
        case Tok::NotEq: prec = 6; op = Op::Ne; return true;
        case Tok::Lt: prec = 7; op = Op::Lt; return true;
        case Tok::Le: prec = 7; op = Op::Le; return true;
        case Tok::Gt: prec = 7; op = Op::Gt; return true;
        case Tok::Ge: prec = 7; op = Op::Ge; return true;
        case Tok::Shl: prec = 8; op = Op::Shl; return true;
        case Tok::Shr: prec = 8; op = Op::Shr; return true;
        case Tok::Plus: prec = 9; op = Op::Add; return true;
        case Tok::Minus: prec = 9; op = Op::Sub; return true;
        case Tok::Star: prec = 10; op = Op::Mul; return true;
        case Tok::Slash: prec = 10; op = Op::Div; return true;
        case Tok::Percent: prec = 10; op = Op::Rem; return true;
        default: return false;
    }
}

bool isBuiltinGroup(const std::string& name) {
    return name == "threadIdx" || name == "blockIdx" || name == "blockDim" || name == "gridDim";
}

ExprPtr makeExpr(ExprKind kind, Loc loc) {
    auto e = std::make_unique<Expr>();
    e->kind = kind;
    e->loc = loc;
    return e;
}

StmtPtr makeStmt(StmtKind kind, Loc loc) {
    auto s = std::make_unique<Stmt>();
    s->kind = kind;
    s->loc = loc;
    return s;
}

class Parser {
public:
    explicit Parser(const std::vector<Token>& toks) : toks_(toks) {}

    Program program() {
        Program prog;
        while (!check(Tok::End)) prog.functions.push_back(function());
        return prog;
    }

private:
    const std::vector<Token>& toks_;
    size_t pos_ = 0;
    Function* fn_ = nullptr;  // the kernel being parsed (it owns the Vars)

    // ---- token helpers (the token list always ends with Tok::End) ----
    const Token& peek() const { return toks_[pos_]; }
    bool check(Tok t) const { return peek().kind == t; }

    bool accept(Tok t) {
        if (!check(t)) return false;
        pos_++;
        return true;
    }

    const Token& expect(Tok t) {
        if (!check(t)) error(std::string("expected ") + tokName(t) + " but found " + describe(peek()));
        return toks_[pos_++];
    }

    static std::string describe(const Token& t) {
        return t.kind == Tok::End ? "end of file" : "'" + t.text + "'";
    }

    [[noreturn]] void error(const std::string& msg) const { throw CompileError(peek().loc, msg); }

    Var* newVar(const std::string& name, Type type, Loc loc) {
        fn_->vars.push_back(std::make_unique<Var>(Var{name, type, loc}));
        return fn_->vars.back().get();
    }

    // ---- declarations ----
    std::unique_ptr<Function> function() {
        auto f = std::make_unique<Function>();
        fn_ = f.get();
        f->loc = peek().loc;
        // "export fn": other files may call it. "extern fn ...;": it is defined in another file.
        f->isExport = accept(Tok::KwExport);
        f->isExtern = !f->isExport && accept(Tok::KwExtern);
        if (accept(Tok::KwFn))
            f->isKernel = false;
        else if (f->isExport || f->isExtern)
            error("only device functions can be export or extern ('" + std::string(f->isExport ? "export" : "extern") +
                  " fn ...'); kernels are always visible");
        else if (!accept(Tok::KwKernel))
            error("expected 'kernel' or 'fn' but found " + describe(peek()));
        f->name = expect(Tok::Ident).text;
        expect(Tok::LParen);
        if (!check(Tok::RParen)) {
            do {
                const Token& name = expect(Tok::Ident);
                expect(Tok::Colon);
                Type t = type();
                f->params.push_back(newVar(name.text, t, name.loc));
            } while (accept(Tok::Comma));
        }
        expect(Tok::RParen);
        if (accept(Tok::Arrow)) {
            if (f->isKernel) error("kernels cannot return a value (they run once per thread, launched from the CPU)");
            f->retType = type();
        }
        if (f->isExtern) {
            if (check(Tok::LBrace)) error("an extern fn is defined in another file, so it has no body here: end it with ';'");
            expect(Tok::Semi);
        } else {
            f->body = block();
        }
        return f;
    }

    Type type() {
        Loc loc = peek().loc;
        if (accept(Tok::Star)) {
            Type elem = type();
            if (elem.ptr || elem.base == Base::Bool)
                throw CompileError(loc, "only pointers to i32 or f32 are supported (*i32, *f32)");
            elem.ptr = true;
            return elem;
        }
        if (accept(Tok::KwI32)) return scalar(Base::I32);
        if (accept(Tok::KwF32)) return scalar(Base::F32);
        if (accept(Tok::KwBool)) return scalar(Base::Bool);
        error("expected a type (i32, f32, bool, *i32 or *f32) but found " + describe(peek()));
    }

    // ---- statements ----
    std::vector<StmtPtr> block() {
        expect(Tok::LBrace);
        std::vector<StmtPtr> stmts;
        while (!check(Tok::RBrace) && !check(Tok::End)) stmts.push_back(statement());
        expect(Tok::RBrace);
        return stmts;
    }

    StmtPtr statement() {
        Loc loc = peek().loc;
        switch (peek().kind) {
            case Tok::KwLet: {
                StmtPtr s = let();
                expect(Tok::Semi);
                return s;
            }
            case Tok::KwIf: return ifStmt();
            case Tok::KwWhile: {
                pos_++;
                StmtPtr s = makeStmt(StmtKind::While, loc);
                expect(Tok::LParen);
                s->cond = expr();
                expect(Tok::RParen);
                s->body = block();
                return s;
            }
            case Tok::KwFor: return forStmt();
            case Tok::KwShared: {
                StmtPtr s = shared();
                expect(Tok::Semi);
                return s;
            }
            case Tok::KwBreak:
            case Tok::KwContinue: {
                StmtKind kind = check(Tok::KwBreak) ? StmtKind::Break : StmtKind::Continue;
                pos_++;
                expect(Tok::Semi);
                return makeStmt(kind, loc);
            }
            case Tok::KwReturn: {
                pos_++;
                StmtPtr s = makeStmt(StmtKind::Return, loc);
                if (!check(Tok::Semi)) s->value = expr();
                expect(Tok::Semi);
                return s;
            }
            case Tok::LBrace: {
                StmtPtr s = makeStmt(StmtKind::Block, loc);
                s->body = block();
                return s;
            }
            default: {
                StmtPtr s = simpleStatement();
                expect(Tok::Semi);
                return s;
            }
        }
    }

    StmtPtr shared() {
        StmtPtr s = makeStmt(StmtKind::Shared, expect(Tok::KwShared).loc);
        const Token& name = expect(Tok::Ident);
        expect(Tok::Colon);
        Type elem;
        if (accept(Tok::KwI32))
            elem = scalar(Base::I32);
        else if (accept(Tok::KwF32))
            elem = scalar(Base::F32);
        else
            error("shared arrays hold i32 or f32 values, e.g. 'shared buf: f32[256];'");
        expect(Tok::LBracket);
        const Token& size = expect(Tok::IntLit);
        expect(Tok::RBracket);
        if (size.text.size() > 6) throw CompileError(size.loc, "shared array is too large");
        elem.ptr = true;
        elem.shared = true;
        s->var = newVar(name.text, elem, name.loc);
        s->var->arraySize = std::stoi(size.text);
        return s;
    }

    StmtPtr let() {
        StmtPtr s = makeStmt(StmtKind::Let, expect(Tok::KwLet).loc);
        const Token& name = expect(Tok::Ident);
        if (accept(Tok::Colon)) s->declType = type();
        if (!check(Tok::Assign)) error("expected '=' after the variable name (every variable needs a starting value)");
        pos_++;
        s->var = newVar(name.text, s->declType, name.loc);
        s->value = expr();
        return s;
    }

    StmtPtr ifStmt() {
        StmtPtr s = makeStmt(StmtKind::If, expect(Tok::KwIf).loc);
        expect(Tok::LParen);
        s->cond = expr();
        expect(Tok::RParen);
        s->body = block();
        if (accept(Tok::KwElse)) {
            if (check(Tok::KwIf))
                s->elseBody.push_back(ifStmt());  // "else if" is an if inside the else
            else
                s->elseBody = block();
        }
        return s;
    }

    StmtPtr forStmt() {
        StmtPtr s = makeStmt(StmtKind::For, expect(Tok::KwFor).loc);
        expect(Tok::LParen);
        if (!check(Tok::Semi)) s->init = check(Tok::KwLet) ? let() : assignment();
        expect(Tok::Semi);
        s->cond = expr();
        expect(Tok::Semi);
        if (!check(Tok::RParen)) s->step = assignment();
        expect(Tok::RParen);
        s->body = block();
        return s;
    }

    // An assignment, or a call used as a statement (like barrier();).
    StmtPtr simpleStatement() {
        Loc loc = peek().loc;
        ExprPtr e = expr();
        if (check(Tok::Semi)) {
            StmtPtr s = makeStmt(StmtKind::ExprStmt, loc);
            s->value = std::move(e);
            return s;
        }
        return assignmentRest(std::move(e), loc);
    }

    StmtPtr assignment() {
        Loc loc = peek().loc;
        ExprPtr target = expr();
        return assignmentRest(std::move(target), loc);
    }

    StmtPtr assignmentRest(ExprPtr target, Loc loc) {
        StmtPtr s = makeStmt(StmtKind::Assign, loc);
        switch (peek().kind) {
            case Tok::Assign: break;
            case Tok::PlusAssign: s->compound = true; s->compoundOp = Op::Add; break;
            case Tok::MinusAssign: s->compound = true; s->compoundOp = Op::Sub; break;
            case Tok::StarAssign: s->compound = true; s->compoundOp = Op::Mul; break;
            case Tok::SlashAssign: s->compound = true; s->compoundOp = Op::Div; break;
            default: error("expected an assignment like 'x = ...;' but found " + describe(peek()));
        }
        pos_++;
        s->target = std::move(target);
        s->value = expr();
        return s;
    }

    // ---- expressions ----
    ExprPtr expr() { return binary(1); }

    // Precedence climbing: parse operators with precedence >= minPrec.
    ExprPtr binary(int minPrec) {
        ExprPtr lhs = unary();
        while (true) {
            int prec;
            Op op;
            if (!binaryOp(peek().kind, prec, op) || prec < minPrec) return lhs;
            Loc loc = peek().loc;
            pos_++;
            ExprPtr rhs = binary(prec + 1);  // +1 makes operators left-associative
            ExprPtr e = makeExpr(ExprKind::Binary, loc);
            e->op = op;
            e->kids.push_back(std::move(lhs));
            e->kids.push_back(std::move(rhs));
            lhs = std::move(e);
        }
    }

    ExprPtr unary() {
        Loc loc = peek().loc;
        if (check(Tok::Minus) || check(Tok::Bang)) {
            Op op = check(Tok::Minus) ? Op::Neg : Op::Not;
            pos_++;
            ExprPtr e = makeExpr(ExprKind::Unary, loc);
            e->op = op;
            e->kids.push_back(unary());
            return e;
        }
        return postfix();
    }

    ExprPtr postfix() {
        ExprPtr e = primary();
        while (check(Tok::LBracket)) {
            Loc loc = peek().loc;
            pos_++;
            ExprPtr index = makeExpr(ExprKind::Index, loc);
            index->kids.push_back(std::move(e));
            index->kids.push_back(expr());
            expect(Tok::RBracket);
            e = std::move(index);
        }
        return e;
    }

    ExprPtr primary() {
        const Token& t = peek();
        switch (t.kind) {
            case Tok::IntLit: {
                pos_++;
                if (t.text.size() > 10 || std::stoll(t.text) > 2147483647)
                    throw CompileError(t.loc, "integer literal is too large for i32");
                ExprPtr e = makeExpr(ExprKind::IntLit, t.loc);
                e->intVal = std::stoll(t.text);
                return e;
            }
            case Tok::FloatLit: {
                pos_++;
                ExprPtr e = makeExpr(ExprKind::FloatLit, t.loc);
                e->floatVal = std::stod(t.text);
                return e;
            }
            case Tok::KwTrue:
            case Tok::KwFalse: {
                pos_++;
                ExprPtr e = makeExpr(ExprKind::BoolLit, t.loc);
                e->boolVal = t.kind == Tok::KwTrue;
                return e;
            }
            case Tok::LParen: {
                pos_++;
                ExprPtr e = expr();
                expect(Tok::RParen);
                return e;
            }
            case Tok::KwI32:
            case Tok::KwF32: {  // conversion: i32(x) or f32(x)
                pos_++;
                ExprPtr e = makeExpr(ExprKind::Cast, t.loc);
                e->castTo = scalar(t.kind == Tok::KwI32 ? Base::I32 : Base::F32);
                expect(Tok::LParen);
                e->kids.push_back(expr());
                expect(Tok::RParen);
                return e;
            }
            case Tok::Ident: {
                pos_++;
                if (isBuiltinGroup(t.text)) {
                    expect(Tok::Dot);
                    const Token& c = expect(Tok::Ident);
                    if (c.text != "x" && c.text != "y" && c.text != "z")
                        throw CompileError(c.loc, t.text + " only has .x, .y and .z");
                    ExprPtr e = makeExpr(ExprKind::Builtin, t.loc);
                    e->name = t.text + "." + c.text;
                    return e;
                }
                if (accept(Tok::LParen)) {  // a call: name(args)
                    ExprPtr call = makeExpr(ExprKind::Call, t.loc);
                    call->name = t.text;
                    if (!check(Tok::RParen)) {
                        do {
                            call->kids.push_back(expr());
                        } while (accept(Tok::Comma));
                    }
                    expect(Tok::RParen);
                    return call;
                }
                ExprPtr e = makeExpr(ExprKind::VarRef, t.loc);
                e->name = t.text;
                return e;
            }
            default: error("expected an expression but found " + describe(t));
        }
    }
};

}  // namespace

Program parse(const std::vector<Token>& tokens) { return Parser(tokens).program(); }
