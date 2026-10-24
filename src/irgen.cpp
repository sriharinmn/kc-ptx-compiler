#include "irgen.h"

#include <algorithm>
#include <map>

// How SSA construction works here (Braun et al. 2013), in short:
//
// Variables in the source can be assigned many times, but SSA values cannot.
// So for each variable we remember "its current value in block B" (defs_).
//   - assignment  x = v   ->  writeVar(x, B, v)
//   - use of x            ->  readVar(x, B)
// If B has no value for x yet, we ask its predecessor blocks. If there are
// several predecessors, their answers may differ, so we create a phi that
// picks the right one. Phis that turn out to always pick the same value are
// "trivial" and are removed again.
//
// A loop header is not "sealed" until the loop body (the back edge) has been
// generated, because only then do we know all its predecessors. Reads in an
// unsealed block create placeholder phis that get filled in by seal().

namespace {

IrType irType(Type t) {
    if (t.ptr) return IrType::U64;
    switch (t.base) {
        case Base::Bool: return IrType::Pred;
        case Base::I32: return IrType::S32;
        case Base::F32: return IrType::F32;
        case Base::Void: return IrType::Void;
    }
    return IrType::Void;
}

Opc binaryOpc(Op op) {
    switch (op) {
        case Op::Add: return Opc::Add;
        case Op::Sub: return Opc::Sub;
        case Op::Mul: return Opc::Mul;
        case Op::Div: return Opc::Div;
        case Op::Rem: return Opc::Rem;
        case Op::BitAnd: return Opc::And;
        case Op::BitOr: return Opc::Or;
        case Op::BitXor: return Opc::Xor;
        case Op::Shl: return Opc::Shl;
        case Op::Shr: return Opc::Shr;
        case Op::Lt: return Opc::Lt;
        case Op::Le: return Opc::Le;
        case Op::Gt: return Opc::Gt;
        case Op::Ge: return Opc::Ge;
        case Op::Eq: return Opc::Eq;
        case Op::Ne: return Opc::Ne;
        default: return Opc::Add;  // && || - ! are handled separately
    }
}

// threadIdx.x -> tid.x, blockIdx.x -> ctaid.x, blockDim.x -> ntid.x, gridDim.x -> nctaid.x
std::string sregName(const std::string& builtin) {
    size_t dot = builtin.find('.');
    std::string group = builtin.substr(0, dot);
    std::string component = builtin.substr(dot);
    if (group == "threadIdx") return "tid" + component;
    if (group == "blockIdx") return "ctaid" + component;
    if (group == "blockDim") return "ntid" + component;
    return "nctaid" + component;
}

Inst* resolve(Inst* v) {
    while (v->replacement) v = v->replacement;
    return v;
}

class IrGen {
public:
    explicit IrGen(IrFunction& fn) : fn_(fn) {}

    void kernel(const Function& k) {
        fn_.name = k.name;
        entry_ = fn_.newBlock();
        entry_->sealed = true;  // nothing ever jumps to the entry block
        cur_ = entry_;
        for (size_t i = 0; i < k.params.size(); ++i) {
            const Var* p = k.params[i];
            Inst* in = emit(Opc::Param, irType(p->type));
            in->paramIndex = static_cast<int>(i);
            in->name = p->name;
            fn_.params.push_back(in);
            writeVar(p, entry_, in);
        }
        stmts(k.body);
        if (cur_) emit(Opc::Ret, IrType::Void);
        fn_.renumber();
    }

private:
    IrFunction& fn_;
    Block* entry_ = nullptr;
    Block* cur_ = nullptr;  // block we are adding code to; nullptr after a return (unreachable)
    std::map<const Var*, std::map<Block*, Inst*>> defs_;           // current value of a variable per block
    std::map<Block*, std::map<const Var*, Inst*>> incompletePhis_; // placeholder phis in unsealed blocks
    std::map<IrType, Inst*> undefs_;

    // ---- building instructions ----
    Inst* emit(Opc op, IrType type, std::vector<Inst*> ops = {}) {
        Inst* in = fn_.newInst(op, type);
        in->ops = std::move(ops);
        in->parent = cur_;
        cur_->insts.push_back(in);
        return in;
    }

    void branch(Block* target) {
        Inst* br = emit(Opc::Br, IrType::Void);
        br->targets = {target};
        target->preds.push_back(cur_);
        cur_ = nullptr;
    }

    void condBranch(Inst* cond, Block* ifTrue, Block* ifFalse) {
        Inst* br = emit(Opc::CondBr, IrType::Void, {cond});
        br->targets = {ifTrue, ifFalse};
        ifTrue->preds.push_back(cur_);
        ifFalse->preds.push_back(cur_);
        cur_ = nullptr;
    }

    // Continue generating code in block b. Moving it to the end keeps blocks in
    // source order, so most jumps go to the next block and become fall-throughs.
    void startBlock(Block* b) {
        fn_.moveToEnd(b);
        cur_ = b;
    }

    // ---- SSA construction ----
    void writeVar(const Var* v, Block* b, Inst* value) { defs_[v][b] = value; }

    Inst* readVar(const Var* v, Block* b) {
        auto& perBlock = defs_[v];
        auto it = perBlock.find(b);
        if (it != perBlock.end()) return resolve(it->second);
        return readVarRecursive(v, b);
    }

    Inst* readVarRecursive(const Var* v, Block* b) {
        Inst* value;
        if (!b->sealed) {
            // Not all predecessors are known yet: placeholder phi, completed in seal().
            value = newPhi(b, irType(v->type));
            incompletePhis_[b][v] = value;
        } else if (b->preds.size() == 1) {
            value = readVar(v, b->preds[0]);  // one way in: no phi needed
        } else {
            Inst* phi = newPhi(b, irType(v->type));
            writeVar(v, b, phi);  // written first so a loop leading back here finds it
            value = addPhiOperands(v, phi);
        }
        value = resolve(value);
        writeVar(v, b, value);
        return value;
    }

    Inst* newPhi(Block* b, IrType type) {
        Inst* phi = fn_.newInst(Opc::Phi, type);
        phi->parent = b;
        auto firstNonPhi = std::find_if(b->insts.begin(), b->insts.end(),
                                        [](Inst* in) { return in->op != Opc::Phi; });
        b->insts.insert(firstNonPhi, phi);
        return phi;
    }

    Inst* addPhiOperands(const Var* v, Inst* phi) {
        for (Block* pred : phi->parent->preds) phi->ops.push_back(readVar(v, pred));
        return tryRemoveTrivialPhi(phi);
    }

    // A phi like  x = phi(a, a)  or  x = phi(a, x)  always means 'a': replace it by 'a'.
    Inst* tryRemoveTrivialPhi(Inst* phi) {
        Inst* same = nullptr;
        for (Inst* op : phi->ops) {
            if (op == same || op == phi) continue;
            if (same) return phi;  // merges two different values: a real phi, keep it
            same = op;
        }
        if (!same) same = undef(phi->type);  // never assigned on any path

        // Phis that use this one may become trivial once it is gone.
        std::vector<Inst*> phiUsers;
        for (auto& b : fn_.blocks)
            for (Inst* in : b->insts)
                if (in != phi && in->op == Opc::Phi && std::count(in->ops.begin(), in->ops.end(), phi))
                    phiUsers.push_back(in);

        replaceAllUses(phi, same);
        auto& insts = phi->parent->insts;
        insts.erase(std::find(insts.begin(), insts.end(), phi));
        phi->replacement = same;

        for (Inst* user : phiUsers)
            if (!user->replacement) tryRemoveTrivialPhi(user);
        return resolve(same);
    }

    void replaceAllUses(Inst* from, Inst* to) {
        for (auto& b : fn_.blocks)
            for (Inst* in : b->insts)
                for (Inst*& op : in->ops)
                    if (op == from) op = to;
        for (auto& [var, perBlock] : defs_)
            for (auto& [block, value] : perBlock)
                if (value == from) value = to;
    }

    // Called once all predecessors of b are known.
    void seal(Block* b) {
        auto it = incompletePhis_.find(b);
        if (it != incompletePhis_.end()) {
            for (auto& [var, phi] : it->second) addPhiOperands(var, phi);
            incompletePhis_.erase(it);
        }
        b->sealed = true;
    }

    // Value of a variable that was never assigned on some path. Can only happen
    // in unreachable code paths, since 'let' always requires a starting value.
    Inst* undef(IrType type) {
        Inst*& u = undefs_[type];
        if (!u) {
            u = fn_.newInst(Opc::Const, type);
            u->parent = entry_;
            entry_->insts.insert(entry_->insts.begin(), u);
        }
        return u;
    }

    // ---- statements ----
    void stmts(const std::vector<StmtPtr>& list) {
        for (const auto& s : list) {
            if (!cur_) return;  // code after 'return' can never run: skip it
            stmt(*s);
        }
    }

    void stmt(const Stmt& s) {
        switch (s.kind) {
            case StmtKind::Let: writeVar(s.var, cur_, expr(*s.value)); break;
            case StmtKind::Assign: assign(s); break;
            case StmtKind::If: ifStmt(s); break;
            case StmtKind::While: loop(*s.cond, s.body, nullptr); break;
            case StmtKind::For:
                if (s.init) stmt(*s.init);
                loop(*s.cond, s.body, s.step.get());
                break;
            case StmtKind::Return:
                emit(Opc::Ret, IrType::Void);
                cur_ = nullptr;
                break;
            case StmtKind::Block: stmts(s.body); break;
        }
    }

    void assign(const Stmt& s) {
        const Expr& target = *s.target;
        IrType type = irType(target.type);
        if (target.kind == ExprKind::VarRef) {
            Inst* value = expr(*s.value);
            if (s.compound) value = emit(binaryOpc(s.compoundOp), type, {readVar(target.var, cur_), value});
            writeVar(target.var, cur_, value);
        } else {  // p[i] = value
            Inst* addr = elemAddr(target);
            Inst* value = expr(*s.value);
            if (s.compound) {
                Inst* old = emit(Opc::Load, type, {addr});
                old->memType = type;
                value = emit(binaryOpc(s.compoundOp), type, {old, value});
            }
            Inst* st = emit(Opc::Store, IrType::Void, {addr, value});
            st->memType = type;
        }
    }

    //   cond; condbr then, else   then: ...; br join   else: ...; br join   join: ...
    void ifStmt(const Stmt& s) {
        Inst* c = expr(*s.cond);
        Block* thenB = fn_.newBlock();
        Block* elseB = s.elseBody.empty() ? nullptr : fn_.newBlock();
        Block* join = fn_.newBlock();
        condBranch(c, thenB, elseB ? elseB : join);
        seal(thenB);
        if (elseB) seal(elseB);

        startBlock(thenB);
        stmts(s.body);
        if (cur_) branch(join);
        if (elseB) {
            startBlock(elseB);
            stmts(s.elseBody);
            if (cur_) branch(join);
        }

        if (join->preds.empty()) {
            fn_.removeBlock(join);  // both sides returned: nothing comes after the if
        } else {
            seal(join);
            startBlock(join);
        }
    }

    //   br header   header: cond; condbr body, exit   body: ...; step; br header   exit: ...
    void loop(const Expr& cond, const std::vector<StmtPtr>& body, const Stmt* step) {
        Block* header = fn_.newBlock();
        branch(header);
        startBlock(header);  // not sealed yet: the back edge from the body is still missing
        Inst* c = expr(cond);
        Block* bodyB = fn_.newBlock();
        Block* exit = fn_.newBlock();
        condBranch(c, bodyB, exit);
        seal(bodyB);

        startBlock(bodyB);
        stmts(body);
        if (cur_ && step) stmt(*step);
        if (cur_) branch(header);  // the back edge

        seal(header);  // now all predecessors of the header are known
        seal(exit);
        startBlock(exit);
    }

    // ---- expressions ----
    Inst* expr(const Expr& e) {
        switch (e.kind) {
            case ExprKind::IntLit: {
                Inst* c = emit(Opc::Const, IrType::S32);
                c->imm = e.intVal;
                return c;
            }
            case ExprKind::FloatLit: {
                Inst* c = emit(Opc::Const, IrType::F32);
                c->fimm = static_cast<float>(e.floatVal);
                return c;
            }
            case ExprKind::BoolLit: {
                Inst* c = emit(Opc::Const, IrType::Pred);
                c->imm = e.boolVal ? 1 : 0;
                return c;
            }
            case ExprKind::VarRef: return readVar(e.var, cur_);
            case ExprKind::Builtin: {
                Inst* in = emit(Opc::SReg, IrType::S32);
                in->name = sregName(e.name);
                return in;
            }
            case ExprKind::Index: {
                Inst* addr = elemAddr(e);
                Inst* ld = emit(Opc::Load, irType(e.type), {addr});
                ld->memType = ld->type;
                return ld;
            }
            case ExprKind::Unary: {
                Inst* v = expr(*e.kids[0]);
                return emit(e.op == Op::Neg ? Opc::Neg : Opc::Not, irType(e.type), {v});
            }
            case ExprKind::Binary: {
                if (e.op == Op::LogAnd || e.op == Op::LogOr) return shortCircuit(e);
                Inst* a = expr(*e.kids[0]);
                Inst* b = expr(*e.kids[1]);
                return emit(binaryOpc(e.op), irType(e.type), {a, b});
            }
            case ExprKind::Cast: {
                Inst* v = expr(*e.kids[0]);
                if (e.kids[0]->type == e.castTo) return v;
                return emit(e.castTo.base == Base::F32 ? Opc::CvtI2F : Opc::CvtF2I, irType(e.castTo), {v});
            }
        }
        return nullptr;
    }

    Inst* elemAddr(const Expr& index) {
        Inst* base = expr(*index.kids[0]);
        Inst* i = expr(*index.kids[1]);
        Inst* addr = emit(Opc::ElemAddr, IrType::U64, {base, i});
        addr->memType = irType(scalar(index.kids[0]->type.base));
        return addr;
    }

    // a && b: b is only evaluated if a is true (a || b: only if a is false), as in C.
    // That matters on the GPU: 'i < n && x[i] > 0' must not read x[i] when i >= n.
    Inst* shortCircuit(const Expr& e) {
        Inst* lhs = expr(*e.kids[0]);
        Block* rhsB = fn_.newBlock();
        Block* join = fn_.newBlock();
        if (e.op == Op::LogAnd)
            condBranch(lhs, rhsB, join);  // lhs false: result is lhs (false)
        else
            condBranch(lhs, join, rhsB);  // lhs true: result is lhs (true)
        seal(rhsB);

        startBlock(rhsB);
        Inst* rhs = expr(*e.kids[1]);
        branch(join);

        seal(join);
        startBlock(join);
        Inst* phi = newPhi(join, IrType::Pred);
        phi->ops = {lhs, rhs};  // join->preds is {block of lhs, block of rhs}, in that order
        return phi;
    }
};

}  // namespace

IrModule generateIr(const Program& prog) {
    IrModule m;
    for (const auto& k : prog.kernels) {
        m.funcs.push_back(std::make_unique<IrFunction>());
        IrGen(*m.funcs.back()).kernel(*k);
    }
    return m;
}
