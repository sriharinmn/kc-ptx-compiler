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

    void function(const Function& f) {
        fn_.name = f.name;
        fn_.isKernel = f.isKernel;
        fn_.retType = irType(f.retType);
        entry_ = fn_.newBlock();
        entry_->sealed = true;  // nothing ever jumps to the entry block
        cur_ = entry_;
        for (size_t i = 0; i < f.params.size(); ++i) {
            const Var* p = f.params[i];
            Inst* in = emit(Opc::Param, irType(p->type));
            in->paramIndex = static_cast<int>(i);
            in->name = p->name;
            fn_.params.push_back(in);
            writeVar(p, entry_, in);
        }
        stmts(f.body);
        if (cur_) emit(Opc::Ret, IrType::Void);  // sema guarantees value-returning functions never get here
        fn_.renumber();
    }

private:
    // Where 'continue' and 'break' jump to, for the innermost loop.
    struct LoopTargets {
        Block* continueTo;
        Block* breakTo;
    };

    IrFunction& fn_;
    Block* entry_ = nullptr;
    Block* cur_ = nullptr;  // block we are adding code to; nullptr after a return (unreachable)
    std::map<const Var*, std::map<Block*, Inst*>> defs_;           // current value of a variable per block
    std::map<Block*, std::map<const Var*, Inst*>> incompletePhis_; // placeholder phis in unsealed blocks
    std::map<IrType, Inst*> undefs_;
    std::vector<LoopTargets> loops_;

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

    void condBranch(Inst* cond, Block* ifTrue, Block* ifFalse, Loc loc) {
        Inst* br = emit(Opc::CondBr, IrType::Void, {cond});
        br->loc = loc;  // the condition, named in divergence warnings
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
            case StmtKind::Shared: sharedArray(*s.var); break;
            case StmtKind::Assign: assign(s); break;
            case StmtKind::ExprStmt: expr(*s.value); break;
            case StmtKind::If: ifStmt(s); break;
            case StmtKind::While: loop(*s.cond, s.body, nullptr); break;
            case StmtKind::For:
                if (s.init) stmt(*s.init);
                loop(*s.cond, s.body, s.step.get());
                break;
            case StmtKind::Break: branch(loops_.back().breakTo); break;
            case StmtKind::Continue: branch(loops_.back().continueTo); break;
            case StmtKind::Return:
                if (s.value)
                    emit(Opc::Ret, IrType::Void, {expr(*s.value)});
                else
                    emit(Opc::Ret, IrType::Void);
                cur_ = nullptr;
                break;
            case StmtKind::Block: stmts(s.body); break;
        }
    }

    // The array's "value" is its address in shared memory.
    void sharedArray(const Var& v) {
        std::string name = fn_.name + "_" + v.name;
        for (int n = 2; std::any_of(fn_.sharedArrays.begin(), fn_.sharedArrays.end(),
                                    [&](const SharedArray& a) { return a.name == name; });
             ++n)
            name = fn_.name + "_" + v.name + std::to_string(n);  // same name used twice in one kernel
        fn_.sharedArrays.push_back({name, irType(scalar(v.type.base)), v.arraySize});
        Inst* addr = emit(Opc::SharedAddr, IrType::U64);
        addr->name = name;
        writeVar(&v, cur_, addr);
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
            if (s.compound) value = emit(binaryOpc(s.compoundOp), type, {load(target, addr), value});
            Inst* st = emit(Opc::Store, IrType::Void, {addr, value});
            st->memType = type;
            st->shared = target.kids[0]->type.shared;
        }
    }

    //   cond; condbr then, else   then: ...; br join   else: ...; br join   join: ...
    void ifStmt(const Stmt& s) {
        Inst* c = expr(*s.cond);
        Block* thenB = fn_.newBlock();
        Block* elseB = s.elseBody.empty() ? nullptr : fn_.newBlock();
        Block* join = fn_.newBlock();
        condBranch(c, thenB, elseB ? elseB : join, s.cond->loc);
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

    //   br header
    //   header: cond; condbr body, exit
    //   body:   ...; br step          ('continue' also jumps to step, 'break' to exit)
    //   step:   i += 1; br header     (a for loop's step; a while loop jumps straight to header)
    //   exit:   ...
    void loop(const Expr& cond, const std::vector<StmtPtr>& body, const Stmt* step) {
        Block* header = fn_.newBlock();
        branch(header);
        startBlock(header);  // not sealed yet: the jumps back from the body are still missing
        Inst* c = expr(cond);
        Block* bodyB = fn_.newBlock();
        Block* exit = fn_.newBlock();
        Block* stepB = step ? fn_.newBlock() : nullptr;
        condBranch(c, bodyB, exit, cond.loc);
        seal(bodyB);

        Block* next = stepB ? stepB : header;
        loops_.push_back({next, exit});
        startBlock(bodyB);
        stmts(body);
        if (cur_) branch(next);
        loops_.pop_back();

        if (stepB) {
            if (stepB->preds.empty()) {
                fn_.removeBlock(stepB);  // the body always breaks or returns: the step never runs
            } else {
                seal(stepB);
                startBlock(stepB);
                stmt(*step);
                branch(header);
            }
        }
        seal(header);  // now all predecessors of the header are known
        seal(exit);    // ... and of the exit (the loop condition plus every 'break')
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
            case ExprKind::Index: return load(e, elemAddr(e));
            case ExprKind::Call: return call(e);
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

    // Load the element 'index' (an Index expression) whose address is 'addr'.
    Inst* load(const Expr& index, Inst* addr) {
        Inst* ld = emit(Opc::Load, irType(index.type), {addr});
        ld->memType = ld->type;
        ld->shared = index.kids[0]->type.shared;
        return ld;
    }

    Inst* call(const Expr& e) {
        IrType type = irType(e.type);
        switch (e.builtin) {
            case BuiltinFn::Sqrt: return emit(Opc::Sqrt, type, {expr(*e.kids[0])});
            case BuiltinFn::Abs: return emit(Opc::Abs, type, {expr(*e.kids[0])});
            case BuiltinFn::Min:
            case BuiltinFn::Max: {
                Inst* a = expr(*e.kids[0]);
                Inst* b = expr(*e.kids[1]);
                return emit(e.builtin == BuiltinFn::Min ? Opc::Min : Opc::Max, type, {a, b});
            }
            case BuiltinFn::Popc: return emit(Opc::Popc, type, {expr(*e.kids[0])});
            case BuiltinFn::LaneId: {
                Inst* in = emit(Opc::SReg, type);
                in->name = "laneid";
                return in;
            }
            case BuiltinFn::Barrier:
            case BuiltinFn::WarpSum:
            case BuiltinFn::Ballot:
            case BuiltinFn::ShflDown:
            case BuiltinFn::ShflXor: {
                std::vector<Inst*> args;
                for (const auto& arg : e.kids) args.push_back(expr(*arg));
                Opc op = e.builtin == BuiltinFn::Barrier   ? Opc::Barrier
                         : e.builtin == BuiltinFn::WarpSum ? Opc::WarpSum
                         : e.builtin == BuiltinFn::Ballot  ? Opc::Ballot
                         : e.builtin == BuiltinFn::ShflDown ? Opc::ShflDown
                                                            : Opc::ShflXor;
                Inst* in = emit(op, type, args);
                in->loc = e.loc;  // for the divergence warnings
                return in;
            }
            case BuiltinFn::AtomicAdd: {
                const Expr& target = *e.kids[0];
                Inst* addr = elemAddr(target);
                Inst* add = emit(Opc::AtomicAdd, type, {addr, expr(*e.kids[1])});
                add->memType = type;
                add->shared = target.kids[0]->type.shared;
                return add;
            }
            case BuiltinFn::None: break;
        }
        std::vector<Inst*> args;
        for (const auto& arg : e.kids) args.push_back(expr(*arg));
        Inst* c = emit(Opc::Call, type, args);
        c->name = e.callee->name;
        return c;
    }

    // a && b: b is only evaluated if a is true (a || b: only if a is false), as in C.
    // That matters on the GPU: 'i < n && x[i] > 0' must not read x[i] when i >= n.
    Inst* shortCircuit(const Expr& e) {
        Inst* lhs = expr(*e.kids[0]);
        Block* rhsB = fn_.newBlock();
        Block* join = fn_.newBlock();
        if (e.op == Op::LogAnd)
            condBranch(lhs, rhsB, join, e.loc);  // lhs false: result is lhs (false)
        else
            condBranch(lhs, join, rhsB, e.loc);  // lhs true: result is lhs (true)
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
    for (const auto& f : prog.functions) {
        m.funcs.push_back(std::make_unique<IrFunction>());
        IrGen(*m.funcs.back()).function(*f);
    }
    return m;
}
