#include "verify.h"

#include <map>
#include <set>

#include "analysis.h"

namespace {

class Verifier {
public:
    Verifier(const IrFunction& f, const std::string& pass) : f_(f), pass_(pass) {}

    void run() {
        if (f_.blocks.empty()) fail(nullptr, nullptr, "function has no blocks");
        for (const auto& b : f_.blocks) {
            blocks_.insert(b.get());
            for (size_t i = 0; i < b->insts.size(); ++i) where_[b->insts[i]] = {b.get(), i};
        }
        const Block* entry = f_.blocks[0].get();
        if (!entry->preds.empty()) fail(entry, nullptr, "the entry block must not have predecessors");

        for (const auto& b : f_.blocks) blockShape(b.get());
        predecessorLists();

        DomTree dt(f_);
        for (const auto& b : f_.blocks) {
            if (!dt.reachable(b.get())) fail(b.get(), nullptr, "block cannot be reached from the entry");
            for (const Inst* in : b->insts) {
                operandShape(in);
                operandsDominate(in, dt);
            }
        }
    }

private:
    struct Place {
        const Block* block;
        size_t index;
    };

    const IrFunction& f_;
    const std::string& pass_;
    std::set<const Block*> blocks_;
    std::map<const Inst*, Place> where_;  // every instruction that is in a block

    [[noreturn]] void fail(const Block* b, const Inst* in, const std::string& msg) {
        std::string s = "internal compiler error after '" + pass_ + "': in " + f_.name;
        if (b) s += ", bb" + std::to_string(b->id);
        s += ": " + msg;
        if (in) s += "  [" + printInst(in) + "]";
        throw InternalError(s);
    }

    void blockShape(const Block* b) {
        if (b->insts.empty()) fail(b, nullptr, "empty block");
        bool pastPhis = false;
        for (size_t i = 0; i < b->insts.size(); ++i) {
            const Inst* in = b->insts[i];
            if (in->parent != b) fail(b, in, "instruction does not know its block");
            bool last = i + 1 == b->insts.size();
            if (isTerminator(in->op) != last)
                fail(b, in, last ? "block does not end with br/condbr/ret" : "jump in the middle of a block");
            if (in->op != Opc::Phi)
                pastPhis = true;
            else if (pastPhis)
                fail(b, in, "phi after a normal instruction");
            for (const Block* t : in->targets)
                if (!blocks_.count(t)) fail(b, in, "jumps to a block that is not in the function");
        }
    }

    // b->preds must list exactly the blocks that jump to b (once per jump).
    void predecessorLists() {
        std::map<const Block*, std::multiset<const Block*>> jumpsInto;
        for (const auto& b : f_.blocks)
            for (const Block* t : successors(b.get())) jumpsInto[t].insert(b.get());
        for (const auto& b : f_.blocks) {
            std::multiset<const Block*> listed(b->preds.begin(), b->preds.end());
            if (listed != jumpsInto[b.get()])
                fail(b.get(), nullptr, "predecessor list does not match the jumps into this block");
        }
    }

    void operandShape(const Inst* in) {
        auto count = [&](size_t n) {
            if (in->ops.size() != n) fail(in->parent, in, "expected " + std::to_string(n) + " operand(s)");
        };
        auto type = [&](size_t i, IrType t) {
            if (in->ops[i]->type != t) fail(in->parent, in, "operand " + std::to_string(i) + " has the wrong type");
        };
        switch (in->op) {
            case Opc::Const: case Opc::Param: case Opc::SReg: case Opc::SharedAddr: case Opc::Barrier:
                count(0);
                break;
            case Opc::Add: case Opc::Sub: case Opc::Mul: case Opc::Div: case Opc::Rem:
            case Opc::And: case Opc::Or: case Opc::Xor: case Opc::Shl: case Opc::Shr:
            case Opc::Min: case Opc::Max:
                count(2);
                type(0, in->type);
                type(1, in->type);
                break;
            case Opc::Neg: case Opc::Not: case Opc::Abs: case Opc::Sqrt: case Opc::Popc: case Opc::WarpSum:
                count(1);
                type(0, in->type);
                break;
            case Opc::ShflDown: case Opc::ShflXor:
                count(2);
                type(0, in->type);
                type(1, IrType::S32);
                break;
            case Opc::Ballot: count(1); type(0, IrType::Pred); break;
            case Opc::Lt: case Opc::Le: case Opc::Gt: case Opc::Ge: case Opc::Eq: case Opc::Ne:
                count(2);
                type(1, in->ops[0]->type);
                if (in->type != IrType::Pred) fail(in->parent, in, "a comparison must produce a pred");
                break;
            case Opc::CvtI2F: count(1); type(0, IrType::S32); break;
            case Opc::CvtF2I: count(1); type(0, IrType::F32); break;
            case Opc::ElemAddr: count(2); type(0, IrType::U64); type(1, IrType::S32); break;
            case Opc::Load: count(1); type(0, IrType::U64); break;
            case Opc::Store: count(2); type(0, IrType::U64); type(1, in->memType); break;
            case Opc::AtomicAdd: count(2); type(0, IrType::U64); type(1, in->type); break;
            case Opc::Call: break;  // checked against the callee in verifyModule
            case Opc::Phi:
                count(in->parent->preds.size());  // one value per way into the block
                for (size_t i = 0; i < in->ops.size(); ++i) type(i, in->type);
                break;
            case Opc::Br:
                count(0);
                if (in->targets.size() != 1) fail(in->parent, in, "br needs 1 target");
                break;
            case Opc::CondBr:
                count(1);
                type(0, IrType::Pred);
                if (in->targets.size() != 2) fail(in->parent, in, "condbr needs 2 targets");
                break;
            case Opc::Ret:
                count(f_.retType == IrType::Void ? 0 : 1);
                if (!in->ops.empty()) type(0, f_.retType);
                break;
        }
    }

    // SSA's main rule: a value must be computed before every use, on every path.
    void operandsDominate(const Inst* in, const DomTree& dt) {
        const Place& use = where_.at(in);
        for (size_t i = 0; i < in->ops.size(); ++i) {
            auto it = where_.find(in->ops[i]);
            if (it == where_.end()) fail(use.block, in, "uses a value that is no longer in the function");
            const Place& def = it->second;
            if (in->op == Opc::Phi) {
                // The value is needed at the end of the predecessor it comes from.
                if (!dt.dominates(def.block, in->parent->preds[i]))
                    fail(use.block, in, "phi operand is not ready at the end of its predecessor");
            } else if (def.block == use.block) {
                if (def.index >= use.index) fail(use.block, in, "uses a value before it is computed");
            } else if (!dt.dominates(def.block, use.block)) {
                fail(use.block, in, "uses a value that is not computed on every path to here");
            }
        }
    }
};

}  // namespace

void verifyFunction(const IrFunction& f, const std::string& afterPass) { Verifier(f, afterPass).run(); }

void verifyModule(const IrModule& m, const std::string& afterPass) {
    for (const auto& f : m.funcs) {
        verifyFunction(*f, afterPass);
        // Every call must match the function it calls.
        for (const auto& b : f->blocks) {
            for (const Inst* in : b->insts) {
                if (in->op != Opc::Call) continue;
                const IrFunction* callee = m.find(in->name);
                std::string where = "internal compiler error after '" + afterPass + "': in " + f->name + ": ";
                if (!callee || callee->isKernel) throw InternalError(where + "call to missing function " + in->name);
                bool ok = in->ops.size() == callee->params.size() && in->type == callee->retType;
                for (size_t i = 0; ok && i < in->ops.size(); ++i) ok = in->ops[i]->type == callee->params[i]->type;
                if (!ok) throw InternalError(where + "call does not match the signature of " + in->name);
            }
        }
    }
}
