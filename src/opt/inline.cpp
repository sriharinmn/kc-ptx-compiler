// Pass: inlining. Replaces a call by a copy of the called function's body.
//
// Why kc must do this itself: ptxas does not inline .func calls (the SASS
// still has real CALL instructions, even at -O3). nvcc inlines device
// functions before it writes PTX, so any compiler that produces PTX has to
// do the same to get fast code.
//
// How, for  x = call f(a, b)  in block B:
//   1. split B at the call: the instructions after it move to a new block "after"
//   2. copy f's blocks into the caller; f's parameters become a and b
//   3. B jumps to the copy of f's entry block; every "ret v" in the copy jumps to "after"
//   4. x becomes v (or a phi in "after", when f returns from several places)

#include <algorithm>
#include <functional>
#include <map>
#include <set>

#include "opt.h"

namespace {

// Bigger functions (counted in IR instructions) stay real calls, to limit code growth.
constexpr int kMaxInlineSize = 100;

// Can f end up calling itself, directly or through other functions?
// Recursive functions are never inlined (the copying would never end).
bool isRecursive(const IrModule& m, const IrFunction& f) {
    std::set<const IrFunction*> seen;
    std::function<bool(const IrFunction&)> reaches = [&](const IrFunction& g) {
        for (const auto& b : g.blocks) {
            for (const Inst* in : b->insts) {
                if (in->op != Opc::Call) continue;
                const IrFunction* callee = m.find(in->name);
                if (callee == &f) return true;
                if (seen.insert(callee).second && reaches(*callee)) return true;
            }
        }
        return false;
    };
    return reaches(f);
}

void inlineCall(IrFunction& caller, Inst* call, const IrFunction& callee) {
    Block* before = call->parent;

    // New blocks: one copy per callee block, then "after". They are created at
    // the end of the layout and moved behind "before" at the end.
    std::map<const Block*, Block*> blockCopy;
    for (const auto& cb : callee.blocks) blockCopy[cb.get()] = caller.newBlock();
    Block* after = caller.newBlock();

    // 1. Split: everything after the call (including the jump at the end) moves to "after".
    auto pos = std::find(before->insts.begin(), before->insts.end(), call);
    after->insts.assign(pos + 1, before->insts.end());
    before->insts.erase(pos, before->insts.end());  // this also removes the call
    for (Inst* in : after->insts) in->parent = after;
    for (Block* s : successors(after)) std::replace(s->preds.begin(), s->preds.end(), before, after);

    // 2. Copy the instructions. A parameter is replaced by the matching argument.
    std::map<const Inst*, Inst*> valueCopy;
    for (const Inst* p : callee.params) valueCopy[p] = call->ops[p->paramIndex];
    for (const auto& cb : callee.blocks) {
        Block* nb = blockCopy.at(cb.get());
        for (const Inst* in : cb->insts) {
            if (in->op == Opc::Param) continue;
            Inst* copy = caller.newInst(in->op, in->type);
            int id = copy->id;
            *copy = *in;  // operands and targets still point into the callee: fixed below
            copy->id = id;
            copy->parent = nb;
            nb->insts.push_back(copy);
            valueCopy[in] = copy;
        }
    }
    // Every value has a copy now, so operands, targets and predecessor lists can be redirected.
    std::vector<Inst*> returned;  // value returned by each block that jumps to "after"
    for (const auto& cb : callee.blocks) {
        Block* nb = blockCopy.at(cb.get());
        for (Block* p : cb->preds) nb->preds.push_back(blockCopy.at(p));
        for (Inst* in : nb->insts) {
            for (Inst*& op : in->ops) op = valueCopy.at(op);
            for (Block*& t : in->targets) t = blockCopy.at(t);
        }
        Inst* last = nb->insts.back();
        if (last->op == Opc::Ret) {  // 3. "ret v" becomes "br after"
            returned.push_back(last->ops.empty() ? nullptr : last->ops[0]);
            last->op = Opc::Br;
            last->ops.clear();
            last->targets = {after};
            after->preds.push_back(nb);
        }
    }

    // 3. The caller jumps into the copied body.
    Block* entryCopy = blockCopy.at(callee.blocks[0].get());
    Inst* jump = caller.newInst(Opc::Br, IrType::Void);
    jump->targets = {entryCopy};
    jump->parent = before;
    before->insts.push_back(jump);
    entryCopy->preds.push_back(before);

    // 4. The call's result is the returned value.
    if (call->type != IrType::Void) {
        Inst* result = returned[0];
        if (returned.size() > 1) {
            result = caller.newInst(Opc::Phi, call->type);
            result->ops = returned;  // same order as after->preds
            result->parent = after;
            after->insts.insert(after->insts.begin(), result);
        }
        replaceAllUses(caller, call, result);
    }

    // Layout: put the copied blocks and "after" right behind "before", so the
    // code stays in source order and most jumps still fall through.
    auto& blocks = caller.blocks;
    auto beforeIt = std::find_if(blocks.begin(), blocks.end(), [&](auto& b) { return b.get() == before; });
    std::rotate(beforeIt + 1, blocks.end() - static_cast<long>(callee.blocks.size() + 1), blocks.end());
}

// The first call in f that should be inlined, or nullptr.
Inst* findCallToInline(const IrModule& m, const IrFunction& f, const std::set<const IrFunction*>& recursive) {
    for (const auto& b : f.blocks) {
        for (Inst* in : b->insts) {
            if (in->op != Opc::Call) continue;
            const IrFunction* callee = m.find(in->name);
            // An extern function's body is in another file, so it can't be copied: the call stays.
            if (!callee->isExtern && !recursive.count(callee) && instCount(*callee) <= kMaxInlineSize) return in;
        }
    }
    return nullptr;
}

}  // namespace

int inlineCalls(IrModule& m) {
    std::set<const IrFunction*> recursive;
    for (const auto& f : m.funcs)
        if (isRecursive(m, *f)) recursive.insert(f.get());

    int inlined = 0;
    for (const auto& f : m.funcs) {
        // Inlining f's callee may bring in new calls (the callee's own calls); they are found next time round.
        while (Inst* call = findCallToInline(m, *f, recursive)) {
            inlineCall(*f, call, *m.find(call->name));
            inlined++;
        }
    }
    return inlined;
}

// Device functions are private to the module (.func without .visible), so one
// that is no longer called (because every call was inlined) can be deleted.
// Exported functions stay: another module may call them.
int removeUncalledFunctions(IrModule& m) {
    int removed = 0;
    for (bool again = true; again;) {  // deleting a function can leave its own callees uncalled
        again = false;
        std::set<std::string> called;
        for (const auto& f : m.funcs)
            for (const auto& b : f->blocks)
                for (const Inst* in : b->insts)
                    if (in->op == Opc::Call && in->name != f->name) called.insert(in->name);
        for (auto it = m.funcs.begin(); it != m.funcs.end();) {
            if (!(*it)->isKernel && !(*it)->isExport && !called.count((*it)->name)) {
                it = m.funcs.erase(it);
                removed++;
                again = true;
            } else {
                ++it;
            }
        }
    }
    return removed;
}
