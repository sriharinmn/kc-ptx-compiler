// Pass: loop-invariant code motion (LICM).
// A computation inside a loop whose operands all come from outside the loop
// gives the same result in every iteration, so it is computed once, before
// the loop. Example: in row_sum, 'r * cols' moves out of the column loop.
//
// It moves to the loop's preheader: the one block outside the loop that jumps
// to the loop header. Only pure instructions move, never loads: a load could
// read memory the loop changes, and moving it in front of a loop that runs
// zero times could read out of bounds. Pure instructions cannot fault on the
// GPU (even integer division by zero just gives some value), so running them
// once more than needed is harmless.

#include <algorithm>

#include "analysis.h"
#include "opt.h"

namespace {

// The single block outside the loop that jumps to the header, if it ends in a plain br.
Block* preheader(const Loop& loop) {
    Block* pre = nullptr;
    for (Block* p : loop.header->preds) {
        if (loop.blocks.count(p)) continue;  // a jump back from inside the loop
        if (pre) return nullptr;             // two ways into the loop
        pre = p;
    }
    if (!pre || pre->terminator()->op != Opc::Br) return nullptr;
    return pre;
}

}  // namespace

int hoistLoopInvariants(IrFunction& f) {
    DomTree dt(f);
    int moved = 0;
    for (const Loop& loop : findLoops(f, dt)) {
        Block* pre = preheader(loop);
        if (!pre) continue;
        // Dominators first (reverse postorder), so an operand is moved before the
        // instructions that use it, and those can then move too.
        for (Block* b : dt.rpo()) {
            if (!loop.blocks.count(b)) continue;
            for (Inst* in : std::vector<Inst*>(b->insts)) {
                if (!isPure(in->op)) continue;
                bool invariant = std::none_of(in->ops.begin(), in->ops.end(),
                                              [&](const Inst* op) { return loop.blocks.count(op->parent); });
                if (!invariant) continue;
                eraseInst(in);
                in->parent = pre;
                pre->insts.insert(pre->insts.end() - 1, in);  // just before the jump into the loop
                moved++;
            }
        }
    }
    return moved;
}
