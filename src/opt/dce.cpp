// Pass: dead code elimination (DCE).
// Start from the instructions that must stay (stores, atomics, barriers,
// calls, jumps, returns), mark what they use as live, then what those use,
// and so on. Everything left unmarked is never needed and is deleted.
//
// This "mark what is needed" approach also deletes dead cycles, which
// "delete if nobody uses it" would miss: e.g. a loop counter that only feeds
// its own phi and add, but is never used after the loop.

#include <algorithm>
#include <set>

#include "opt.h"

int removeDeadCode(IrFunction& f) {
    std::set<const Inst*> live;
    std::vector<const Inst*> work;
    for (const auto& b : f.blocks)
        for (const Inst* in : b->insts)
            if (hasSideEffects(in->op)) {
                live.insert(in);
                work.push_back(in);
            }
    while (!work.empty()) {
        const Inst* in = work.back();
        work.pop_back();
        for (const Inst* op : in->ops)
            if (live.insert(op).second) work.push_back(op);
    }

    int removed = 0;
    for (const auto& b : f.blocks) {
        auto& insts = b->insts;
        auto dead = std::stable_partition(insts.begin(), insts.end(), [&](const Inst* in) { return live.count(in) > 0; });
        for (auto it = dead; it != insts.end(); ++it) (*it)->parent = nullptr;
        removed += static_cast<int>(insts.end() - dead);
        insts.erase(dead, insts.end());
    }
    return removed;
}
