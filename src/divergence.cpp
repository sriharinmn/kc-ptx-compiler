#include "divergence.h"

#include <algorithm>

#include "analysis.h"

namespace {

// Values that are different in different threads, whatever their operands are.
bool startsDivergent(const Inst* in, const IrFunction& f) {
    switch (in->op) {
        case Opc::SReg: return in->name.rfind("tid.", 0) == 0 || in->name == "laneid";  // threadIdx, lane_id()
        case Opc::Param: return !f.isKernel;  // a device function may be called with per-thread values
        case Opc::AtomicAdd:                  // every thread gets a different old value
        case Opc::Call:                       // unknown: assume the worst
        case Opc::WarpSum: case Opc::ShflDown: case Opc::ShflXor: case Opc::Ballot:  // differ between warps
            return true;
        default:
            return false;
    }
}

const char* warpOpName(Opc op) {
    switch (op) {
        case Opc::WarpSum: return "warp_sum";
        case Opc::ShflDown: return "shfl_down";
        case Opc::ShflXor: return "shfl_xor";
        default: return "ballot";
    }
}

}  // namespace

DivergenceInfo analyzeDivergence(const IrFunction& f) {
    PostDomTree pdt(f);
    DomTree dt(f);
    std::vector<Loop> loops = findLoops(f, dt);
    DivergenceInfo info;
    std::set<const Block*> merges;  // blocks whose phis can pick different values in different threads

    // The facts feed each other (a divergent value makes a branch divergent, which
    // makes phis divergent, ...), so repeat until nothing new is found.
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& b : f.blocks) {
            for (const Inst* in : b->insts) {
                if (in->type == IrType::Void || info.divergent(in)) continue;
                bool fromOperand = std::any_of(in->ops.begin(), in->ops.end(),
                                               [&](const Inst* op) { return info.divergent(op); });
                bool atMerge = in->op == Opc::Phi && merges.count(b.get());
                if (startsDivergent(in, f) || fromOperand || atMerge) {
                    info.values.insert(in);
                    changed = true;
                }
            }
        }
        for (const auto& b : f.blocks) {
            const Inst* branch = b->terminator();
            if (branch->op != Opc::CondBr || !info.divergent(branch->ops[0])) continue;

            // 1. The threads split here. Walk each side until the paths meet again:
            //    the blocks on the way are run by only some of the threads.
            const Block* meet = pdt.ipdom(b.get());
            std::set<const Block*> side[2];
            for (int k = 0; k < 2; ++k) {
                std::vector<Block*> work = {branch->targets[k]};
                while (!work.empty()) {
                    Block* x = work.back();
                    work.pop_back();
                    if (!side[k].insert(x).second || x == meet) continue;
                    if (info.partial.emplace(x, branch).second) changed = true;
                    for (Block* s : successors(x)) work.push_back(s);
                }
            }
            // 2. A block both sides reach is where the split threads come back together,
            //    so its phis get different values in different threads.
            for (const Block* x : side[0])
                if (side[1].count(x) && merges.insert(x).second) changed = true;
            // 3. If this branch leaves a loop, threads leave after different numbers of
            //    rounds, so the values carried around the loop (header phis) end up different.
            for (const Loop& loop : loops) {
                bool leaves = !loop.blocks.count(branch->targets[0]) || !loop.blocks.count(branch->targets[1]);
                if (loop.blocks.count(b.get()) && leaves && merges.insert(loop.header).second) changed = true;
            }
        }
    }
    return info;
}

std::vector<std::string> divergenceWarnings(const IrModule& m) {
    std::vector<std::string> warnings;
    for (const auto& f : m.funcs) {
        DivergenceInfo info = analyzeDivergence(*f);
        for (const auto& b : f->blocks) {
            auto it = info.partial.find(b.get());
            if (it == info.partial.end()) continue;
            std::string because = "the branch on line " + std::to_string(it->second->loc.line) +
                                  " depends on the thread index";
            for (const Inst* in : b->insts) {
                std::string at = std::to_string(in->loc.line) + ":" + std::to_string(in->loc.col) + ": warning: ";
                if (in->op == Opc::Barrier)
                    warnings.push_back(at + "barrier() is only reached by some threads of the block (" + because +
                                       "); every thread of a block must reach the same barrier()");
                else if (isWarpOp(in->op))
                    warnings.push_back(at + warpOpName(in->op) + "() needs all 32 threads of the warp, but only some " +
                                       "may get here (" + because + ")");
            }
        }
    }
    return warnings;
}
