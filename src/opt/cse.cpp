// Pass: common subexpression elimination (CSE).
// If the same pure computation (same opcode, same operands) was already done
// in a block that dominates this one, reuse that value instead of computing it
// again. Example: saxpy computes the address of y[i] twice; after CSE, once.
//
// It walks the dominator tree from the entry, remembering every pure value it
// has seen on the way down. Values seen in a dominator are always available.
//
// Loads are reused too, but carefully, because memory can change:
//   - any store, atomic, barrier or call forgets all remembered loads
//     (kc does not know which pointers point to the same memory)
//   - remembered loads only carry into a block whose single predecessor is the
//     block we just left, so no other path could have changed memory first.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <tuple>

#include "analysis.h"
#include "opt.h"

namespace {

// Everything that decides an instruction's result. Equal keys = equal values.
using Key = std::tuple<Opc, IrType, IrType, bool, long long, uint32_t, std::string, std::vector<Inst*>>;

bool isCommutative(Opc op) {
    return op == Opc::Add || op == Opc::Mul || op == Opc::And || op == Opc::Or || op == Opc::Xor ||
           op == Opc::Eq || op == Opc::Ne;
}

Key keyOf(const Inst* in) {
    std::vector<Inst*> ops = in->ops;
    if (isCommutative(in->op)) std::sort(ops.begin(), ops.end(), std::less<Inst*>());  // a+b == b+a
    uint32_t floatBits;
    std::memcpy(&floatBits, &in->fimm, sizeof floatBits);
    return {in->op, in->type, in->memType, in->shared, in->imm, floatBits, in->name, ops};
}

using Table = std::map<Key, Inst*>;

class Cse {
public:
    Cse(IrFunction& f) : f_(f), dt_(f) {}

    int run() {
        visit(f_.blocks[0].get(), Table{}, Table{});
        return removed_;
    }

private:
    IrFunction& f_;
    DomTree dt_;
    int removed_ = 0;

    // 'pure' and 'loads' are copies: what a block learns is only passed down to the blocks it dominates.
    void visit(Block* b, Table pure, Table loads) {
        if (b->preds.size() != 1) loads.clear();
        for (Inst* in : std::vector<Inst*>(b->insts)) {
            if (isPure(in->op) || in->op == Opc::Load) {
                Table& seen = in->op == Opc::Load ? loads : pure;
                auto [it, isNew] = seen.emplace(keyOf(in), in);
                if (!isNew) {
                    replaceAllUses(f_, in, it->second);
                    eraseInst(in);
                    removed_++;
                }
            } else if (hasSideEffects(in->op) && !isTerminator(in->op)) {
                loads.clear();  // a store, atomic, barrier or call: memory may have changed
            }
        }
        for (Block* child : dt_.children(b)) visit(child, pure, loads);
    }
};

}  // namespace

int eliminateCommonSubexpressions(IrFunction& f) { return Cse(f).run(); }
