// Pass: loop unrolling. Runs 4 iterations of a simple counted loop per trip
// around the loop, so the GPU can have 4 loads in flight at once instead of
// waiting for each one before starting the next.
//
// Why kc must do this itself: ptxas does not unroll these loops (row_sum's
// SASS still has one load per iteration), while nvcc's NVVM stage unrolls them
// 4x before writing PTX. The benchmark showed nvcc 1.45x faster on such loops.
//
// Which loops: "for (let i = C; i < n; i += 1) { ... }" with C a constant >= 0,
// n not changed by the loop, no break or return inside, no loop inside, and a
// small body. Shape after the transformation:
//
//   preheader -> main:      phis; if (i < n && n - i > 3) run body 4 times, else go to rest
//                body x 4   (copies; each copy's results feed the next copy)
//                -> main
//   rest:        the original loop, which now runs the last 0-3 iterations
//
// "n - i > 3" is only trusted when i < n: then 0 <= i < n, so n - i cannot overflow.

#include <algorithm>
#include <map>
#include <set>

#include "analysis.h"
#include "opt.h"

namespace {

constexpr int kFactor = 4;
constexpr int kMaxBodySize = 50;  // IR instructions; bigger loops are left alone

bool isConstInt(const Inst* v, long long value) {
    return v->op == Opc::Const && v->type == IrType::S32 && v->imm == value;
}

// Everything the transformation needs to know about a loop that qualifies.
struct CountedLoop {
    Block* header = nullptr;
    Block* preheader = nullptr;
    Block* latch = nullptr;    // the one block in the loop that jumps back to the header
    Block* first = nullptr;    // first block of the body (the header's "true" side)
    Inst* counter = nullptr;   // the header phi that counts up by 1
    Inst* limit = nullptr;     // n
    std::vector<Block*> body;  // loop blocks except the header, in layout order
};

bool qualifies(const IrFunction& f, const Loop& loop, const std::vector<Loop>& all, CountedLoop& out) {
    Block* h = loop.header;
    if (h->preds.size() != 2) return false;
    for (const Loop& other : all)  // innermost loops only (unrolling an outer loop would copy the inner one)
        if (other.header != h && loop.blocks.count(other.header)) return false;

    out.header = h;
    for (Block* p : h->preds) (loop.blocks.count(p) ? out.latch : out.preheader) = p;
    if (!out.latch || !out.preheader || out.preheader->terminator()->op != Opc::Br) return false;

    // The header must be exactly: phis, "c = lt counter, n", "condbr c, body, exit".
    auto& insts = h->insts;
    size_t phis = std::find_if(insts.begin(), insts.end(), [](Inst* in) { return in->op != Opc::Phi; }) - insts.begin();
    if (insts.size() != phis + 2) return false;
    Inst* cmp = insts[phis];
    Inst* br = insts[phis + 1];
    if (cmp->op != Opc::Lt || cmp->ops[0]->type != IrType::S32 || br->op != Opc::CondBr || br->ops[0] != cmp)
        return false;
    if (!loop.blocks.count(br->targets[0]) || loop.blocks.count(br->targets[1])) return false;
    out.first = br->targets[0];
    if (out.first->preds.size() != 1) return false;

    // The counter: a header phi that starts at a constant >= 0 and goes up by 1 each time.
    out.counter = cmp->ops[0];
    out.limit = cmp->ops[1];
    if (out.counter->op != Opc::Phi || out.counter->parent != h || loop.blocks.count(out.limit->parent)) return false;
    size_t fromPre = out.preheader == h->preds[0] ? 0 : 1;
    const Inst* start = out.counter->ops[fromPre];
    const Inst* step = out.counter->ops[1 - fromPre];
    if (start->op != Opc::Const || start->imm < 0) return false;
    if (step->op != Opc::Add || !loop.blocks.count(step->parent)) return false;
    bool plusOne = (step->ops[0] == out.counter && isConstInt(step->ops[1], 1)) ||
                   (step->ops[1] == out.counter && isConstInt(step->ops[0], 1));
    if (!plusOne) return false;

    // Only the header may leave the loop, the body must be small, and it must not use
    // the comparison (the copies run without the header, so it isn't computed for them).
    int size = 0;
    for (const auto& b : f.blocks) {
        if (!loop.blocks.count(b.get())) continue;
        size += static_cast<int>(b->insts.size());
        if (b.get() == h) continue;
        out.body.push_back(b.get());
        for (Block* s : successors(b.get()))
            if (!loop.blocks.count(s)) return false;
        for (const Inst* in : b->insts)
            if (std::count(in->ops.begin(), in->ops.end(), cmp)) return false;
    }
    return size <= kMaxBodySize;
}

Inst* append(IrFunction& f, Block* b, Opc op, IrType type, std::vector<Inst*> ops) {
    Inst* in = f.newInst(op, type);
    in->ops = std::move(ops);
    in->parent = b;
    b->insts.push_back(in);
    return in;
}

void unroll(IrFunction& f, const CountedLoop& loop) {
    Block* h = loop.header;
    size_t fromPre = loop.preheader == h->preds[0] ? 0 : 1;
    std::vector<Inst*> phis;
    for (Inst* in : h->insts)
        if (in->op == Opc::Phi) phis.push_back(in);

    // The new main loop header, with one phi per original header phi.
    Block* main = f.newBlock();
    std::map<const Inst*, Inst*> mainPhi;
    for (Inst* p : phis) mainPhi[p] = append(f, main, Opc::Phi, p->type, {});

    // The blocks of the 4 copies (filled below).
    std::vector<std::map<const Block*, Block*>> blockCopy(kFactor);
    for (int k = 0; k < kFactor; ++k)
        for (Block* b : loop.body) blockCopy[k][b] = f.newBlock();

    // main: if (i < n && n - i > 3) go to copy 0, else to the original loop.
    Inst* i = mainPhi.at(loop.counter);
    Inst* inRange = append(f, main, Opc::Lt, IrType::Pred, {i, loop.limit});
    Inst* left = append(f, main, Opc::Sub, IrType::S32, {loop.limit, i});
    Inst* three = append(f, main, Opc::Const, IrType::S32, {});
    three->imm = kFactor - 1;
    Inst* enough = append(f, main, Opc::Gt, IrType::Pred, {left, three});
    Inst* go = append(f, main, Opc::And, IrType::Pred, {inRange, enough});
    Inst* condbr = append(f, main, Opc::CondBr, IrType::Void, {go});
    condbr->targets = {blockCopy[0].at(loop.first), h};

    // Copy the body 4 times. In copy k, the header phis stand for the values the
    // previous copy computed for the next iteration (copy 0: the main loop's phis).
    std::map<const Inst*, Inst*> prev;  // values of the previous copy
    for (int k = 0; k < kFactor; ++k) {
        std::map<const Inst*, Inst*> value;
        auto map = [](const std::map<const Inst*, Inst*>& m, Inst* v) {
            auto it = m.find(v);
            return it == m.end() ? v : it->second;  // values from outside the loop stay as they are
        };
        for (Inst* p : phis) value[p] = k == 0 ? mainPhi.at(p) : map(prev, p->ops[1 - fromPre]);
        for (Block* b : loop.body) {
            Block* nb = blockCopy[k].at(b);
            for (Block* p : b->preds) nb->preds.push_back(p == h ? (k == 0 ? main : blockCopy[k - 1].at(loop.latch))
                                                                 : blockCopy[k].at(p));
            for (const Inst* in : b->insts) {
                Inst* copy = f.newInst(in->op, in->type);
                int id = copy->id;
                *copy = *in;
                copy->id = id;
                copy->parent = nb;
                nb->insts.push_back(copy);
                value[in] = copy;
            }
        }
        for (Block* b : loop.body) {
            for (Inst* in : blockCopy[k].at(b)->insts) {
                for (Inst*& op : in->ops) op = map(value, op);
                for (Block*& t : in->targets)
                    t = t != h ? blockCopy[k].at(t) : k + 1 < kFactor ? blockCopy[k + 1].at(loop.first) : main;
            }
        }
        prev = value;
    }

    // Wire up the main loop: entered from the preheader, repeated from the last copy.
    main->preds = {loop.preheader, blockCopy[kFactor - 1].at(loop.latch)};
    for (Inst* p : phis) {
        auto it = prev.find(p->ops[1 - fromPre]);
        mainPhi.at(p)->ops = {p->ops[fromPre], it == prev.end() ? p->ops[1 - fromPre] : it->second};
    }
    loop.preheader->terminator()->targets = {main};
    // The original loop is now entered from main, with main's values.
    h->preds[fromPre] = main;
    for (Inst* p : phis) p->ops[fromPre] = mainPhi.at(p);

    // Layout: main and the copies go right before the original loop.
    auto& blocks = f.blocks;
    size_t added = 1 + kFactor * loop.body.size();
    auto headerAt = std::find_if(blocks.begin(), blocks.end(), [&](auto& b) { return b.get() == h; });
    std::rotate(headerAt, blocks.end() - static_cast<long>(added), blocks.end());
}

}  // namespace

int unrollLoops(IrFunction& f) {
    int unrolled = 0;
    std::set<const Block*> done;  // headers already handled (also the new main loops)
    for (bool again = true; again;) {
        again = false;
        DomTree dt(f);
        std::vector<Loop> loops = findLoops(f, dt);
        for (const Loop& loop : loops) {
            CountedLoop counted;
            if (done.count(loop.header) || !qualifies(f, loop, loops, counted)) continue;
            done.insert(loop.header);
            unroll(f, counted);
            done.insert(counted.preheader->terminator()->targets[0]);  // the new main loop
            unrolled++;
            again = true;  // the blocks changed: find the loops again
            break;
        }
    }
    return unrolled;
}
