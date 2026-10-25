// Pass: simplify. Small local clean-ups:
//   - constant folding: 2 + 3 becomes 5, computed exactly as the GPU would
//   - simple algebra:   x + 0, x * 1, x | 0, ... become x;  x * 0, x - x become 0
//   - a branch on a constant becomes a plain jump; blocks nobody can reach are deleted
//   - a phi whose operands are all the same value becomes that value
//   - a block whose only predecessor jumps straight to it is glued onto that predecessor
//
// Folding must give bit-for-bit the same answer as the GPU instruction would,
// otherwise -O1 would change what programs compute. When the GPU's answer is
// not clear (division by zero, NaN results), the instruction is left alone.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>

#include "analysis.h"
#include "opt.h"

namespace {

bool isConst(const Inst* v) { return v->op == Opc::Const; }
bool isInt(const Inst* v, long long value) { return isConst(v) && v->type == IrType::S32 && v->imm == value; }

// Turn 'in' into a constant in place: its users now see the constant.
// (A comparison becomes a pred constant: imm is 0 or 1.)
void becomeInt(Inst* in, long long value) {
    in->op = Opc::Const;
    in->ops.clear();
    in->imm = value;
}
void becomeFloat(Inst* in, float value) {
    in->op = Opc::Const;
    in->ops.clear();
    in->fimm = value;
}

// 32-bit integer math as the GPU does it: two's complement, wrapping on overflow.
// (Done with unsigned numbers, because signed overflow is undefined in C++.)
std::optional<long long> foldInt(Opc op, int32_t a, int32_t b) {
    uint32_t ua = static_cast<uint32_t>(a), ub = static_cast<uint32_t>(b);
    switch (op) {
        case Opc::Add: return static_cast<int32_t>(ua + ub);
        case Opc::Sub: return static_cast<int32_t>(ua - ub);
        case Opc::Mul: return static_cast<int32_t>(ua * ub);  // mul.lo: the low 32 bits
        case Opc::Div:
        case Opc::Rem:
            // No defined answer for these: leave them for the GPU.
            if (b == 0 || (a == INT32_MIN && b == -1)) return std::nullopt;
            return op == Opc::Div ? a / b : a % b;  // C++ and PTX both round toward zero
        case Opc::And: return a & b;
        case Opc::Or: return a | b;
        case Opc::Xor: return a ^ b;
        // PTX reads the shift amount as unsigned and clamps it to 32:
        // shl then gives 0, and shr.s32 fills with the sign bit.
        case Opc::Shl: return ub >= 32 ? 0 : static_cast<int32_t>(ua << ub);
        case Opc::Shr: return ub >= 32 ? (a < 0 ? -1 : 0) : a >> ub;
        case Opc::Min: return std::min(a, b);
        case Opc::Max: return std::max(a, b);
        case Opc::Lt: return a < b;
        case Opc::Le: return a <= b;
        case Opc::Gt: return a > b;
        case Opc::Ge: return a >= b;
        case Opc::Eq: return a == b;
        case Opc::Ne: return a != b;
        default: return std::nullopt;
    }
}

// f32 math: the CPU and the GPU both follow IEEE 754 here (add/sub/mul, div.rn, sqrt.rn).
std::optional<float> foldFloat(Opc op, float a, float b) {
    switch (op) {
        case Opc::Add: return a + b;
        case Opc::Sub: return a - b;
        case Opc::Mul: return a * b;
        case Opc::Div: return a / b;
        case Opc::Min:
        case Opc::Max:
            // NaN inputs and min(-0.0, +0.0) are corner cases: not worth the risk.
            if (std::isnan(a) || std::isnan(b) || (a == 0 && b == 0)) return std::nullopt;
            return op == Opc::Min ? std::min(a, b) : std::max(a, b);
        default: return std::nullopt;
    }
}

// Float comparisons as in C: anything compared with NaN is false, except !=.
std::optional<long long> compareFloat(Opc op, float a, float b) {
    switch (op) {
        case Opc::Lt: return a < b;
        case Opc::Le: return a <= b;
        case Opc::Gt: return a > b;
        case Opc::Ge: return a >= b;
        case Opc::Eq: return a == b;
        case Opc::Ne: return a != b;
        default: return std::nullopt;
    }
}

std::optional<long long> foldBool(Opc op, bool a, bool b) {
    switch (op) {
        case Opc::And: return a && b;
        case Opc::Or: return a || b;
        case Opc::Xor:
        case Opc::Ne: return a != b;
        case Opc::Eq: return a == b;
        default: return std::nullopt;
    }
}

// cvt.rzi.s32.f32: round toward zero, clamp to the i32 range, NaN gives 0.
long long floatToInt(float x) {
    if (std::isnan(x)) return 0;
    if (x >= 2147483648.0f) return INT32_MAX;
    if (x <= -2147483648.0f) return INT32_MIN;
    return static_cast<int32_t>(x);
}

// If every operand of 'in' is a constant, compute the result now.
bool foldConstant(Inst* in) {
    if (in->ops.empty() || !isPure(in->op)) return false;
    for (const Inst* op : in->ops)
        if (!isConst(op)) return false;

    const Inst* a = in->ops[0];
    IrType t = a->type;
    int32_t ia = static_cast<int32_t>(a->imm);
    if (in->ops.size() == 2) {
        const Inst* b = in->ops[1];
        int32_t ib = static_cast<int32_t>(b->imm);
        std::optional<long long> i;
        if (t == IrType::S32) i = foldInt(in->op, ia, ib);
        if (t == IrType::Pred) i = foldBool(in->op, ia, ib);
        if (t == IrType::F32 && isCompare(in->op)) i = compareFloat(in->op, a->fimm, b->fimm);
        if (i) {
            becomeInt(in, *i);
            return true;
        }
        if (t == IrType::F32) {
            std::optional<float> f = foldFloat(in->op, a->fimm, b->fimm);
            // NaN bit patterns differ between CPU and GPU, so a NaN result is left to the GPU.
            if (!f || std::isnan(*f)) return false;
            becomeFloat(in, *f);
            return true;
        }
        return false;
    }

    switch (in->op) {  // one operand
        case Opc::Neg:
            if (t == IrType::S32) becomeInt(in, static_cast<int32_t>(0u - static_cast<uint32_t>(ia)));
            else becomeFloat(in, -a->fimm);
            return true;
        case Opc::Not:
            becomeInt(in, t == IrType::Pred ? !ia : ~ia);
            return true;
        case Opc::Abs:
            if (t == IrType::S32) {
                if (ia == INT32_MIN) return false;  // has no positive i32
                becomeInt(in, std::abs(ia));
            } else {
                becomeFloat(in, std::fabs(a->fimm));
            }
            return true;
        case Opc::Sqrt:
            if (a->fimm < 0 || std::isnan(a->fimm)) return false;  // NaN result
            becomeFloat(in, std::sqrt(a->fimm));
            return true;
        case Opc::CvtI2F: becomeFloat(in, static_cast<float>(ia)); return true;  // rounds to nearest, like cvt.rn
        case Opc::CvtF2I: becomeInt(in, floatToInt(a->fimm)); return true;
        default: return false;
    }
}

// Creates an instruction right before 'at', in the same block.
Inst* insertBefore(IrFunction& f, Inst* at, Opc op, IrType type, std::vector<Inst*> ops) {
    Inst* in = f.newInst(op, type);
    in->ops = std::move(ops);
    in->parent = at->parent;
    in->loc = at->loc;  // same source line as the code it replaces
    auto& insts = at->parent->insts;
    insts.insert(std::find(insts.begin(), insts.end(), at), in);
    return in;
}
Inst* intBefore(IrFunction& f, Inst* at, long long value) {
    Inst* c = insertBefore(f, at, Opc::Const, IrType::S32, {});
    c->imm = value;
    return c;
}

// Can v never be negative? Thread and block numbers can't, nor can constants
// >= 0, and some operations keep a value >= 0. A phi is >= 0 if all its inputs
// are; for a phi we are already checking (a loop), we assume yes: that is a proof
// by induction (true when the loop starts, and each round keeps it true).
bool nonNegative(const Inst* v, std::set<const Inst*>& checking) {
    auto nn = [&](const Inst* x) { return nonNegative(x, checking); };
    switch (v->op) {
        case Opc::Const: return v->type == IrType::S32 && v->imm >= 0;
        case Opc::SReg: return true;  // %tid, %ctaid, %ntid, %nctaid, %laneid
        case Opc::And: return nn(v->ops[0]) || nn(v->ops[1]);  // the sign bit stays 0
        case Opc::Shr: case Opc::Rem: return nn(v->ops[0]);
        case Opc::Div: return nn(v->ops[0]) && nn(v->ops[1]);
        case Opc::Min: return nn(v->ops[0]) && nn(v->ops[1]);
        case Opc::Max: return nn(v->ops[0]) || nn(v->ops[1]);
        case Opc::Phi: {
            if (checking.count(v)) return true;  // the induction assumption
            checking.insert(v);
            bool all = std::all_of(v->ops.begin(), v->ops.end(), nn);
            checking.erase(v);
            return all;
        }
        default: return false;  // e.g. x + y can overflow into the negative numbers
    }
}

// Strength reduction: x / 2^k and x % 2^k without a division instruction.
// ptxas turns div.s32 into a ~20-instruction division routine, even for "/ 2"
// (checked in the SASS), so kc uses shifts instead, like nvcc does:
//   x >= 0:     x / 2^k = x >> k                  x % 2^k = x & (2^k - 1)
//   otherwise:  t = x + (x < 0 ? 2^k - 1 : 0)     (so the shift rounds toward zero, like division)
//               x / 2^k = t >> k                  x % 2^k = x - (t & -2^k)
bool divideByPowerOfTwo(IrFunction& f, Inst* in) {
    Inst* x = in->ops[0];
    Inst* d = in->ops[1];
    if (d->op != Opc::Const || d->imm < 2 || (d->imm & (d->imm - 1)) != 0) return false;
    int k = 0;
    while ((1LL << k) != d->imm) k++;

    std::set<const Inst*> checking;
    Inst* result;
    if (nonNegative(x, checking)) {
        result = in->op == Opc::Div ? insertBefore(f, in, Opc::Shr, IrType::S32, {x, intBefore(f, in, k)})
                                    : insertBefore(f, in, Opc::And, IrType::S32, {x, intBefore(f, in, d->imm - 1)});
    } else {
        Inst* sign = insertBefore(f, in, Opc::Shr, IrType::S32, {x, intBefore(f, in, 31)});  // -1 if x < 0, else 0
        Inst* bias = insertBefore(f, in, Opc::And, IrType::S32, {sign, intBefore(f, in, d->imm - 1)});
        Inst* t = insertBefore(f, in, Opc::Add, IrType::S32, {x, bias});
        if (in->op == Opc::Div) {
            result = insertBefore(f, in, Opc::Shr, IrType::S32, {t, intBefore(f, in, k)});
        } else {
            Inst* rounded = insertBefore(f, in, Opc::And, IrType::S32, {t, intBefore(f, in, -d->imm)});
            result = insertBefore(f, in, Opc::Sub, IrType::S32, {x, rounded});
        }
    }
    replaceAllUses(f, in, result);
    eraseInst(in);
    return true;
}

// x + 0 -> x, x * 1 -> x, x * 0 -> 0, ... Only for i32: for floats most of
// these are not exact (x + 0.0 is wrong for x = -0.0; x * 0.0 is wrong for NaN).
bool simplifyAlgebra(IrFunction& f, Inst* in) {
    if (in->ops.size() != 2) return false;
    Inst* a = in->ops[0];
    Inst* b = in->ops[1];
    Inst* same = nullptr;  // set if the instruction just gives back this value
    if (in->op == Opc::ElemAddr && isInt(b, 0)) same = a;  // the address of p[0] is p
    if (in->type != IrType::S32 && !same) return false;
    // (x + c1) + c2 -> x + (c1 + c2). Turns the unrolled loop's i+1+1+1 into i+3.
    if (in->op == Opc::Add && isConst(b) && a->op == Opc::Add && isConst(a->ops[1])) {
        in->ops = {a->ops[0], intBefore(f, in, static_cast<int32_t>(static_cast<uint32_t>(a->ops[1]->imm) +
                                                                     static_cast<uint32_t>(b->imm)))};
        return true;
    }
    if ((in->op == Opc::Div || in->op == Opc::Rem) && divideByPowerOfTwo(f, in)) return true;
    switch (in->op) {
        case Opc::Add:
        case Opc::Or:
            if (isInt(b, 0)) same = a;
            else if (isInt(a, 0)) same = b;
            break;
        case Opc::Sub:
        case Opc::Xor:
            if (isInt(b, 0)) {
                same = a;
            } else if (a == b) {
                becomeInt(in, 0);
                return true;
            }
            break;
        case Opc::Mul:
            if (isInt(b, 1)) {
                same = a;
            } else if (isInt(a, 1)) {
                same = b;
            } else if (isInt(a, 0) || isInt(b, 0)) {
                becomeInt(in, 0);
                return true;
            }
            break;
        case Opc::Div:
        case Opc::Shl:
        case Opc::Shr:
            if (isInt(b, in->op == Opc::Div ? 1 : 0)) same = a;
            break;
        case Opc::And:
            if (isInt(a, 0) || isInt(b, 0)) {
                becomeInt(in, 0);
                return true;
            }
            break;
        default: break;
    }
    if (!same) return false;
    replaceAllUses(f, in, same);
    eraseInst(in);
    return true;
}

// phi(v, v, ...) or phi(v, itself) is just v.
bool simplifyPhi(IrFunction& f, Inst* phi) {
    Inst* same = nullptr;
    for (Inst* op : phi->ops) {
        if (op == phi || op == same) continue;
        if (same) return false;  // two different values: a real phi
        same = op;
    }
    if (!same) return false;
    replaceAllUses(f, phi, same);
    eraseInst(phi);
    return true;
}

// condbr on a constant -> br to the side that is always taken.
bool foldBranch(Block* b) {
    Inst* t = b->terminator();
    if (t->op != Opc::CondBr || !isConst(t->ops[0])) return false;
    bool cond = t->ops[0]->imm != 0;
    removeEdge(b, cond ? t->targets[1] : t->targets[0]);
    t->targets = {cond ? t->targets[0] : t->targets[1]};
    t->op = Opc::Br;
    t->ops.clear();
    return true;
}

int removeUnreachableBlocks(IrFunction& f) {
    std::vector<Block*> order = reversePostOrder(f);
    std::set<Block*> reachable(order.begin(), order.end());
    std::vector<Block*> dead;
    for (const auto& b : f.blocks)
        if (!reachable.count(b.get())) dead.push_back(b.get());
    for (Block* d : dead)
        for (Block* s : successors(d)) removeEdge(d, s);
    for (Block* d : dead) f.removeBlock(d);
    return static_cast<int>(dead.size());
}

// If b's only predecessor p ends with "br b", move b's instructions onto the end of p.
bool mergeIntoPredecessor(IrFunction& f, Block* b) {
    if (b == f.blocks[0].get() || b->preds.size() != 1) return false;
    Block* p = b->preds[0];
    if (p == b || p->terminator()->op != Opc::Br) return false;

    while (b->insts.front()->op == Opc::Phi) {  // with one predecessor, a phi has one operand
        Inst* phi = b->insts.front();
        replaceAllUses(f, phi, phi->ops[0]);
        eraseInst(phi);
    }
    eraseInst(p->terminator());
    for (Inst* in : b->insts) {
        in->parent = p;
        p->insts.push_back(in);
    }
    b->insts.clear();
    for (Block* s : successors(p)) std::replace(s->preds.begin(), s->preds.end(), b, p);
    f.removeBlock(b);
    return true;
}

}  // namespace

int simplify(IrFunction& f) {
    int changes = 0;
    for (bool again = true; again;) {
        again = false;
        for (const auto& b : f.blocks) {
            std::vector<Inst*> insts = b->insts;  // a copy: instructions may be erased while we go
            for (Inst* in : insts) {
                if (foldConstant(in) || simplifyAlgebra(f, in) || (in->op == Opc::Phi && simplifyPhi(f, in))) {
                    changes++;
                    again = true;
                }
            }
            if (foldBranch(b.get())) {
                changes++;
                again = true;
            }
        }
        int removed = removeUnreachableBlocks(f);
        for (size_t i = 0; i < f.blocks.size();) {
            if (mergeIntoPredecessor(f, f.blocks[i].get()))
                removed++;  // block i is gone; the next block is now at index i
            else
                ++i;
        }
        if (removed) {
            changes += removed;
            again = true;
        }
    }
    return changes;
}
