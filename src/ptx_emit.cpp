#include "ptx_emit.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

// Leaving SSA ("phi elimination"):
// PTX has no phi instruction, so each phi gets two registers:
//   - its normal register R
//   - a "mailbox" register M
// Every predecessor block writes its value into M right before jumping
// ("mov M, value"), and the phi's block starts by copying "mov R, M".
// Using a separate mailbox per phi keeps this correct even when phis read each
// other (e.g. a loop that swaps two variables). The extra movs cost nothing in
// the end: ptxas removes them when it allocates the real registers.

namespace {

// PTX writes f32 constants as hex bits (0f3F800000 = 1.0), which is exact.
std::string f32Hex(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    char buf[16];
    std::snprintf(buf, sizeof buf, "0f%08X", bits);
    return buf;
}

// Type suffix of a PTX instruction, e.g. the ".s32" in "add.s32".
std::string sfx(IrType t) {
    switch (t) {
        case IrType::Pred: return "pred";
        case IrType::S32: return "s32";
        case IrType::F32: return "f32";
        case IrType::U64: return "u64";
        case IrType::Void: return "";
    }
    return "";
}

class FunctionEmitter {
public:
    FunctionEmitter(const IrFunction& fn, int index) : fn_(fn), index_(index) {}

    std::string run() {
        // 1. Give every value a register (and every phi a mailbox).
        for (const auto& b : fn_.blocks) {
            for (const Inst* in : b->insts) {
                if (in->type == IrType::Void) continue;
                reg_[in] = newReg(in->type);
                if (in->op == Opc::Phi) mailbox_[in] = newReg(in->type);
            }
        }

        // 2. The body, block by block in layout order.
        for (size_t i = 0; i < fn_.blocks.size(); ++i) {
            const Block* b = fn_.blocks[i].get();
            const Block* next = i + 1 < fn_.blocks.size() ? fn_.blocks[i + 1].get() : nullptr;
            if (!b->preds.empty()) body_ << label(b) << ":\n";
            for (const Inst* in : b->insts) {
                if (in->op == Opc::Phi) {
                    ins("mov." + sfx(in->type) + " " + r(in) + ", " + mailbox_.at(in));
                } else if (isTerminator(in->op)) {
                    phiMoves(b);
                    terminator(in, next);
                } else {
                    inst(in);
                }
            }
        }

        // 3. Signature and register declarations (the counts are only known now).
        std::ostringstream out;
        out << ".visible .entry " << fn_.name << "(";
        for (size_t i = 0; i < fn_.params.size(); ++i) {
            IrType t = fn_.params[i]->type;
            const char* ptxType = t == IrType::F32 ? ".f32" : t == IrType::U64 ? ".u64" : ".u32";
            out << (i ? ",\n" : "\n") << "\t.param " << ptxType << " " << paramName(static_cast<int>(i));
        }
        out << "\n)\n{\n";
        static const char* kDecl[] = {"", ".pred", ".b32", ".f32", ".b64"};
        for (int t = 1; t < 5; ++t)
            if (count_[t]) out << "\t.reg " << kDecl[t] << " \t" << kPrefix[t] << "<" << count_[t] << ">;\n";
        out << "\n" << body_.str() << "}\n";
        return out.str();
    }

private:
    static constexpr const char* kPrefix[] = {"", "%p", "%r", "%f", "%rd"};

    const IrFunction& fn_;
    int index_;
    std::ostringstream body_;
    std::map<const Inst*, std::string> reg_;
    std::map<const Inst*, std::string> mailbox_;
    int count_[5] = {};  // registers used, per IrType

    std::string newReg(IrType t) {
        int i = static_cast<int>(t);
        return kPrefix[i] + std::to_string(count_[i]++);
    }

    const std::string& r(const Inst* v) const { return reg_.at(v); }

    // Same style of label as nvcc: $L__BB<function>_<block>
    std::string label(const Block* b) const {
        return "$L__BB" + std::to_string(index_) + "_" + std::to_string(b->id);
    }

    std::string paramName(int i) const { return fn_.name + "_param_" + std::to_string(i); }

    // Writes one instruction, with a tab between opcode and operands (like nvcc's output).
    void ins(const std::string& text) {
        size_t opcodeStart = text[0] == '@' ? text.find(' ') + 1 : 0;  // skip a "@%p0 " guard
        size_t split = text.find(' ', opcodeStart);
        if (split == std::string::npos)
            body_ << "\t" << text << ";\n";
        else
            body_ << "\t" << text.substr(0, split) << " \t" << text.substr(split + 1) << ";\n";
    }

    // "op d, a, b" for an instruction with two operands.
    void binary(const std::string& op, const Inst* in) {
        ins(op + " " + r(in) + ", " + r(in->ops[0]) + ", " + r(in->ops[1]));
    }

    void inst(const Inst* in) {
        const bool isF32 = in->type == IrType::F32;
        const bool isPred = in->type == IrType::Pred;
        switch (in->op) {
            case Opc::Const:
                if (isF32)
                    ins("mov.f32 " + r(in) + ", " + f32Hex(in->fimm));
                else
                    ins("mov." + sfx(in->type) + " " + r(in) + ", " + std::to_string(in->imm));
                break;
            case Opc::Param:
                if (in->type == IrType::U64) {
                    // Pointer arguments are generic addresses; ld.global/st.global
                    // need global-memory addresses, so convert them (like nvcc does).
                    std::string tmp = newReg(IrType::U64);
                    ins("ld.param.u64 " + tmp + ", [" + paramName(in->paramIndex) + "]");
                    ins("cvta.to.global.u64 " + r(in) + ", " + tmp);
                } else {
                    ins(std::string("ld.param.") + (isF32 ? "f32 " : "u32 ") + r(in) + ", [" +
                        paramName(in->paramIndex) + "]");
                }
                break;
            case Opc::SReg: ins("mov.u32 " + r(in) + ", %" + in->name); break;
            case Opc::Add: binary("add." + sfx(in->type), in); break;
            case Opc::Sub: binary("sub." + sfx(in->type), in); break;
            // mul.lo keeps the low 32 bits of the 64-bit product (normal int multiply).
            case Opc::Mul: binary(isF32 ? "mul.f32" : "mul.lo.s32", in); break;
            // div.rn = IEEE round-to-nearest division, same as nvcc's default.
            case Opc::Div: binary(isF32 ? "div.rn.f32" : "div.s32", in); break;
            case Opc::Rem: binary("rem.s32", in); break;
            case Opc::And: binary(isPred ? "and.pred" : "and.b32", in); break;
            case Opc::Or: binary(isPred ? "or.pred" : "or.b32", in); break;
            case Opc::Xor: binary(isPred ? "xor.pred" : "xor.b32", in); break;
            case Opc::Shl: binary("shl.b32", in); break;
            case Opc::Shr: binary("shr.s32", in); break;  // arithmetic shift, like >> on int in C
            case Opc::Neg: ins("neg." + sfx(in->type) + " " + r(in) + ", " + r(in->ops[0])); break;
            case Opc::Not: ins(std::string(isPred ? "not.pred " : "not.b32 ") + r(in) + ", " + r(in->ops[0])); break;
            case Opc::Lt:
            case Opc::Le:
            case Opc::Gt:
            case Opc::Ge:
            case Opc::Eq:
            case Opc::Ne: compare(in); break;
            case Opc::CvtI2F: ins("cvt.rn.f32.s32 " + r(in) + ", " + r(in->ops[0])); break;
            // .rzi rounds toward zero, which is what C does for (int)x.
            case Opc::CvtF2I: ins("cvt.rzi.s32.f32 " + r(in) + ", " + r(in->ops[0])); break;
            case Opc::ElemAddr: {
                // addr = base + index * 4   (mul.wide gives a 64-bit result from 32-bit inputs)
                std::string offset = newReg(IrType::U64);
                ins("mul.wide.s32 " + offset + ", " + r(in->ops[1]) + ", 4");
                ins("add.s64 " + r(in) + ", " + r(in->ops[0]) + ", " + offset);
                break;
            }
            case Opc::Load:
                ins(std::string("ld.global.") + (isF32 ? "f32 " : "u32 ") + r(in) + ", [" + r(in->ops[0]) + "]");
                break;
            case Opc::Store:
                ins(std::string("st.global.") + (in->memType == IrType::F32 ? "f32" : "u32") + " [" +
                    r(in->ops[0]) + "], " + r(in->ops[1]));
                break;
            case Opc::Phi:
            case Opc::Br:
            case Opc::CondBr:
            case Opc::Ret: break;  // handled in run()
        }
    }

    void compare(const Inst* in) {
        const Inst* a = in->ops[0];
        const Inst* b = in->ops[1];
        if (a->type == IrType::Pred) {  // bool == bool, bool != bool
            if (in->op == Opc::Ne) {
                binary("xor.pred", in);
            } else {
                std::string tmp = newReg(IrType::Pred);
                ins("xor.pred " + tmp + ", " + r(a) + ", " + r(b));
                ins("not.pred " + r(in) + ", " + tmp);
            }
            return;
        }
        std::string cmp;
        switch (in->op) {
            case Opc::Lt: cmp = "lt"; break;
            case Opc::Le: cmp = "le"; break;
            case Opc::Gt: cmp = "gt"; break;
            case Opc::Ge: cmp = "ge"; break;
            case Opc::Eq: cmp = "eq"; break;
            // For floats, != must be true when an operand is NaN (as in C): "neu" = not-equal-or-unordered.
            default: cmp = a->type == IrType::F32 ? "neu" : "ne"; break;
        }
        binary("setp." + cmp + "." + sfx(a->type), in);
    }

    // Before leaving block 'from', fill the mailbox of every phi in the blocks we may jump to.
    void phiMoves(const Block* from) {
        for (const Block* succ : from->terminator()->targets) {
            size_t predIndex = std::find(succ->preds.begin(), succ->preds.end(), from) - succ->preds.begin();
            for (const Inst* phi : succ->insts) {
                if (phi->op != Opc::Phi) break;
                ins("mov." + sfx(phi->type) + " " + mailbox_.at(phi) + ", " + r(phi->ops[predIndex]));
            }
        }
    }

    // A jump to the block that comes next in the layout is left out (fall-through).
    void terminator(const Inst* t, const Block* next) {
        switch (t->op) {
            case Opc::Br:
                // .uni tells ptxas every thread takes this jump (no divergence).
                if (t->targets[0] != next) ins("bra.uni " + label(t->targets[0]));
                break;
            case Opc::CondBr: {
                const std::string& c = r(t->ops[0]);
                const Block* yes = t->targets[0];
                const Block* no = t->targets[1];
                if (no == next) {
                    ins("@" + c + " bra " + label(yes));
                } else if (yes == next) {
                    ins("@!" + c + " bra " + label(no));
                } else {
                    ins("@" + c + " bra " + label(yes));
                    ins("bra.uni " + label(no));
                }
                break;
            }
            default: ins("ret"); break;
        }
    }
};

}  // namespace

std::string emitPtx(const IrModule& m, const std::string& arch) {
    std::ostringstream out;
    out << "//\n// Generated by kc (a small kernel compiler)\n//\n\n";
    // PTX ISA 8.0 shipped with CUDA 12.0, so any CUDA 12+ driver can load this.
    out << ".version 8.0\n";
    out << ".target " << arch << "\n";
    out << ".address_size 64\n";
    for (size_t i = 0; i < m.funcs.size(); ++i) out << "\n" << FunctionEmitter(*m.funcs[i], static_cast<int>(i)).run();
    return out.str();
}
