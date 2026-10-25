#include "ptx_emit.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

#include "divergence.h"

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

// Device functions pass every argument and the return value through .param
// memory as plain bits: .b32 for i32/f32/bool, .b64 for pointers. (A bool
// travels as a 32-bit 0 or 1, because .param memory cannot hold predicates.)
const char* paramBits(IrType t) { return t == IrType::U64 ? ".b64" : ".b32"; }

std::string retvalName(const IrFunction& f) { return f.name + "_retval"; }
std::string paramName(const IrFunction& f, int i) { return f.name + "_param_" + std::to_string(i); }

// ".visible .entry k(...)" for kernels, ".func (.param .b32 f_retval) f(...)" for device functions.
// The first word is the linkage, which the linker reads:
//   .func           private to this module
//   .visible .func  defined here, other modules may call it ("export fn")
//   .extern .func   defined in another module ("extern fn"); the linker connects the calls
std::string signature(const IrFunction& f) {
    std::string s;
    if (f.isKernel) {
        s = ".visible .entry " + f.name + "(";
    } else {
        s = f.isExtern ? ".extern .func " : f.isExport ? ".visible .func " : ".func ";
        if (f.retType != IrType::Void) s += std::string("(.param ") + paramBits(f.retType) + " " + retvalName(f) + ") ";
        s += f.name + "(";
    }
    for (size_t i = 0; i < f.params.size(); ++i) {
        IrType t = f.params[i]->type;
        const char* ptxType = !f.isKernel ? paramBits(t) : t == IrType::F32 ? ".f32" : t == IrType::U64 ? ".u64" : ".u32";
        s += std::string(i ? ",\n" : "\n") + "\t.param " + ptxType + " " + paramName(f, static_cast<int>(i));
    }
    return s + "\n)";
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

// "sm_89" -> 89. Decides which instructions the target GPU has.
int smNumber(const std::string& arch) {
    return arch.rfind("sm_", 0) == 0 ? std::atoi(arch.c_str() + 3) : 0;
}

// Warp instructions name the threads that take part (the "member mask"): all 32 of them.
const char* const kFullWarp = "0xffffffff";

class FunctionEmitter {
public:
    FunctionEmitter(const IrFunction& fn, int index, int sm, bool lineInfo)
        : fn_(fn), index_(index), sm_(sm), lineInfo_(lineInfo), divergence_(analyzeDivergence(fn)) {}

    std::string run() {
        // 1. Give every value a register (and every phi a mailbox).
        for (const auto& b : fn_.blocks) {
            for (const Inst* in : b->insts) {
                if (in->type == IrType::Void || isImmediate(in)) continue;
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
                    sourceLine(in);
                    phiMoves(b);
                    terminator(in, next);
                } else if (!isImmediate(in)) {
                    sourceLine(in);
                    inst(in);
                }
            }
        }

        // 3. Signature and register declarations (the counts are only known now).
        std::ostringstream out;
        out << signature(fn_) << "\n{\n";
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
    int sm_;  // target architecture number, e.g. 89
    bool lineInfo_;
    int lastLine_ = 0;  // source line of the last .loc written
    DivergenceInfo divergence_;
    std::ostringstream body_;
    std::map<const Inst*, std::string> reg_;
    std::map<const Inst*, std::string> mailbox_;
    int count_[5] = {};  // registers used, per IrType

    std::string newReg(IrType t) {
        int i = static_cast<int>(t);
        return kPrefix[i] + std::to_string(count_[i]++);
    }

    // PTX accepts i32 and f32 constants directly as operands ("add.s32 %r2, %r1, 1"),
    // so they need no register. Bool constants still do: a predicate operand must be a register.
    static bool isImmediate(const Inst* v) { return v->op == Opc::Const && v->type != IrType::Pred; }

    // How an operand is written: its register, or the constant itself.
    std::string r(const Inst* v) const {
        if (isImmediate(v)) return v->type == IrType::F32 ? f32Hex(v->fimm) : std::to_string(v->imm);
        return reg_.at(v);
    }

    // Same style of label as nvcc: $L__BB<function>_<block>
    std::string label(const Block* b) const {
        return "$L__BB" + std::to_string(index_) + "_" + std::to_string(b->id);
    }


    // Line info: ".loc 1 <line> <column>" says the instructions after it come from that place in
    // file 1 (declared with .file). It holds until the next .loc, so one is written only when the line changes.
    void sourceLine(const Inst* in) {
        if (!lineInfo_ || in->loc.line == 0 || in->loc.line == lastLine_) return;
        body_ << "\t.loc 1 " << in->loc.line << " " << in->loc.col << "\n";
        lastLine_ = in->loc.line;
    }

    // Writes one instruction, with a tab between opcode and operands (like nvcc's output).
    // Declarations such as ".param .b32 param0" are written as they are.
    void ins(const std::string& text) {
        size_t opcodeStart = text[0] == '@' ? text.find(' ') + 1 : 0;  // skip a "@%p0 " guard
        size_t split = text.find(' ', opcodeStart);
        if (split == std::string::npos || text[0] == '.')
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
            case Opc::Const:  // only bools get here: other constants are written into their users
                ins("mov.pred " + r(in) + ", " + std::to_string(in->imm));
                break;
            case Opc::Param: {
                std::string p = "[" + paramName(fn_, in->paramIndex) + "]";
                if (in->type == IrType::U64 && fn_.isKernel) {
                    // Kernel pointer arguments are generic addresses; ld.global/st.global
                    // need global-memory addresses, so convert them (like nvcc does).
                    // Device functions receive pointers that are already converted.
                    std::string tmp = newReg(IrType::U64);
                    ins("ld.param.u64 " + tmp + ", " + p);
                    ins("cvta.to.global.u64 " + r(in) + ", " + tmp);
                } else {
                    readParam(r(in), in->type, p);
                }
                break;
            }
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
            case Opc::Min: binary("min." + sfx(in->type), in); break;
            case Opc::Max: binary("max." + sfx(in->type), in); break;
            case Opc::Abs: ins("abs." + sfx(in->type) + " " + r(in) + ", " + r(in->ops[0])); break;
            // sqrt.rn = IEEE-correct square root (nvcc's default too).
            case Opc::Sqrt: ins("sqrt.rn.f32 " + r(in) + ", " + r(in->ops[0])); break;
            case Opc::Popc: ins("popc.b32 " + r(in) + ", " + r(in->ops[0])); break;
            case Opc::WarpSum: warpSum(in); break;
            case Opc::ShflDown: shuffle("down", r(in), r(in->ops[0]), r(in->ops[1])); break;
            case Opc::ShflXor: shuffle("bfly", r(in), r(in->ops[0]), r(in->ops[1])); break;  // "butterfly" = XOR
            case Opc::Ballot: ins("vote.sync.ballot.b32 " + r(in) + ", " + r(in->ops[0]) + ", " + kFullWarp); break;
            case Opc::ElemAddr: {
                // addr = base + index * 4   (mul.wide gives a 64-bit result from 32-bit inputs)
                if (isImmediate(in->ops[1])) {  // constant index: the byte offset is known now
                    ins("add.s64 " + r(in) + ", " + r(in->ops[0]) + ", " + std::to_string(in->ops[1]->imm * 4));
                    break;
                }
                std::string offset = newReg(IrType::U64);
                ins("mul.wide.s32 " + offset + ", " + r(in->ops[1]) + ", 4");
                ins("add.s64 " + r(in) + ", " + r(in->ops[0]) + ", " + offset);
                break;
            }
            // The address of a .shared variable is its position in the block's shared memory.
            case Opc::SharedAddr: ins("mov.u64 " + r(in) + ", " + in->name); break;
            case Opc::Load:
                ins("ld" + space(in) + (isF32 ? ".f32 " : ".u32 ") + r(in) + ", [" + r(in->ops[0]) + "]");
                break;
            case Opc::Store:
                ins("st" + space(in) + (in->memType == IrType::F32 ? ".f32" : ".u32") + " [" + r(in->ops[0]) +
                    "], " + r(in->ops[1]));
                break;
            case Opc::AtomicAdd:
                ins("atom" + space(in) + ".add." + sfx(in->type) + " " + r(in) + ", [" + r(in->ops[0]) + "], " +
                    r(in->ops[1]));
                break;
            // Barrier 0 is the one __syncthreads() uses.
            case Opc::Barrier: ins("bar.sync 0"); break;
            case Opc::Call: call(in); break;
            case Opc::Phi:
            case Opc::Br:
            case Opc::CondBr:
            case Opc::Ret: break;  // handled in run()
        }
    }

    static std::string space(const Inst* in) { return in->shared ? ".shared" : ".global"; }

    // shfl.sync.<mode>.b32 d, value, lane, 0x1f, mask. 0x1f says the whole warp is one group.
    // .b32 means "any 32 bits", so the same instruction moves i32 and f32 values.
    void shuffle(const std::string& mode, const std::string& dst, const std::string& value, const std::string& lane) {
        ins("shfl.sync." + mode + ".b32 " + dst + ", " + value + ", " + lane + ", 0x1f, " + kFullWarp);
    }

    // warp_sum, the same source code with a different lowering per GPU generation:
    //   sm_80 and newer, i32: one instruction, redux.sync (the hardware adds across the warp)
    //   older GPUs, or f32:   5 butterfly shuffles. After adding the value from 16, 8, 4, 2
    //                         and 1 lanes away, every thread holds the sum of all 32.
    void warpSum(const Inst* in) {
        if (in->type == IrType::S32 && sm_ >= 80) {
            ins("redux.sync.add.s32 " + r(in) + ", " + r(in->ops[0]) + ", " + kFullWarp);
            return;
        }
        std::string sum = r(in->ops[0]);
        for (int distance = 16; distance >= 1; distance /= 2) {
            std::string other = newReg(in->type);
            std::string next = distance == 1 ? r(in) : newReg(in->type);
            shuffle("bfly", other, sum, std::to_string(distance));
            ins("add." + sfx(in->type) + " " + next + ", " + sum + ", " + other);
            sum = next;
        }
    }

    // Reads a .param slot into a register. A bool arrives as a 32-bit 0/1 and becomes a predicate.
    void readParam(const std::string& dst, IrType t, const std::string& slot) {
        if (t == IrType::Pred) {
            std::string bits = newReg(IrType::S32);
            ins("ld.param.b32 " + bits + ", " + slot);
            ins("setp.ne.s32 " + dst + ", " + bits + ", 0");
        } else {
            const char* type = t == IrType::F32 ? "f32" : t == IrType::U64 ? "b64" : "b32";
            ins(std::string("ld.param.") + type + " " + dst + ", " + slot);
        }
    }

    // Writes a register into a .param slot (the reverse of readParam).
    void writeParam(const std::string& slot, const Inst* v) {
        if (v->type == IrType::Pred) {
            std::string bits = newReg(IrType::S32);
            ins("selp.b32 " + bits + ", 1, 0, " + r(v));
            ins("st.param.b32 " + slot + ", " + bits);
        } else {
            const char* type = v->type == IrType::F32 ? "f32" : v->type == IrType::U64 ? "b64" : "b32";
            ins(std::string("st.param.") + type + " " + slot + ", " + r(v));
        }
    }

    // The PTX calling convention: arguments and the return value go through
    // .param variables declared in a { } scope around the call.
    //   { .param .b32 param0; st.param.b32 [param0], %r1;
    //     .param .b32 retval0; call (retval0), f, (param0); ld.param.b32 %r2, [retval0]; }
    void call(const Inst* in) {
        body_ << "\t{\n";
        std::string args;
        for (size_t i = 0; i < in->ops.size(); ++i) {
            std::string p = "param" + std::to_string(i);
            ins(std::string(".param ") + paramBits(in->ops[i]->type) + " " + p);
            writeParam("[" + p + "]", in->ops[i]);
            args += (i ? ", " : "") + p;
        }
        std::string target = in->name + (args.empty() ? "" : ", (" + args + ")");
        if (in->type == IrType::Void) {
            ins("call " + target);
        } else {
            ins(std::string(".param ") + paramBits(in->type) + " retval0");
            ins("call (retval0), " + target);
            readParam(r(in), in->type, "[retval0]");
        }
        body_ << "\t}\n";
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
                // The divergence analysis knows when every thread decides the same way:
                // then the branch is marked .uni too.
                std::string bra = divergence_.divergent(t->ops[0]) ? " bra " : " bra.uni ";
                if (no == next) {
                    ins("@" + c + bra + label(yes));
                } else if (yes == next) {
                    ins("@!" + c + bra + label(no));
                } else {
                    ins("@" + c + bra + label(yes));
                    ins("bra.uni " + label(no));
                }
                break;
            }
            default:  // Ret
                if (!t->ops.empty()) writeParam("[" + retvalName(fn_) + "]", t->ops[0]);
                ins("ret");
                break;
        }
    }
};

}  // namespace

std::string emitPtx(const IrModule& m, const std::string& arch, const std::string& sourceFile) {
    std::ostringstream out;
    out << "//\n// Generated by kc (a small kernel compiler)\n//\n\n";
    // PTX ISA 8.0 shipped with CUDA 12.0, so any CUDA 12+ driver can load this.
    out << ".version 8.0\n";
    out << ".target " << arch << "\n";
    out << ".address_size 64\n";
    if (!sourceFile.empty()) out << "\n.file 1 \"" << sourceFile << "\"\n";  // file 1, for .loc

    // PTX needs a function declared before it is called: declare all device
    // functions up front, so they can call each other in any order.
    bool any = false;
    for (const auto& f : m.funcs) {
        if (f->isKernel) continue;
        out << (any ? "" : "\n") << signature(*f) << ";\n";
        any = true;
    }
    // Shared arrays: one .shared variable each, visible to the whole module.
    any = false;
    for (const auto& f : m.funcs) {
        for (const SharedArray& a : f->sharedArrays) {
            out << (any ? "" : "\n") << ".shared .align 4 ." << (a.elem == IrType::F32 ? "f32" : "s32") << " "
                << a.name << "[" << a.size << "];\n";
            any = true;
        }
    }

    for (size_t i = 0; i < m.funcs.size(); ++i)
        if (!m.funcs[i]->isExtern)  // an extern function only gets its declaration above
            out << "\n" << FunctionEmitter(*m.funcs[i], static_cast<int>(i), smNumber(arch), !sourceFile.empty()).run();
    return out.str();
}
