#include "ir.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

bool isTerminator(Opc op) { return op == Opc::Br || op == Opc::CondBr || op == Opc::Ret; }
bool isCompare(Opc op) { return op >= Opc::Lt && op <= Opc::Ne; }

bool isPure(Opc op) {
    switch (op) {
        case Opc::Const:
        case Opc::SReg:  // %tid.x etc. never change while a thread runs
        case Opc::Add: case Opc::Sub: case Opc::Mul: case Opc::Div: case Opc::Rem:
        case Opc::And: case Opc::Or: case Opc::Xor: case Opc::Shl: case Opc::Shr:
        case Opc::Neg: case Opc::Not:
        case Opc::Lt: case Opc::Le: case Opc::Gt: case Opc::Ge: case Opc::Eq: case Opc::Ne:
        case Opc::CvtI2F: case Opc::CvtF2I:
        case Opc::Min: case Opc::Max: case Opc::Abs: case Opc::Sqrt:
        case Opc::ElemAddr: case Opc::SharedAddr:
            return true;
        default:
            return false;  // Param, Load, Phi, memory writes, calls, jumps
    }
}

bool hasSideEffects(Opc op) {
    return op == Opc::Store || op == Opc::AtomicAdd || op == Opc::Barrier || op == Opc::Call || isTerminator(op);
}

Inst* Block::terminator() const {
    if (insts.empty() || !isTerminator(insts.back()->op)) return nullptr;
    return insts.back();
}

IrFunction* IrModule::find(const std::string& name) const {
    for (const auto& f : funcs)
        if (f->name == name) return f.get();
    return nullptr;
}

std::vector<Block*> successors(const Block* b) {
    Inst* t = b->terminator();
    return t ? t->targets : std::vector<Block*>{};
}

int instCount(const IrFunction& f) {
    int n = 0;
    for (const auto& b : f.blocks) n += static_cast<int>(b->insts.size());
    return n;
}

// There are no use lists, so this scans the whole function. That is fine for
// kernels of a few hundred instructions, and much simpler to keep correct.
void replaceAllUses(IrFunction& f, Inst* from, Inst* to) {
    for (auto& b : f.blocks)
        for (Inst* in : b->insts)
            for (Inst*& op : in->ops)
                if (op == from) op = to;
}

void eraseInst(Inst* in) {
    auto& insts = in->parent->insts;
    insts.erase(std::find(insts.begin(), insts.end(), in));
    in->parent = nullptr;
}

// Phi operand i belongs to preds[i], so both lists must shrink together.
void removeEdge(Block* from, Block* to) {
    size_t i = std::find(to->preds.begin(), to->preds.end(), from) - to->preds.begin();
    to->preds.erase(to->preds.begin() + i);
    for (Inst* in : to->insts)
        if (in->op == Opc::Phi) in->ops.erase(in->ops.begin() + i);
}

Inst* IrFunction::newInst(Opc op, IrType type) {
    pool.push_back(std::make_unique<Inst>());
    Inst* in = pool.back().get();
    in->id = static_cast<int>(pool.size()) - 1;
    in->op = op;
    in->type = type;
    return in;
}

Block* IrFunction::newBlock() {
    blocks.push_back(std::make_unique<Block>());
    blocks.back()->id = static_cast<int>(blocks.size()) - 1;
    return blocks.back().get();
}

void IrFunction::moveToEnd(Block* b) {
    auto it = std::find_if(blocks.begin(), blocks.end(), [&](auto& p) { return p.get() == b; });
    std::rotate(it, it + 1, blocks.end());
}

void IrFunction::removeBlock(Block* b) {
    blocks.erase(std::find_if(blocks.begin(), blocks.end(), [&](auto& p) { return p.get() == b; }));
}

void IrFunction::renumber() {
    int nextInst = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i]->id = static_cast<int>(i);
        for (Inst* in : blocks[i]->insts) in->id = nextInst++;
    }
}

const char* irTypeName(IrType t) {
    switch (t) {
        case IrType::Void: return "void";
        case IrType::Pred: return "pred";
        case IrType::S32: return "s32";
        case IrType::F32: return "f32";
        case IrType::U64: return "u64";
    }
    return "?";
}

namespace {

const char* opcName(Opc op) {
    switch (op) {
        case Opc::Const: return "const";
        case Opc::Param: return "param";
        case Opc::SReg: return "sreg";
        case Opc::Add: return "add";
        case Opc::Sub: return "sub";
        case Opc::Mul: return "mul";
        case Opc::Div: return "div";
        case Opc::Rem: return "rem";
        case Opc::And: return "and";
        case Opc::Or: return "or";
        case Opc::Xor: return "xor";
        case Opc::Shl: return "shl";
        case Opc::Shr: return "shr";
        case Opc::Neg: return "neg";
        case Opc::Not: return "not";
        case Opc::Lt: return "lt";
        case Opc::Le: return "le";
        case Opc::Gt: return "gt";
        case Opc::Ge: return "ge";
        case Opc::Eq: return "eq";
        case Opc::Ne: return "ne";
        case Opc::CvtI2F: return "i2f";
        case Opc::CvtF2I: return "f2i";
        case Opc::Min: return "min";
        case Opc::Max: return "max";
        case Opc::Abs: return "abs";
        case Opc::Sqrt: return "sqrt";
        case Opc::ElemAddr: return "elemaddr";
        case Opc::SharedAddr: return "sharedaddr";
        case Opc::Load: return "load";
        case Opc::Store: return "store";
        case Opc::AtomicAdd: return "atomic_add";
        case Opc::Barrier: return "barrier";
        case Opc::Call: return "call";
        case Opc::Phi: return "phi";
        case Opc::Br: return "br";
        case Opc::CondBr: return "condbr";
        case Opc::Ret: return "ret";
    }
    return "?";
}

std::string val(const Inst* v) { return "%" + std::to_string(v->id); }

}  // namespace

// Also used for verifier messages, so it must not crash on a broken instruction.
std::string printInst(const Inst* in) {
    std::ostringstream os;
    if (in->type != IrType::Void) os << val(in) << " = ";
    os << opcName(in->op);

    // Type suffix: what the instruction operates on (like PTX's add.s32).
    bool memory = in->op == Opc::Load || in->op == Opc::Store || in->op == Opc::AtomicAdd;
    if (memory && in->shared) os << ".shared";
    if (memory || in->op == Opc::ElemAddr)
        os << "." << irTypeName(in->memType);
    else if (isCompare(in->op) && !in->ops.empty())
        os << "." << irTypeName(in->ops[0]->type);
    else if (in->type != IrType::Void)
        os << "." << irTypeName(in->type);

    switch (in->op) {
        case Opc::Const:
            if (in->type == IrType::F32) {
                char buf[32];
                std::snprintf(buf, sizeof buf, "%.9g", in->fimm);
                os << " " << buf;
            } else if (in->type == IrType::Pred) {
                os << (in->imm ? " true" : " false");
            } else {
                os << " " << in->imm;
            }
            break;
        case Opc::Param: os << " " << in->name; break;
        case Opc::SReg: os << " %" << in->name; break;
        case Opc::SharedAddr: os << " " << in->name; break;
        case Opc::Call:
            os << " " << in->name << "(";
            for (size_t i = 0; i < in->ops.size(); ++i) os << (i ? ", " : "") << val(in->ops[i]);
            os << ")";
            break;
        case Opc::Phi:
            for (size_t i = 0; i < in->ops.size(); ++i)
                os << (i ? ", " : " ") << "[bb"
                   << (in->parent && i < in->parent->preds.size() ? std::to_string(in->parent->preds[i]->id) : "?")
                   << ": " << val(in->ops[i]) << "]";
            break;
        case Opc::Br:
        case Opc::CondBr:
            for (const Inst* op : in->ops) os << " " << val(op) << ",";
            for (size_t i = 0; i < in->targets.size(); ++i) os << (i ? ", bb" : " bb") << in->targets[i]->id;
            break;
        default:
            for (size_t i = 0; i < in->ops.size(); ++i) os << (i ? ", " : " ") << val(in->ops[i]);
            break;
    }
    return os.str();
}

std::string printIr(const IrModule& m) {
    std::ostringstream os;
    for (const auto& f : m.funcs) {
        os << (f->isKernel ? "kernel " : "fn ") << f->name;
        if (f->retType != IrType::Void) os << " -> " << irTypeName(f->retType);
        os << "\n";
        for (const SharedArray& a : f->sharedArrays)
            os << "  shared " << a.name << ": " << irTypeName(a.elem) << "[" << a.size << "]\n";
        for (const auto& b : f->blocks) {
            os << "bb" << b->id << ":";
            if (!b->preds.empty()) {
                os << "    ; preds:";
                for (const Block* p : b->preds) os << " bb" << p->id;
            }
            os << "\n";
            for (const Inst* in : b->insts) os << "  " << printInst(in) << "\n";
        }
        os << "\n";
    }
    return os.str();
}
