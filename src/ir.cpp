#include "ir.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

bool isTerminator(Opc op) { return op == Opc::Br || op == Opc::CondBr || op == Opc::Ret; }

Inst* Block::terminator() const {
    if (insts.empty() || !isTerminator(insts.back()->op)) return nullptr;
    return insts.back();
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
        case Opc::ElemAddr: return "elemaddr";
        case Opc::Load: return "load";
        case Opc::Store: return "store";
        case Opc::Phi: return "phi";
        case Opc::Br: return "br";
        case Opc::CondBr: return "condbr";
        case Opc::Ret: return "ret";
    }
    return "?";
}

bool isCompare(Opc op) { return op >= Opc::Lt && op <= Opc::Ne; }

std::string val(const Inst* v) { return "%" + std::to_string(v->id); }

std::string printInst(const Inst* in) {
    std::ostringstream os;
    if (in->type != IrType::Void) os << val(in) << " = ";
    os << opcName(in->op);

    // Type suffix: what the instruction operates on (like PTX's add.s32).
    if (in->op == Opc::ElemAddr || in->op == Opc::Load || in->op == Opc::Store)
        os << "." << irTypeName(in->memType);
    else if (isCompare(in->op))
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
        case Opc::Phi:
            for (size_t i = 0; i < in->ops.size(); ++i)
                os << (i ? ", " : " ") << "[bb" << in->parent->preds[i]->id << ": " << val(in->ops[i]) << "]";
            break;
        case Opc::Br: os << " bb" << in->targets[0]->id; break;
        case Opc::CondBr:
            os << " " << val(in->ops[0]) << ", bb" << in->targets[0]->id << ", bb" << in->targets[1]->id;
            break;
        default:
            for (size_t i = 0; i < in->ops.size(); ++i) os << (i ? ", " : " ") << val(in->ops[i]);
            break;
    }
    return os.str();
}

}  // namespace

std::string printIr(const IrModule& m) {
    std::ostringstream os;
    for (const auto& f : m.funcs) {
        os << "kernel " << f->name << "\n";
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
