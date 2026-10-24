#pragma once

// The IR (intermediate representation): what the compiler works on between
// the AST and PTX. It is in SSA form:
//   - every value (Inst) is defined exactly once and never changes
//   - code is split into basic blocks; only the last instruction of a block jumps
//   - where control flow merges, a "phi" picks the value from the block we came from
// This is the same idea LLVM uses, kept much smaller.

#include <memory>
#include <string>
#include <vector>

// Types of IR values. They map 1:1 onto PTX register kinds.
enum class IrType {
    Void,  // instruction produces no value (store, branch, ...)
    Pred,  // bool          -> PTX .pred register  (%p)
    S32,   // i32           -> PTX .b32 register   (%r)
    F32,   // f32           -> PTX .f32 register   (%f)
    U64,   // pointer       -> PTX .b64 register   (%rd)
};

enum class Opc {
    Const,     // constant (imm or fimm)
    Param,     // kernel parameter number paramIndex
    SReg,      // special register, e.g. name = "tid.x" for threadIdx.x
    Add, Sub, Mul, Div, Rem,
    And, Or, Xor, Shl, Shr,
    Neg, Not,
    Lt, Le, Gt, Ge, Eq, Ne,  // comparisons: result is Pred
    CvtI2F, CvtF2I,          // i32 -> f32, f32 -> i32
    ElemAddr,  // address of ops[0][ops[1]] (pointer + index * element size)
    Load,      // value at address ops[0]
    Store,     // store ops[1] at address ops[0]
    Phi,       // ops[i] is the value when coming from parent->preds[i]
    Br,        // jump to targets[0]
    CondBr,    // if ops[0] jump to targets[0], else to targets[1]
    Ret,
};

struct Block;

struct Inst {
    int id = 0;
    Opc op = Opc::Const;
    IrType type = IrType::Void;    // type of the result
    std::vector<Inst*> ops;        // operands
    std::vector<Block*> targets;   // Br / CondBr
    Block* parent = nullptr;

    long long imm = 0;             // Const (i32 and bool)
    float fimm = 0;                // Const (f32)
    IrType memType = IrType::Void; // ElemAddr / Load / Store: element type in memory
    int paramIndex = -1;           // Param
    std::string name;              // Param name, or SReg name

    // Set when a phi is removed during SSA construction; points to the value that replaced it.
    Inst* replacement = nullptr;
};

struct Block {
    int id = 0;
    std::vector<Inst*> insts;   // phis first, terminator (Br/CondBr/Ret) last
    std::vector<Block*> preds;  // blocks that jump here
    bool sealed = false;        // used while building SSA: all preds are known

    Inst* terminator() const;
};

struct IrFunction {
    std::string name;
    std::vector<Inst*> params;                  // Param instructions, in order
    std::vector<std::unique_ptr<Block>> blocks; // blocks[0] is the entry; order = layout in PTX
    std::vector<std::unique_ptr<Inst>> pool;    // owns every Inst ever created

    Inst* newInst(Opc op, IrType type);  // creates an Inst but does not place it in a block
    Block* newBlock();                   // creates a block at the end of the layout
    void moveToEnd(Block* b);
    void removeBlock(Block* b);
    void renumber();                     // ids = position, for readable output
};

struct IrModule {
    std::vector<std::unique_ptr<IrFunction>> funcs;
};

bool isTerminator(Opc op);
const char* irTypeName(IrType t);
std::string printIr(const IrModule& m);
