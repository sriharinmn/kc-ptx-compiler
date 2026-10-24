#pragma once

// Stage 5: the PTX backend. Turns SSA IR into PTX assembly text.
// PTX has unlimited virtual registers, so there is no register allocation here:
// every IR value simply gets its own register. ptxas (NVIDIA's PTX assembler,
// also built into the driver) does the real register allocation for the GPU.

#include <string>

#include "ir.h"

std::string emitPtx(const IrModule& m, const std::string& arch);
