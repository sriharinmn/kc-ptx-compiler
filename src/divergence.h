#pragma once

// Divergence analysis: which values can be different in different threads
// ("divergent"), and which code only some of the threads of a block run.
//
//   - Divergence starts at threadIdx and lane_id() (and at results that differ per
//     thread: atomics, warp operations, calls, and a device function's parameters).
//   - It spreads to every value computed from a divergent value.
//   - A branch on a divergent value splits the threads. Until the two paths meet
//     again (at the branch's closest post-dominator), only some threads run each
//     block on the way.
//   - Where the split paths meet, a phi is divergent even if all its inputs are
//     uniform: threads arrive from different sides. The same holds for the values
//     carried around a loop that threads leave after different numbers of rounds
//     (e.g. the loop counter of "while (k < threadIdx.x)").
// Everything else is "uniform": the same in every thread (kernel parameters,
// blockIdx, blockDim, constants, and values computed only from those).
//
// Used for warnings (barrier() and warp operations need every thread), and to
// mark uniform branches as bra.uni in the PTX.

#include <map>
#include <set>
#include <string>
#include <vector>

#include "ir.h"

struct DivergenceInfo {
    std::set<const Inst*> values;                  // divergent values
    std::map<const Block*, const Inst*> partial;   // block run by only some threads -> the branch that splits them

    bool divergent(const Inst* v) const { return values.count(v) > 0; }
};

DivergenceInfo analyzeDivergence(const IrFunction& f);

// "L:C: warning: ..." for each barrier() or warp operation that only some threads may reach.
std::vector<std::string> divergenceWarnings(const IrModule& m);
