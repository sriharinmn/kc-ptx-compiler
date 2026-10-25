// The pass pipeline: runs the passes in order, checks the IR after each one
// with the verifier, and counts what each pass changed for the report.

#include <cstdio>
#include <cstdlib>

#include "opt.h"
#include "verify.h"

namespace {

struct FunctionPass {
    const char* name;
    int (*run)(IrFunction&);
    const char* what;  // for the report: "<count> <what>"
};

const FunctionPass kPasses[] = {
    {"simplify", simplify, "instructions or branches simplified (constants folded, x+0, ...)"},
    {"cse", eliminateCommonSubexpressions, "repeated computations reused"},
    {"licm", hoistLoopInvariants, "instructions moved out of loops"},
    {"dce", removeDeadCode, "unused instructions deleted"},
};
constexpr int kNumPasses = sizeof kPasses / sizeof kPasses[0];

// One pass can create work for another (folding a branch makes code dead,
// CSE makes a value loop-invariant, ...), so the passes repeat until nothing
// changes. Usually 2-3 rounds; the limit only guards against a bug looping forever.
constexpr int kMaxRounds = 10;

// Debugging aid: KC_SKIP_PASSES=licm,unroll turns those passes off. When a
// program goes wrong at -O1, skipping passes one by one shows which one is to blame.
bool skipped(const std::string& pass) {
    const char* list = std::getenv("KC_SKIP_PASSES");
    return list && ("," + std::string(list) + ",").find("," + pass + ",") != std::string::npos;
}

int totalInsts(const IrModule& m) {
    int n = 0;
    for (const auto& f : m.funcs) n += instCount(*f);
    return n;
}

std::string line(const char* name, int count, const char* what) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "%-10s %4d  %s\n", name, count, what);
    return buf;
}

}  // namespace

std::string optimize(IrModule& m, int level) {
    if (level <= 0) return "";
    int sizeBefore = totalInsts(m);

    int inlined = skipped("inline") ? 0 : inlineCalls(m);
    verifyModule(m, "inline");

    int counts[kNumPasses] = {};
    auto rounds = [&](IrFunction& f) {
        for (int round = 0; round < kMaxRounds; ++round) {
            int changes = 0;
            for (int p = 0; p < kNumPasses; ++p) {
                if (skipped(kPasses[p].name)) continue;
                int n = kPasses[p].run(f);
                verifyFunction(f, kPasses[p].name);
                counts[p] += n;
                changes += n;
            }
            if (changes == 0) break;
        }
    };
    int unrolled = 0;
    for (const auto& f : m.funcs) {
        if (f->isExtern) continue;  // defined in another module: nothing to optimize here
        rounds(*f);
        // Unrolling comes after the first rounds, so loops are already clean (invariants
        // moved out), and the copies it makes get cleaned up by one more set of rounds.
        int n = skipped("unroll") ? 0 : unrollLoops(*f);
        verifyFunction(*f, "unroll");
        if (n) rounds(*f);
        unrolled += n;
        f->renumber();
    }

    int removedFns = removeUncalledFunctions(m);
    verifyModule(m, "cleanup");

    std::string report = line("inline", inlined, "calls inlined");
    for (int p = 0; p < kNumPasses; ++p) report += line(kPasses[p].name, counts[p], kPasses[p].what);
    report += line("unroll", unrolled, "loops unrolled 4x");
    report += line("cleanup", removedFns, "device functions deleted (no calls left)");
    report += "IR size: " + std::to_string(sizeBefore) + " -> " + std::to_string(totalInsts(m)) + " instructions\n";
    return report;
}
