// kc-fuzz: random program testing ("differential testing").
//   kc-fuzz [programs=200] [first seed=1]
//
// Makes random kc kernels, runs each one twice on the GPU with the same input:
// compiled at -O0 and at -O1. The outputs must be identical: the optimizer may
// make code faster but must never change what it computes. A difference (or an
// internal compiler error from the verifier) is saved as fuzz_fail_<seed>.kc.
//
// The programs only use i32 math, where every result is exactly defined (floats
// could legitimately differ by FMA rounding). Loops have small bounds, divisors
// are never 0 or -1, and every array index is kept in range, so each program is
// valid and terminates.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "compiler.h"
#include "diag.h"
#include "gpu.h"
#include "verify.h"

namespace {

class Generator {
public:
    explicit Generator(unsigned seed) : rng_(seed) {}

    std::string program() {
        out_ << "kernel fuzz(x: *i32, out: *i32, n: i32) {\n"
                "    let i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    if (i >= n) {\n        return;\n    }\n";
        declare("x[i]", 1);
        declare("i", 1);
        statements(4 + pick(6), 1, false);
        out_ << "    out[i] = ";
        for (size_t k = 0; k < vars_.size(); ++k) out_ << (k ? " ^ (" : "(") << vars_[k].name << " * " << 2 * k + 1 << ")";
        out_ << ";\n}\n";
        return out_.str();
    }

private:
    struct Var {
        std::string name;
        bool writable;  // loop counters are read-only, so every loop ends
    };
    std::mt19937 rng_;
    std::ostringstream out_;
    std::vector<Var> vars_;
    int nextName_ = 0;

    int pick(int n) { return std::uniform_int_distribution<int>(0, n - 1)(rng_); }
    bool chance(int percent) { return pick(100) < percent; }
    void indent(int level) { out_ << std::string(4 * level, ' '); }

    void declare(const std::string& value, int level) {
        std::string name = "v" + std::to_string(nextName_++);
        indent(level);
        out_ << "let " << name << " = " << value << ";\n";
        vars_.push_back({name, true});
    }

    std::string constant() {
        static const char* special[] = {"0", "1", "2", "7", "31", "32", "255", "1000", "65536",
                                        "2147483647", "(-2147483647 - 1)", "(-1)", "(-8)"};
        if (chance(40)) return special[pick(13)];
        return std::to_string(pick(200) - 100);
    }

    std::string variable() { return vars_[pick(static_cast<int>(vars_.size()))].name; }

    std::string expr(int depth) {
        if (depth == 0 || chance(25)) return chance(70) ? variable() : constant();
        std::string a = expr(depth - 1), b = expr(depth - 1);
        static const char* divisors[] = {"1", "2", "3", "4", "7", "8", "16", "64", "(-2)", "(-3)", "(-8)", "10"};
        switch (pick(10)) {
            case 0: case 1: {
                static const char* ops[] = {" + ", " - ", " * ", " & ", " | ", " ^ "};
                return "(" + a + ops[pick(6)] + b + ")";
            }
            case 2: return "(" + a + (chance(50) ? " << (" : " >> (") + b + " & 31))";
            case 3: return "(" + a + (chance(50) ? " / " : " % ") + divisors[pick(12)] + ")";
            case 4: return "(" + a + (chance(50) ? " / ((" : " % ((") + b + " & 15) + 1))";  // divisor 1..16
            case 5: return std::string(chance(50) ? "min(" : "max(") + a + ", " + b + ")";
            case 6: return "abs(" + a + ")";
            case 7: return "(-(" + a + "))";
            case 8: return "x[((" + a + ") & 1023) % n]";  // always in range: 0 <= index < n
            default: return "(" + a + " + " + constant() + ")";
        }
    }

    std::string condition(int depth) {
        static const char* cmp[] = {" < ", " <= ", " > ", " >= ", " == ", " != "};
        std::string c = "(" + expr(depth) + cmp[pick(6)] + expr(depth) + ")";
        if (depth > 0 && chance(30)) c = "(" + c + (chance(50) ? " && " : " || ") + condition(depth - 1) + ")";
        if (chance(15)) c = "!" + c;
        return c;
    }

    void statements(int count, int level, bool inLoop) {
        for (int s = 0; s < count; ++s) statement(level, inLoop);
    }

    void statement(int level, bool inLoop) {
        size_t scope = vars_.size();  // variables declared inside a block disappear at its end
        int kind = level > 3 ? pick(3) : pick(7);
        switch (kind) {
            case 0: declare(expr(3), level); return;
            case 1:
            case 2: {
                std::vector<const Var*> writable;
                for (const Var& v : vars_)
                    if (v.writable) writable.push_back(&v);
                static const char* ops[] = {" = ", " += ", " -= ", " *= "};
                indent(level);
                out_ << writable[pick(static_cast<int>(writable.size()))]->name << ops[pick(4)] << expr(3) << ";\n";
                return;
            }
            case 3:
            case 4: {
                indent(level);
                out_ << "if (" << condition(2) << ") {\n";
                statements(1 + pick(3), level + 1, inLoop);
                vars_.resize(scope);
                indent(level);
                if (chance(50)) {
                    out_ << "} else {\n";
                    statements(1 + pick(3), level + 1, inLoop);
                    vars_.resize(scope);
                    indent(level);
                }
                out_ << "}\n";
                return;
            }
            case 5: {  // a for loop with at most 7 rounds
                std::string k = "k" + std::to_string(nextName_++);
                indent(level);
                out_ << "for (let " << k << " = 0; " << k << " < (" << expr(2) << " & 7); " << k << " += 1) {\n";
                vars_.push_back({k, false});
                statements(1 + pick(3), level + 1, true);
                vars_.resize(scope);
                indent(level);
                out_ << "}\n";
                return;
            }
            default: {
                if (inLoop && chance(50)) {  // leave the loop early, or skip to the next round
                    indent(level);
                    out_ << "if (" << condition(1) << ") {\n";
                    indent(level + 1);
                    out_ << (chance(50) ? "break;\n" : "continue;\n");
                    indent(level);
                    out_ << "}\n";
                    return;
                }
                // A while loop with at most 7 rounds. The counter goes down first thing in the
                // body, so a "continue" further down cannot skip it (that would loop forever).
                std::string k = "w" + std::to_string(nextName_++);
                indent(level);
                out_ << "let " << k << " = " << expr(2) << " & 7;\n";
                indent(level);
                out_ << "while (" << k << " > 0) {\n";
                indent(level + 1);
                out_ << k << " -= 1;\n";
                vars_.push_back({k, false});
                statements(1 + pick(3), level + 1, true);
                vars_.resize(scope);
                indent(level);
                out_ << "}\n";
                return;
            }
        }
    }
};

std::vector<int> runOnGpu(const std::string& ptx, const std::vector<int>& input, int ptxasOptLevel = 4) {
    Module m(ptx, ptxasOptLevel);
    DeviceBuffer<int> dx(input), dout(input.size());
    int n = static_cast<int>(input.size());
    launch(m.kernel("fuzz"), (n + 127) / 128, 128, {dx.arg(), dout.arg(), &n});
    return dout.download();
}

// When -O0 and -O1 differ, whose bug is it? Run both PTX files again with ptxas'
// optimizer turned off. If kc's two PTX versions now agree, they mean the same thing:
// kc is right, and the difference was made by ptxas' optimizer.
bool blameOnPtxas(const CompileResult& r0, const CompileResult& r1, const std::vector<int>& input) {
    return runOnGpu(r0.ptx, input, 0) == runOnGpu(r1.ptx, input, 0);
}

void save(const std::string& source, unsigned seed, const std::string& why) {
    std::string file = "fuzz_fail_" + std::to_string(seed) + ".kc";
    std::ofstream(file) << "// " << why << "\n" << source;
    std::printf("  seed %u: %s (saved as %s)\n", seed, why.c_str(), file.c_str());
}

}  // namespace

// The same input for every program: random values plus the corner cases.
std::vector<int> fuzzInput() {
    std::mt19937 rng(12345);
    std::vector<int> input(1000);
    for (int& v : input) v = static_cast<int>(rng());
    input[0] = 0, input[1] = -1, input[2] = 2147483647, input[3] = -2147483647 - 1;
    return input;
}

// kc-fuzz --check file.kc: run one saved program at -O0 and -O1 and compare
// (used to shrink a failing program by hand; KC_SKIP_PASSES narrows down the pass).
int check(Gpu& gpu, const char* path) {
    std::ifstream in(path);
    std::stringstream source;
    source << in.rdbuf();
    CompileOptions o0, o1;
    o0.arch = o1.arch = gpu.arch();
    o0.optLevel = 0;
    std::vector<int> input = fuzzInput();
    CompileResult r0 = compile(source.str(), o0), r1 = compile(source.str(), o1);
    std::vector<int> out0 = runOnGpu(r0.ptx, input), out1 = runOnGpu(r1.ptx, input);
    for (size_t k = 0; k < out0.size(); ++k) {
        if (out0[k] != out1[k]) {
            std::printf("DIFFER at element %zu: -O0 gives %d, -O1 gives %d (%s)\n", k, out0[k], out1[k],
                        blameOnPtxas(r0, r1, input) ? "they agree with ptxas' optimizer off: suspect ptxas"
                                                    : "they differ even with ptxas' optimizer off: a kc bug");
            return 1;
        }
    }
    std::printf("same (element 0 is %d)\n", out0[0]);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--check") {
        Gpu gpu;
        return check(gpu, argv[2]);
    }
    int programs = argc > 1 ? std::atoi(argv[1]) : 200;
    unsigned firstSeed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 1;
    try {
        Gpu gpu;
        std::vector<int> input = fuzzInput();

        int failures = 0, ptxasSuspects = 0;
        long sizeBefore = 0, sizeAfter = 0;
        for (int p = 0; p < programs; ++p) {
            unsigned seed = firstSeed + static_cast<unsigned>(p);
            std::string source = Generator(seed).program();
            CompileOptions o0, o1;
            o0.arch = o1.arch = gpu.arch();
            o0.optLevel = 0;
            CompileResult r0, r1;
            try {
                r0 = compile(source, o0);
                r1 = compile(source, o1);
            } catch (const CompileError& e) {
                save(source, seed, std::string("the generator made an invalid program: ") + e.what());
                return 2;
            } catch (const InternalError& e) {
                save(source, seed, e.what());
                failures++;
                continue;
            }
            int before = 0, after = 0;
            if (std::sscanf(r1.passes.c_str() + r1.passes.find("IR size:"), "IR size: %d -> %d", &before, &after) == 2) {
                sizeBefore += before;
                sizeAfter += after;
            }
            std::vector<int> out0 = runOnGpu(r0.ptx, input), out1 = runOnGpu(r1.ptx, input);
            for (size_t k = 0; k < out0.size(); ++k) {
                if (out0[k] == out1[k]) continue;
                std::string what = "-O0 and -O1 differ at element " + std::to_string(k) + ": " +
                                   std::to_string(out0[k]) + " vs " + std::to_string(out1[k]);
                if (blameOnPtxas(r0, r1, input)) {
                    save(source, seed, what + "; with ptxas' optimizer off they agree: suspect ptxas, not kc");
                    ptxasSuspects++;
                } else {
                    save(source, seed, what + "; they differ even with ptxas' optimizer off: a kc bug");
                    failures++;
                }
                break;
            }
        }
        std::printf("kc-fuzz: %d random programs (seeds %u-%u): %d kc bugs, %d suspected ptxas bugs; the optimizer "
                    "shrank them from %ld to %ld IR instructions\n",
                    programs, firstSeed, firstSeed + programs - 1, failures, ptxasSuspects, sizeBefore, sizeAfter);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("kc-fuzz: %s\n", e.what());
        return 1;
    }
}
