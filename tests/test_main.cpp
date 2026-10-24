// End-to-end tests: compile .kc programs with kc, run them on the real GPU,
// and compare every output element with the same computation done on the CPU.
// Usage: kc-tests [-v]     (-v prints the JIT log: registers used per kernel)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "compiler.h"
#include "diag.h"
#include "gpu.h"

namespace {

bool gVerbose = false;

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CHECK(cond)                                                                                      \
    do {                                                                                                 \
        if (!(cond)) throw TestFailure("line " + std::to_string(__LINE__) + ": CHECK(" #cond ") failed"); \
    } while (0)

// ---- helpers --------------------------------------------------------------

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string example(const std::string& name) { return readFile(std::string(KC_SOURCE_DIR) + "/examples/" + name); }

// Compile kc source for this GPU and load it (the driver JIT-compiles the PTX).
Module build(Gpu& gpu, const std::string& source) {
    CompileOptions opts;
    opts.arch = gpu.arch();
    Module m(compile(source, opts).ptx);
    if (gVerbose) std::printf("%s", m.jitLog().c_str());
    return m;
}

unsigned blocksFor(int n, unsigned threads) { return (n + threads - 1) / threads; }

// Deterministic pseudo-random data, so a failure can be reproduced.
unsigned nextRandom(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
}

std::vector<float> randomFloats(int n, unsigned seed) {  // values in -10.00 .. 10.00
    std::vector<float> v(n);
    for (auto& x : v) x = static_cast<int>(nextRandom(seed) % 2001 - 1000) / 100.0f;
    return v;
}

std::vector<int> randomInts(int n, unsigned seed, int lo, int hi) {
    std::vector<int> v(n);
    for (auto& x : v) x = lo + static_cast<int>(nextRandom(seed) % static_cast<unsigned>(hi - lo + 1));
    return v;
}

void checkFloats(const std::vector<float>& got, const std::vector<float>& want) {
    CHECK(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        // Small tolerance: ptxas may fuse a*b+c into one fma instruction, which rounds once instead of twice.
        if (std::fabs(got[i] - want[i]) > 1e-5f * std::max(1.0f, std::fabs(want[i])))
            throw TestFailure("element " + std::to_string(i) + ": got " + std::to_string(got[i]) + ", want " +
                              std::to_string(want[i]));
    }
}

void checkInts(const std::vector<int>& got, const std::vector<int>& want) {
    CHECK(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i])
            throw TestFailure("element " + std::to_string(i) + ": got " + std::to_string(got[i]) + ", want " +
                              std::to_string(want[i]));
}

// Compiling 'source' must fail, with an error message that contains 'expected'.
void expectError(const std::string& source, const std::string& expected) {
    try {
        compile(source);
    } catch (const CompileError& e) {
        if (std::string(e.what()).find(expected) == std::string::npos)
            throw TestFailure("wrong error \"" + std::string(e.what()) + "\", expected it to mention \"" + expected + "\"");
        return;
    }
    throw TestFailure("program compiled, but it should fail with: " + expected);
}

// ---- tests ----------------------------------------------------------------

void runVectorAdd(Module& m) {
    const int n = 1000;  // not a multiple of the block size, so the 'i < n' check matters
    std::vector<float> a = randomFloats(n, 1), b = randomFloats(n, 2), want(n);
    for (int i = 0; i < n; ++i) want[i] = a[i] + b[i];
    DeviceBuffer<float> da(a), db(b), dc(n);
    int nArg = n;
    launch(m.kernel("vector_add"), blocksFor(n, 256), 256, {da.arg(), db.arg(), dc.arg(), &nArg});
    checkFloats(dc.download(), want);
}

void testHandwrittenPtx(Gpu&) {
    Module m(readFile(std::string(KC_SOURCE_DIR) + "/tests/ptx/vector_add.ptx"));
    runVectorAdd(m);
}

void testVectorAdd(Gpu& gpu) {
    Module m = build(gpu, example("vector_add.kc"));
    runVectorAdd(m);
}

void testSaxpy(Gpu& gpu) {
    Module m = build(gpu, example("saxpy.kc"));
    const int n = 5000;
    float alpha = 2.5f;
    std::vector<float> x = randomFloats(n, 3), y = randomFloats(n, 4), want(n);
    for (int i = 0; i < n; ++i) want[i] = alpha * x[i] + y[i];
    DeviceBuffer<float> dx(x), dy(y);
    int nArg = n;
    launch(m.kernel("saxpy"), blocksFor(n, 256), 256, {&nArg, &alpha, dx.arg(), dy.arg()});
    checkFloats(dy.download(), want);
}

void testRelu(Gpu& gpu) {
    Module m = build(gpu, example("relu.kc"));
    const int n = 1000;
    std::vector<float> x = randomFloats(n, 5), want(n);
    for (int i = 0; i < n; ++i) want[i] = x[i] < 0 ? 0 : x[i];
    DeviceBuffer<float> dx(x), dy(n);
    int nArg = n;
    launch(m.kernel("relu"), blocksFor(n, 128), 128, {dx.arg(), dy.arg(), &nArg});
    checkFloats(dy.download(), want);
}

void testRowSum(Gpu& gpu) {
    Module m = build(gpu, example("row_sum.kc"));
    int rows = 300, cols = 37;
    std::vector<float> mat = randomFloats(rows * cols, 6), want(rows, 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) want[r] += mat[r * cols + c];
    DeviceBuffer<float> dm(mat), dout(rows);
    launch(m.kernel("row_sum"), blocksFor(rows, 64), 64, {dm.arg(), dout.arg(), &rows, &cols});
    checkFloats(dout.download(), want);
}

void testGridStride(Gpu& gpu) {
    Module m = build(gpu, example("grid_stride.kc"));
    const int n = 100000;  // far more elements than the 4 * 128 threads launched
    std::vector<float> x = randomFloats(n, 7), want(n);
    for (int i = 0; i < n; ++i) want[i] = 2.0f * x[i];
    DeviceBuffer<float> dx(x), dy(n);
    int nArg = n;
    launch(m.kernel("scale2"), 4, 128, {dx.arg(), dy.arg(), &nArg});
    checkFloats(dy.download(), want);
}

void testMatmul(Gpu& gpu) {
    Module m = build(gpu, example("matmul.kc"));
    int n = 48;
    // Small whole numbers, so every float result is exact and must match bit for bit.
    std::vector<int> ai = randomInts(n * n, 8, -3, 3), bi = randomInts(n * n, 9, -3, 3);
    std::vector<float> a(ai.begin(), ai.end()), b(bi.begin(), bi.end()), want(n * n, 0.0f);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            for (int k = 0; k < n; ++k) want[r * n + c] += a[r * n + k] * b[k * n + c];
    DeviceBuffer<float> da(a), db(b), dc(n * n);
    launch(m.kernel("matmul"), blocksFor(n * n, 256), 256, {da.arg(), db.arg(), dc.arg(), &n});
    checkFloats(dc.download(), want);
}

void testIntegerOps(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel int_ops(a: *i32, b: *i32, out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                let x = a[i];
                let y = b[i];
                let o = i * 8;
                out[o] = x + y;
                out[o + 1] = x - y;
                out[o + 2] = x * y;
                out[o + 3] = x / y;
                out[o + 4] = x % y;
                out[o + 5] = (x & y) | (x ^ 255);
                out[o + 6] = (x << (y & 7)) >> 2;
                out[o + 7] = -x;
            }
        }
    )");
    const int n = 2000;
    std::vector<int> a = randomInts(n, 10, -100000, 100000), b = randomInts(n, 11, 1, 50), want(n * 8);
    for (int i = 0; i < n; ++i) {
        if (i % 2) b[i] = -b[i];  // also test negative divisors (never zero)
        int x = a[i], y = b[i];
        int* o = &want[i * 8];
        o[0] = x + y;
        o[1] = x - y;
        o[2] = x * y;
        o[3] = x / y;  // C++ and PTX both round toward zero
        o[4] = x % y;
        o[5] = (x & y) | (x ^ 255);
        o[6] = static_cast<int32_t>(static_cast<uint32_t>(x) << (y & 7)) >> 2;
        o[7] = -x;
    }
    DeviceBuffer<int> da(a), db(b), dout(n * 8);
    int nArg = n;
    launch(m.kernel("int_ops"), blocksFor(n, 256), 256, {da.arg(), db.arg(), dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

void testConversions(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel convert(x: *f32, outI: *i32, outF: *f32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                outI[i] = i32(x[i] * 3.0);     // f32 -> i32 rounds toward zero
                outF[i] = f32(i) / 4.0 - x[i];
            }
        }
    )");
    const int n = 1000;
    std::vector<float> x = randomFloats(n, 12), wantF(n);
    std::vector<int> wantI(n);
    for (int i = 0; i < n; ++i) {
        wantI[i] = static_cast<int>(x[i] * 3.0f);
        wantF[i] = static_cast<float>(i) / 4.0f - x[i];
    }
    DeviceBuffer<float> dx(x), doutF(n);
    DeviceBuffer<int> doutI(n);
    int nArg = n;
    launch(m.kernel("convert"), blocksFor(n, 256), 256, {dx.arg(), doutI.arg(), doutF.arg(), &nArg});
    checkInts(doutI.download(), wantI);
    checkFloats(doutF.download(), wantF);
}

void testBoolsAndShortCircuit(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel classify(x: *f32, out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            // Short-circuit &&: threads with i >= n never read x[i].
            let pos = i < n && x[i] > 0.0;
            let big = i < n && (x[i] > 5.0 || x[i] < -5.0);
            if (i < n) {
                let code = 0;
                if (pos) { code += 1; }
                if (big) { code += 2; }
                if (!pos && !big) { code = -1; }
                if (pos == big) { code += 10; }
                out[i] = code;
            }
        }
    )");
    const int n = 1000;
    std::vector<float> x = randomFloats(n, 13);
    std::vector<int> want(n);
    for (int i = 0; i < n; ++i) {
        bool pos = x[i] > 0, big = x[i] > 5 || x[i] < -5;
        int code = 0;
        if (pos) code += 1;
        if (big) code += 2;
        if (!pos && !big) code = -1;
        if (pos == big) code += 10;
        want[i] = code;
    }
    DeviceBuffer<float> dx(x);
    DeviceBuffer<int> dout(n);
    int nArg = n;
    launch(m.kernel("classify"), blocksFor(n, 256), 256, {dx.arg(), dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

// The "swap problem": a = phi(10, b), b = phi(20, a) in the loop header.
// Naive phi elimination gets this wrong; the mailbox registers get it right.
void testLoopSwap(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel swap_loop(out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                let a = 10;
                let b = 20;
                let k = 0;
                while (k < i) {
                    let t = a;
                    a = b;
                    b = t;
                    k += 1;
                }
                out[i] = a * 100 + b;
            }
        }
    )");
    const int n = 100;
    std::vector<int> want(n);
    for (int i = 0; i < n; ++i) want[i] = i % 2 == 0 ? 1020 : 2010;
    DeviceBuffer<int> dout(n);
    int nArg = n;
    launch(m.kernel("swap_loop"), 1, 128, {dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

void testFibonacci(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel fib(out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                let a = 0;
                let b = 1;
                for (let k = 0; k < i; k += 1) {
                    let t = a + b;
                    a = b;
                    b = t;
                }
                out[i] = a;
            }
        }
    )");
    const int n = 45;  // fib(44) still fits in an i32
    std::vector<int> want(n);
    int a = 0, b = 1;
    for (int i = 0; i < n; ++i) {
        want[i] = a;
        int t = a + b;
        a = b;
        b = t;
    }
    DeviceBuffer<int> dout(n);
    int nArg = n;
    launch(m.kernel("fib"), 1, 64, {dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

void testCollatz(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel collatz(out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            let x = i + 1;
            let steps = 0;
            while (x != 1) {
                if (x % 2 == 0) {
                    x = x / 2;
                } else {
                    x = 3 * x + 1;
                }
                steps += 1;
            }
            out[i] = steps;
        }
    )");
    const int n = 10000;
    std::vector<int> want(n);
    for (int i = 0; i < n; ++i) {
        long long x = i + 1;
        int steps = 0;
        while (x != 1) {
            x = x % 2 == 0 ? x / 2 : 3 * x + 1;
            steps++;
        }
        want[i] = steps;
    }
    DeviceBuffer<int> dout(n);
    int nArg = n;
    launch(m.kernel("collatz"), blocksFor(n, 256), 256, {dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

void testNestedLoops(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel nested(out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                let total = 0;
                for (let p = 0; p <= i % 7; p += 1) {
                    for (let q = 0; q <= p; q += 1) {
                        total += p * q + i;
                    }
                    if (total > 1000) {
                        total -= 1000;
                    }
                }
                out[i] = total;
            }
        }
    )");
    const int n = 500;
    std::vector<int> want(n);
    for (int i = 0; i < n; ++i) {
        int total = 0;
        for (int p = 0; p <= i % 7; ++p) {
            for (int q = 0; q <= p; ++q) total += p * q + i;
            if (total > 1000) total -= 1000;
        }
        want[i] = total;
    }
    DeviceBuffer<int> dout(n);
    int nArg = n;
    launch(m.kernel("nested"), blocksFor(n, 128), 128, {dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

void testCompileErrors(Gpu&) {
    expectError("kernel k(n: i32) { let x = 1 + 2.0; }", "must have the same type");
    expectError("kernel k() { y = 1; }", "unknown variable 'y'");
    expectError("kernel k() { threadIdx.x = 1; }", "cannot assign to threadIdx.x");
    expectError("kernel k(n: i32) { if (n) { } }", "condition must be bool");
    expectError("kernel k() { let x = 1 }", "expected ';'");
    expectError("kernel k() { let x: f32 = 1; }", "declared as f32");
    expectError("kernel k() { let x = 1; let x = 2; }", "already declared");
    expectError("kernel k(n: i32) { let v = n[0]; }", "cannot index");
    expectError("kernel k(p: *f32) { p[1.0] = 0.0; }", "index must be i32");
    expectError("kernel k() { let x = 1 @ 2; }", "unexpected character '@'");
    expectError("kernel k(b: bool) { }", "kernel parameters must be");
    expectError("kernel k() { let x = 5000000000; }", "too large");
    expectError("kernel k() {}\nkernel k() {}", "defined twice");
    expectError("kernel k() {\n  let a = 1;\n  let b = a + true;\n}", "3:13:");  // line:column
}

struct Test {
    const char* name;
    void (*fn)(Gpu&);
};

const Test kTests[] = {
    {"hand-written PTX runs on the GPU", testHandwrittenPtx},
    {"vector_add.kc", testVectorAdd},
    {"saxpy.kc", testSaxpy},
    {"relu.kc (early return, if)", testRelu},
    {"row_sum.kc (for loop)", testRowSum},
    {"grid_stride.kc (grid-stride loop)", testGridStride},
    {"matmul.kc (naive matrix multiply)", testMatmul},
    {"integer operators", testIntegerOps},
    {"i32 <-> f32 conversions", testConversions},
    {"bools and short-circuit && ||", testBoolsAndShortCircuit},
    {"loop that swaps two variables (phi swap problem)", testLoopSwap},
    {"fibonacci (loop-carried values)", testFibonacci},
    {"collatz (if/else inside while)", testCollatz},
    {"nested loops", testNestedLoops},
    {"compile errors are reported", testCompileErrors},
};

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "-v") == 0) gVerbose = true;

    try {
        Gpu gpu;
        std::printf("GPU: %s (%s)\n", gpu.name().c_str(), gpu.arch().c_str());
        int passed = 0, total = 0;
        for (const Test& t : kTests) {
            total++;
            try {
                t.fn(gpu);
                passed++;
                std::printf("  PASS  %s\n", t.name);
            } catch (const std::exception& e) {
                std::printf("  FAIL  %s\n        %s\n", t.name, e.what());
            }
        }
        std::printf("%d/%d tests passed\n", passed, total);
        return passed == total ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("cannot use the GPU: %s\n", e.what());
        return 1;
    }
}
