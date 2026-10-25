// End-to-end tests: compile .kc programs with kc, run them on the real GPU,
// and compare every output element with the same computation done on the CPU.
// Every test runs twice: without the optimizer (-O0) and with it (-O1), so an
// optimizer bug shows up as a test that passes at -O0 but fails at -O1.
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
#include "ir.h"
#include "verify.h"

namespace {

bool gVerbose = false;
int gOptLevel = 1;  // set by main() before each run of a test

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
// 'arch' can name an older GPU: PTX for sm_75 also runs on newer GPUs.
Module build(Gpu& gpu, const std::string& source, const std::string& arch = "") {
    CompileOptions opts;
    opts.arch = arch.empty() ? gpu.arch() : arch;
    opts.optLevel = gOptLevel;
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

int countOf(const std::string& text, const std::string& what) {
    int n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) n++;
    return n;
}

std::string ptxAt(const std::string& source, int level) {
    CompileOptions opts;
    opts.optLevel = level;
    return compile(source, opts).ptx;
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

void testSharedReduction(Gpu& gpu) {
    Module m = build(gpu, example("reduce.kc"));
    const int n = 10000, threads = 256, blocks = blocksFor(n, threads);
    // Whole numbers, so the sum is exact no matter in which order the GPU adds.
    std::vector<int> xi = randomInts(n, 14, -10, 10);
    std::vector<float> x(xi.begin(), xi.end()), want(blocks, 0.0f);
    for (int i = 0; i < n; ++i) want[i / threads] += x[i];
    DeviceBuffer<float> dx(x), dout(blocks);
    int nArg = n;
    launch(m.kernel("block_sum"), blocks, threads, {dx.arg(), dout.arg(), &nArg});
    checkFloats(dout.download(), want);
}

void testHistogram(Gpu& gpu) {
    Module m = build(gpu, example("histogram.kc"));
    const int n = 100000;
    int nbins = 16;
    std::vector<int> data = randomInts(n, 15, 0, 1000), want(nbins, 0);
    for (int v : data) want[v % nbins]++;
    DeviceBuffer<int> dd(data), dbins(nbins);
    int nArg = n;
    launch(m.kernel("histogram"), blocksFor(n, 256), 256, {dd.arg(), dbins.arg(), &nArg, &nbins});
    checkInts(dbins.download(), want);
}

void testDeviceFunctions(Gpu& gpu) {
    Module m = build(gpu, example("device_fn.kc"));
    const int n = 3000;
    float gain = 0.08f, bias = 0.5f;
    std::vector<float> x = randomFloats(n, 16), want(n);
    for (int i = 0; i < n; ++i)
        want[i] = std::fabs(x[i]) > 8.0f ? 0.0f : std::min(std::max(gain * x[i] + bias, 0.0f), 1.0f);
    DeviceBuffer<float> dx(x), dy(n);
    int nArg = n;
    launch(m.kernel("adjust"), blocksFor(n, 256), 256, {dx.arg(), dy.arg(), &nArg, &gain, &bias});
    checkFloats(dy.download(), want);
}

void testRecursion(Gpu& gpu) {
    Module m = build(gpu, R"(
        fn fact(k: i32) -> i32 {
            if (k <= 1) {
                return 1;
            }
            return k * fact(k - 1);
        }
        fn gcd(a: i32, b: i32) -> i32 {
            if (b == 0) {
                return a;
            } else {
                return gcd(b, a % b);
            }
        }
        kernel k(out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                out[2 * i] = fact(i % 13);
                out[2 * i + 1] = gcd(i * 6, 84);
            }
        }
    )");
    const int n = 200;
    std::vector<int> want(2 * n);
    for (int i = 0; i < n; ++i) {
        int f = 1;
        for (int k = 2; k <= i % 13; ++k) f *= k;
        int a = i * 6, b = 84;
        while (b) {
            int t = a % b;
            a = b;
            b = t;
        }
        want[2 * i] = f;
        want[2 * i + 1] = a;
    }
    DeviceBuffer<int> dout(2 * n);
    int nArg = n;
    launch(m.kernel("k"), blocksFor(n, 64), 64, {dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

// The hard cases for the inliner: a loop inside the callee, several returns,
// a call inside a loop condition, nested calls, and a call with no result.
void testInliningCases(Gpu& gpu) {
    Module m = build(gpu, R"(
        fn count_digits(v: i32) -> i32 {
            let n = 1;
            let x = v;
            while (x >= 10) {
                x = x / 10;
                n += 1;
            }
            return n;
        }
        fn sign(v: i32) -> i32 {
            if (v > 0) {
                return 1;
            }
            if (v < 0) {
                return -1;
            }
            return 0;
        }
        fn store_twice(p: *i32, i: i32, v: i32) {
            p[2 * i] = v;
            p[2 * i + 1] = v;
        }
        kernel k(a: *i32, out: *i32, pairs: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            let v = a[i];
            let steps = 0;
            while (count_digits(abs(v) + steps) < 4) {
                steps += 1;
            }
            out[i] = sign(v) * 100000 + steps * 10 + sign(sign(v) - 1);
            store_twice(pairs, i, v);
        }
    )");
    const int n = 1000;
    std::vector<int> a = randomInts(n, 23, -2000, 2000), want(n), wantPairs(2 * n);
    auto sign = [](int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; };
    for (int i = 0; i < n; ++i) {
        int steps = std::max(0, 1000 - std::abs(a[i]));  // until |v| + steps has 4 digits
        want[i] = sign(a[i]) * 100000 + steps * 10 + sign(sign(a[i]) - 1);
        wantPairs[2 * i] = wantPairs[2 * i + 1] = a[i];
    }
    DeviceBuffer<int> da(a), dout(n), dpairs(2 * n);
    int nArg = n;
    launch(m.kernel("k"), blocksFor(n, 128), 128, {da.arg(), dout.arg(), dpairs.arg(), &nArg});
    checkInts(dout.download(), want);
    checkInts(dpairs.download(), wantPairs);
}

void testBreakContinue(Gpu& gpu) {
    Module m = build(gpu, R"(
        // For each row: sum the positive values, skipping negatives (continue),
        // and stop at the first zero (break).
        kernel scan(m: *i32, out: *i32, rows: i32, cols: i32) {
            let r = blockIdx.x * blockDim.x + threadIdx.x;
            if (r >= rows) {
                return;
            }
            let sum = 0;
            for (let c = 0; c < cols; c += 1) {
                let v = m[r * cols + c];
                if (v == 0) {
                    break;
                }
                if (v < 0) {
                    continue;
                }
                sum += v;
            }
            let steps = 0;
            while (true) {
                steps += 1;
                if (steps >= r % 5 + 1) {
                    break;
                }
            }
            out[r] = sum * 10 + steps;
        }
    )");
    int rows = 500, cols = 20;
    std::vector<int> mat = randomInts(rows * cols, 17, -3, 9), want(rows);
    for (int r = 0; r < rows; ++r) {
        int sum = 0;
        for (int c = 0; c < cols; ++c) {
            int v = mat[r * cols + c];
            if (v == 0) break;
            if (v < 0) continue;
            sum += v;
        }
        want[r] = sum * 10 + (r % 5 + 1);
    }
    DeviceBuffer<int> dm(mat), dout(rows);
    launch(m.kernel("scan"), blocksFor(rows, 128), 128, {dm.arg(), dout.arg(), &rows, &cols});
    checkInts(dout.download(), want);
}

void testMathBuiltins(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel math(x: *f32, a: *i32, outF: *f32, outI: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                outF[3 * i] = sqrt(abs(x[i]));
                outF[3 * i + 1] = min(x[i], 1.5);
                outF[3 * i + 2] = max(x[i], -2.0);
                outI[2 * i] = abs(a[i]);
                outI[2 * i + 1] = min(a[i], 7) + max(a[i], -7);
            }
        }
    )");
    const int n = 1000;
    std::vector<float> x = randomFloats(n, 18), wantF(3 * n);
    std::vector<int> a = randomInts(n, 19, -100, 100), wantI(2 * n);
    for (int i = 0; i < n; ++i) {
        wantF[3 * i] = std::sqrt(std::fabs(x[i]));
        wantF[3 * i + 1] = std::min(x[i], 1.5f);
        wantF[3 * i + 2] = std::max(x[i], -2.0f);
        wantI[2 * i] = std::abs(a[i]);
        wantI[2 * i + 1] = std::min(a[i], 7) + std::max(a[i], -7);
    }
    DeviceBuffer<float> dx(x), doutF(3 * n);
    DeviceBuffer<int> da(a), doutI(2 * n);
    int nArg = n;
    launch(m.kernel("math"), blocksFor(n, 256), 256, {dx.arg(), da.arg(), doutF.arg(), doutI.arg(), &nArg});
    checkFloats(doutF.download(), wantF);
    checkInts(doutI.download(), wantI);
}

void testSharedAtomics(Gpu& gpu) {
    // A faster histogram: each block counts into shared memory first, then adds
    // its counts to global memory once (fewer slow global atomics).
    Module m = build(gpu, R"(
        kernel histogram_shared(data: *i32, bins: *i32, n: i32) {
            shared local: i32[64];
            let tid = threadIdx.x;
            if (tid < 64) {
                local[tid] = 0;
            }
            barrier();
            let i = blockIdx.x * blockDim.x + tid;
            if (i < n) {
                atomic_add(local[data[i] % 64], 1);
            }
            barrier();
            if (tid < 64) {
                atomic_add(bins[tid], local[tid]);
            }
        }
        kernel total(x: *f32, out: *f32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                atomic_add(out[0], x[i]);
            }
        }
    )");
    const int n = 50000;
    std::vector<int> data = randomInts(n, 20, 0, 100000), want(64, 0);
    for (int v : data) want[v % 64]++;
    DeviceBuffer<int> dd(data), dbins(64);
    int nArg = n;
    launch(m.kernel("histogram_shared"), blocksFor(n, 256), 256, {dd.arg(), dbins.arg(), &nArg});
    checkInts(dbins.download(), want);

    std::vector<int> xi = randomInts(n, 21, -5, 5);  // whole numbers: the sum is exact in any order
    std::vector<float> x(xi.begin(), xi.end());
    float sum = 0;
    for (float v : x) sum += v;
    DeviceBuffer<float> dx(x), dtotal(1);
    launch(m.kernel("total"), blocksFor(n, 256), 256, {dx.arg(), dtotal.arg(), &nArg});
    checkFloats(dtotal.download(), {sum});
}

// warp_sum is one redux.sync instruction on sm_80+, and 5 shuffles on older GPUs.
// PTX made for sm_75 also runs on this GPU, so both versions are tested here.
void testWarpReduce(Gpu& gpu) {
    const int n = 100000;
    std::vector<int> x = randomInts(n, 24, -100, 100);
    int want = 0;
    for (int v : x) want += v;
    for (const char* arch : {"sm_89", "sm_75"}) {
        Module m = build(gpu, example("warp_reduce.kc"), arch);
        DeviceBuffer<int> dx(x), dout(1);
        int nArg = n;
        launch(m.kernel("warp_reduce"), blocksFor(n, 256), 256, {dx.arg(), dout.arg(), &nArg});
        int got = dout.download()[0];
        if (got != want)
            throw TestFailure(std::string(arch) + ": got " + std::to_string(got) + ", want " + std::to_string(want));
    }
}

void testWarpBuiltins(Gpu& gpu) {
    const char* source = R"(
        kernel warp_ops(x: *i32, f: *f32, out: *i32, outF: *f32) {
            let t = threadIdx.x;
            let v = x[t];
            out[6 * t] = lane_id();
            out[6 * t + 1] = warp_sum(v);
            out[6 * t + 2] = shfl_down(v, 1);
            out[6 * t + 3] = shfl_xor(v, 3);
            out[6 * t + 4] = ballot(v > 0);
            out[6 * t + 5] = popc(ballot(v > 0));
            outF[2 * t] = warp_sum(f[t]);
            outF[2 * t + 1] = shfl_down(f[t], 4);
        }
    )";
    const int threads = 64;  // two warps
    std::vector<int> x = randomInts(threads, 25, -50, 50), want(6 * threads);
    std::vector<int> fi = randomInts(threads, 26, -1000, 1000);  // whole numbers: float sums are exact
    std::vector<float> f(fi.begin(), fi.end()), wantF(2 * threads);
    for (int t = 0; t < threads; ++t) {
        int lane = t % 32, base = t - lane;
        int sum = 0, mask = 0;
        float fsum = 0;
        for (int l = 0; l < 32; ++l) {
            sum += x[base + l];
            fsum += f[base + l];
            if (x[base + l] > 0) mask |= static_cast<int>(1u << l);
        }
        want[6 * t] = lane;
        want[6 * t + 1] = sum;
        want[6 * t + 2] = lane + 1 < 32 ? x[t + 1] : x[t];  // past the last lane: keeps its own value
        want[6 * t + 3] = x[base + (lane ^ 3)];
        want[6 * t + 4] = mask;
        want[6 * t + 5] = __builtin_popcount(static_cast<unsigned>(mask));
        wantF[2 * t] = fsum;
        wantF[2 * t + 1] = lane + 4 < 32 ? f[t + 4] : f[t];
    }
    for (const char* arch : {"sm_89", "sm_75"}) {
        Module m = build(gpu, source, arch);
        DeviceBuffer<int> dx(x), dout(6 * threads);
        DeviceBuffer<float> df(f), doutF(2 * threads);
        launch(m.kernel("warp_ops"), 1, threads, {dx.arg(), df.arg(), dout.arg(), doutF.arg()});
        checkInts(dout.download(), want);
        checkFloats(doutF.download(), wantF);
    }
}

// At -O0 the GPU computes these; at -O1 the optimizer folds them on the CPU.
// Both must give exactly the GPU's answers, bit for bit, including the odd cases.
void testConstantFolding(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel consts(outI: *i32, outF: *f32) {
            outI[0] = 7 / -2;                  // rounds toward zero
            outI[1] = -7 % 3;
            outI[2] = 1 << 33;                 // PTX clamps shift amounts to 32
            outI[3] = -8 >> 40;
            outI[4] = 2147483647 + 1;          // wraps around
            outI[5] = i32(3.99);
            outI[6] = i32(-3.99);
            outI[7] = i32(10000000000.0);      // clamps to the i32 range
            outI[8] = i32(-10000000000.0);
            outI[9] = abs(-5) + min(3, -4) + max(3, -4);
            outI[10] = (12 & 10) | (12 ^ 10);
            outI[11] = 100000 * 100000;        // keeps the low 32 bits
            outI[12] = 0;
            if (3 < 4 && !(2.0 > 3.0)) {
                outI[12] = 1;
            }
            outF[0] = 1.0 / 3.0;
            outF[1] = sqrt(2.0);
            outF[2] = f32(16777217);           // not exactly representable as f32
            outF[3] = 0.1 + 0.2;
            outF[4] = -(0.0);                  // negative zero
            outF[5] = 1.0 / 0.0;               // infinity
            outF[6] = min(-1.5, 2.5);
            outF[7] = f32(-7) * 0.5;
        }
    )");
    DeviceBuffer<int> dI(13);
    DeviceBuffer<float> dF(8);
    launch(m.kernel("consts"), 1, 1, {dI.arg(), dF.arg()});
    checkInts(dI.download(), {-3, -1, 0, -1, INT32_MIN, 3, -3, INT32_MAX, INT32_MIN, 4, 14, 1410065408, 1});
    std::vector<float> wantF = {1.0f / 3.0f, std::sqrt(2.0f), 16777216.0f, 0.1f + 0.2f,
                                -0.0f, INFINITY, -1.5f, -3.5f};
    std::vector<float> gotF = dF.download();
    for (size_t i = 0; i < wantF.size(); ++i)
        if (std::memcmp(&gotF[i], &wantF[i], sizeof(float)) != 0)
            throw TestFailure("outF[" + std::to_string(i) + "]: got " + std::to_string(gotF[i]) + ", want " +
                              std::to_string(wantF[i]) + " (compared bit for bit)");
}

// At -O1, x / 2^k and x % 2^k become shifts and masks (ptxas would emit a whole
// division routine). Negative x needs a correction so it still rounds toward zero.
void testDivisionByPowersOfTwo(Gpu& gpu) {
    const char* source = R"(
        kernel pow2(x: *i32, out: *i32, n: i32) {
            let i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                let v = x[i];
                out[6 * i] = v / 2;
                out[6 * i + 1] = v / 8;
                out[6 * i + 2] = v % 2;
                out[6 * i + 3] = v % 16;
                out[6 * i + 4] = threadIdx.x / 4;    // never negative: a single shift
                out[6 * i + 5] = threadIdx.x % 32;   // never negative: a single and
            }
        }
    )";
    if (gOptLevel == 1) {
        std::string ptx = ptxAt(source, 1);
        CHECK(countOf(ptx, "div.s32") == 0 && countOf(ptx, "rem.s32") == 0);
    }
    Module m = build(gpu, source);
    const int n = 2000;
    std::vector<int> x = randomInts(n, 27, -100000, 100000), want(6 * n);
    x[0] = INT32_MIN;  // the corner cases
    x[1] = INT32_MAX;
    x[2] = -1;
    x[3] = -7;
    for (int i = 0; i < n; ++i) {
        int v = x[i], t = i % 128;  // launched with 128 threads per block
        int* o = &want[6 * i];
        o[0] = v / 2;  // C++ rounds toward zero, like PTX's div.s32
        o[1] = v / 8;
        o[2] = v % 2;
        o[3] = v % 16;
        o[4] = t / 4;
        o[5] = t % 32;
    }
    DeviceBuffer<int> dx(x), dout(6 * n);
    int nArg = n;
    launch(m.kernel("pow2"), blocksFor(n, 128), 128, {dx.arg(), dout.arg(), &nArg});
    checkInts(dout.download(), want);
}

// CSE may reuse a load only if memory cannot have changed in between.
void testLoadsAfterStoresAndBarriers(Gpu& gpu) {
    Module m = build(gpu, R"(
        kernel k(p: *i32, out: *i32) {
            shared s: i32[64];
            let t = threadIdx.x;
            let a = p[t];
            p[t] = a + 1;
            let b = p[t];                  // must load again: the store changed p[t]
            s[t] = 0;
            barrier();
            let before = s[(t + 1) % 64];  // 0
            barrier();
            s[t] = t;
            barrier();
            let after = s[(t + 1) % 64];   // must load again: another thread wrote it
            out[t] = a * 10000 + b * 100 + before + after;
        }
    )");
    std::vector<int> p = randomInts(64, 22, 0, 50), want(64);
    for (int t = 0; t < 64; ++t) want[t] = p[t] * 10000 + (p[t] + 1) * 100 + (t + 1) % 64;
    DeviceBuffer<int> dp(p), dout(64);
    launch(m.kernel("k"), 1, 64, {dp.arg(), dout.arg()});
    checkInts(dout.download(), want);
}

// Each pass does the job it is meant to do (checked in the PTX it produces).
void testOptimizerPasses(Gpu&) {
    // inline: no calls or .func left (ptxas would not inline them for us)
    CHECK(countOf(ptxAt(example("device_fn.kc"), 0), "call") == 2);
    CHECK(countOf(ptxAt(example("device_fn.kc"), 1), "call") == 0);
    CHECK(countOf(ptxAt(example("device_fn.kc"), 1), ".func") == 0);
    // cse: saxpy computes the address of y[i] once instead of twice
    CHECK(countOf(ptxAt(example("saxpy.kc"), 0), "mul.wide") == 3);
    CHECK(countOf(ptxAt(example("saxpy.kc"), 1), "mul.wide") == 2);
    // licm: row_sum's loop body ($L__BB0_2 up to the jump back) no longer multiplies r * cols
    auto loopBody = [](const std::string& ptx) {
        size_t start = ptx.find("$L__BB0_2:");
        return ptx.substr(start, ptx.find("bra.uni \t$L__BB0_2;") - start);
    };
    CHECK(countOf(loopBody(ptxAt(example("row_sum.kc"), 0)), "mul.lo") == 1);
    CHECK(countOf(loopBody(ptxAt(example("row_sum.kc"), 1)), "mul.lo") == 0);
    // simplify + dce: constants folded, the impossible branch and the unused load are gone
    std::string folded = ptxAt(R"(
        kernel k(a: *i32, out: *i32) {
            let unused = a[0] * 2;
            let x = 2 + 3 * 4;
            if (x > 100) {
                out[1] = 5;
            }
            out[0] = x;
        }
    )", 1);
    CHECK(countOf(folded, "ld.global") == 0);
    CHECK(countOf(folded, "bra") == 0);
    CHECK(countOf(folded, "st.global.u32 \t[%rd0], 14;") == 1);
    // the same input always gives the same output
    CHECK(ptxAt(example("matmul.kc"), 1) == ptxAt(example("matmul.kc"), 1));
}

std::string warningsFor(const std::string& source) { return compile(source).warnings; }

// barrier() and warp operations need every thread; the divergence analysis warns
// when only some threads can reach them, and stays quiet when all threads do.
void testDivergenceWarnings(Gpu&) {
    auto expectWarning = [](const std::string& source, const std::string& expected) {
        std::string w = warningsFor(source);
        if (w.find(expected) == std::string::npos)
            throw TestFailure("expected a warning containing \"" + expected + "\", got: \"" + w + "\"");
    };
    auto expectQuiet = [](const std::string& source) {
        std::string w = warningsFor(source);
        if (!w.empty()) throw TestFailure("unexpected warning: " + w);
    };
    // depends on threadIdx: warn, and name the line of the branch
    expectWarning("kernel k() {\n if (threadIdx.x < 16) {\n  barrier();\n }\n}", "3:3: warning: barrier() is only "
                  "reached by some threads of the block (the branch on line 2");
    // threads that returned early never reach the barrier
    expectWarning("kernel k(n: i32) {\n let i = blockIdx.x * blockDim.x + threadIdx.x;\n if (i >= n) { return; }\n"
                  " barrier();\n}", "4:2: warning: barrier()");
    // the loop counter differs between threads after a loop they leave at different times
    expectWarning("kernel k() {\n let c = 0;\n while (c < threadIdx.x) { c += 1; }\n if (c > 3) { barrier(); }\n}",
                  "the branch on line 4");
    expectWarning("kernel k(x: *i32, n: i32) {\n let i = threadIdx.x;\n if (i < n) { x[i] = warp_sum(x[i]); }\n}",
                  "warp_sum() needs all 32 threads");
    // uniform conditions: blockIdx, parameters, a value loaded from one address, a uniform loop
    expectQuiet("kernel k(x: *f32, n: i32) {\n if (blockIdx.x == 0) { barrier(); }\n if (n > 5) { barrier(); }\n"
                " if (x[0] > 0.0) { barrier(); }\n for (let s = 0; s < n; s += 1) { barrier(); }\n}");
    for (const char* name : {"reduce.kc", "warp_reduce.kc", "histogram.kc", "matmul.kc"}) expectQuiet(example(name));
}

// Appends an instruction to block b (for building IR by hand).
Inst* append(IrFunction& f, Block* b, Opc op, IrType type, std::vector<Inst*> ops = {}) {
    Inst* in = f.newInst(op, type);
    in->ops = std::move(ops);
    in->parent = b;
    b->insts.push_back(in);
    return in;
}

void expectVerifyError(const IrFunction& f, const std::string& expected) {
    try {
        verifyFunction(f, "test");
    } catch (const InternalError& e) {
        if (std::string(e.what()).find(expected) == std::string::npos)
            throw TestFailure("wrong verifier error \"" + std::string(e.what()) + "\"");
        return;
    }
    throw TestFailure("verifier accepted broken IR, expected: " + expected);
}

// The verifier catches broken IR (built by hand here, the way a buggy pass would leave it).
void testVerifier(Gpu&) {
    {
        IrFunction f;
        f.name = "f";
        Block* b = f.newBlock();
        Inst* one = f.newInst(Opc::Const, IrType::S32);
        append(f, b, Opc::Add, IrType::S32, {one, one});
        one->parent = b;
        b->insts.push_back(one);  // defined after its use
        append(f, b, Opc::Ret, IrType::Void);
        expectVerifyError(f, "uses a value before it is computed");
        std::swap(b->insts[0], b->insts[1]);
        verifyFunction(f, "test");  // in the right order it is fine
    }
    {
        IrFunction f;
        f.name = "f";
        Block* b0 = f.newBlock();
        Block* b1 = f.newBlock();
        append(f, b0, Opc::Br, IrType::Void)->targets = {b1};  // b1->preds does not list b0
        append(f, b1, Opc::Ret, IrType::Void);
        expectVerifyError(f, "predecessor list does not match");
    }
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
    // session 2 features
    expectError("kernel k() { break; }", "'break' can only be used inside a loop");
    expectError("kernel k() { return 1; }", "kernels cannot return a value");
    expectError("fn f(x: i32) -> i32 { if (x > 0) { return 1; } }", "must end with 'return <value>;'");
    expectError("fn f(x: i32) -> i32 { return x; } kernel k() { let a = f(1, 2); }", "takes 1 argument(s), but 2");
    expectError("fn f(x: i32) -> i32 { return x; } kernel k() { let a = f(1.0); }", "argument 1 of 'f' must be i32");
    expectError("kernel a() {} kernel b() { a(); }", "kernels are launched from the CPU");
    expectError("fn f() { shared s: f32[4]; }", "only be declared in kernels");
    expectError("kernel k() { shared s: f32[20000]; }", "more than 48 KB");
    expectError("kernel k(p: *i32) { atomic_add(p, 1); }", "must be an array element");
    expectError("fn f() {} kernel k() { let a = f(); }", "does not return a value");
    expectError("kernel k(n: i32) { n + 1; }", "only function calls can be used as statements");
    expectError("kernel k() { let x = nope(1); }", "unknown function 'nope'");
    expectError("kernel k() { shared s: f32[4]; s = s; }", "cannot assign to shared array");
    expectError("fn sqrt(x: f32) -> f32 { return x; }", "built-in function");
    // session 4: warp builtins
    expectError("kernel k() { let s = warp_sum(true); }", "warp_sum needs an i32 or f32");
    expectError("kernel k() { let b = ballot(1); }", "ballot needs a bool");
    expectError("kernel k() { let v = shfl_down(1.0, 2.0); }", "an i32 lane offset");
    expectError("kernel k() { let c = popc(1.5); }", "popc needs an i32");
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
    {"reduce.kc (shared memory + barrier)", testSharedReduction},
    {"histogram.kc (atomic_add)", testHistogram},
    {"device_fn.kc (device functions, bool return)", testDeviceFunctions},
    {"recursive device functions", testRecursion},
    {"inlining: loops, several returns, calls in conditions", testInliningCases},
    {"break and continue", testBreakContinue},
    {"sqrt, abs, min, max", testMathBuiltins},
    {"atomics on shared memory and on f32", testSharedAtomics},
    {"warp_reduce.kc (redux.sync on sm_89, shuffles on sm_75)", testWarpReduce},
    {"warp builtins: lane_id, warp_sum, shfl_down, shfl_xor, ballot, popc", testWarpBuiltins},
    {"constant folding gives the GPU's answers, bit for bit", testConstantFolding},
    {"loads are not reused across stores and barriers", testLoadsAfterStoresAndBarriers},
    {"division by powers of two becomes shifts (also for negative numbers)", testDivisionByPowersOfTwo},
    {"optimizer passes: inline, cse, licm, simplify, dce", testOptimizerPasses},
    {"IR verifier catches broken IR", testVerifier},
    {"divergence warnings: barrier() and warp ops reached by some threads", testDivergenceWarnings},
    {"compile errors are reported", testCompileErrors},
};

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "-v") == 0) gVerbose = true;

    try {
        Gpu gpu;
        std::printf("GPU: %s (%s), each test runs at -O0 and at -O1\n", gpu.name().c_str(), gpu.arch().c_str());
        int passed = 0, total = 0;
        for (const Test& t : kTests) {
            total++;
            std::string failure;
            for (int level : {0, 1}) {
                gOptLevel = level;
                try {
                    t.fn(gpu);
                } catch (const std::exception& e) {
                    failure = "at -O" + std::to_string(level) + ": " + e.what();
                    break;
                }
            }
            if (failure.empty()) {
                passed++;
                std::printf("  PASS  %s\n", t.name);
            } else {
                std::printf("  FAIL  %s\n        %s\n", t.name, failure.c_str());
            }
        }
        std::printf("%d/%d tests passed\n", passed, total);
        return passed == total ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("cannot use the GPU: %s\n", e.what());
        return 1;
    }
}
