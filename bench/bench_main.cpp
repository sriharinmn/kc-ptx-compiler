// kc vs nvcc on the same kernels:
//   kc:   examples/*.kc   -> kc -O0 / kc -O1 -> PTX --+
//   nvcc: bench/kernels.cu -> nvcc ----------> PTX --+--> the same ptxas (PTX Compiler API) -> GPU
// Every version must give the right answer first; then each kernel is timed on
// the GPU (median of 5 runs, each the average of 20 back-to-back launches).
// Usage: kc-bench      (prints a Markdown table)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "compiler.h"
#include "gpu.h"
#include "ptx_compiler.h"

namespace {

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Deterministic pseudo-random inputs (the same generator as the tests).
unsigned nextRandom(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
}
std::vector<int> randomInts(size_t n, unsigned seed, int lo, int hi) {
    std::vector<int> v(n);
    for (auto& x : v) x = lo + static_cast<int>(nextRandom(seed) % static_cast<unsigned>(hi - lo + 1));
    return v;
}
std::vector<float> randomFloats(size_t n, unsigned seed) {  // -10.00 .. 10.00
    std::vector<int> i = randomInts(n, seed, -1000, 1000);
    std::vector<float> v(n);
    for (size_t k = 0; k < n; ++k) v[k] = i[k] / 100.0f;
    return v;
}

bool close(const std::vector<float>& got, const std::vector<float>& want) {
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - want[i]) > 1e-4f * std::max(1.0f, std::fabs(want[i]))) return false;
    return true;
}

struct Variant {
    std::string label;
    Module module;
    std::string log;  // ptxas info log (registers per kernel)
};

Variant makeVariant(const std::string& label, const std::string& ptx, const std::string& arch) {
    CompiledPtx c = compilePtx(ptx, arch);
    return {label, Module(c.cubin), c.infoLog};
}

struct Row {
    std::string kernel;
    std::vector<float> ms;
    std::vector<int> regs;
};

class Bench {
public:
    explicit Bench(std::vector<Variant>& variants) : variants_(variants) {}

    // reset: puts the inputs back; check: is the output right? (both run around one launch per variant)
    void run(const std::string& kernel, unsigned blocks, unsigned threads, std::vector<void*> args,
             const std::function<void()>& reset, const std::function<bool()>& check) {
        Row row{kernel, {}, {}};
        for (Variant& v : variants_) {
            CUfunction f = v.module.kernel(kernel);
            reset();
            launch(f, blocks, threads, args);
            if (!check()) throw std::runtime_error(kernel + ": wrong result from " + v.label);
            std::vector<float> times;
            for (int i = 0; i < 5; ++i) times.push_back(timeKernelMs(f, blocks, threads, args, 20));
            std::sort(times.begin(), times.end());
            row.ms.push_back(times[2]);
            row.regs.push_back(registersUsed(v.log, kernel));
        }
        rows_.push_back(row);
    }

    void print() const {
        std::printf("| kernel |");
        for (const Variant& v : variants_) std::printf(" %s (ms) |", v.label.c_str());
        std::printf(" kc -O1 / nvcc | registers (kc -O1, nvcc) |\n|---|");
        for (size_t i = 0; i < variants_.size(); ++i) std::printf("---:|");
        std::printf("---:|---:|\n");
        for (const Row& r : rows_) {
            std::printf("| %s |", r.kernel.c_str());
            for (float ms : r.ms) std::printf(" %.3f |", ms);
            std::printf(" %.2f |", r.ms[1] / r.ms[2]);
            std::printf(" %d, %d |\n", r.regs[1], r.regs[2]);
        }
    }

private:
    std::vector<Variant>& variants_;
    std::vector<Row> rows_;
};

}  // namespace

int main() {
    try {
        Gpu gpu;
        std::string arch = gpu.arch();
        std::printf("GPU: %s (%s). Every version compiled by the same ptxas (nvPTXCompiler %s).\n\n",
                    gpu.name().c_str(), arch.c_str(), ptxCompilerVersion().c_str());

        // All the example kernels in one kc program, compiled at -O0 and at -O1.
        std::string kcSource;
        for (const char* f : {"saxpy", "row_sum", "matmul", "reduce", "warp_reduce", "histogram", "device_fn"})
            kcSource += readFile(std::string(KC_SOURCE_DIR) + "/examples/" + f + ".kc") + "\n";
        CompileOptions o0, o1;
        o0.arch = o1.arch = arch;
        o0.optLevel = 0;
        std::vector<Variant> variants;
        variants.push_back(makeVariant("kc -O0", compile(kcSource, o0).ptx, arch));
        variants.push_back(makeVariant("kc -O1", compile(kcSource, o1).ptx, arch));
        variants.push_back(makeVariant("nvcc", readFile(NVCC_PTX), arch));
        Bench bench(variants);

        const int n = 1 << 24;  // 16M elements
        const unsigned threads = 256, blocks = n / threads;
        int nArg = n;

        {  // saxpy: y = a * x + y
            std::vector<float> x = randomFloats(n, 1), y = randomFloats(n, 2), want(n);
            float a = 2.5f;
            for (int i = 0; i < n; ++i) want[i] = a * x[i] + y[i];
            DeviceBuffer<float> dx(x), dy(y);
            bench.run("saxpy", blocks, threads, {&nArg, &a, dx.arg(), dy.arg()}, [&] { dy.upload(y); },
                      [&] { return close(dy.download(), want); });
        }
        {  // row_sum: one thread per row
            int rows = 1 << 16, cols = 256;
            std::vector<float> m = randomFloats(static_cast<size_t>(rows) * cols, 3), want(rows, 0.0f);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) want[r] += m[static_cast<size_t>(r) * cols + c];
            DeviceBuffer<float> dm(m), dout(rows);
            bench.run("row_sum", rows / threads, threads, {dm.arg(), dout.arg(), &rows, &cols}, [] {},
                      [&] { return close(dout.download(), want); });
        }
        {  // matmul: naive, one thread per output element; small whole numbers, so exact
            int size = 512;
            std::vector<int> ai = randomInts(size * size, 4, -3, 3), bi = randomInts(size * size, 5, -3, 3);
            std::vector<float> a(ai.begin(), ai.end()), b(bi.begin(), bi.end()), want(size * size, 0.0f);
            for (int r = 0; r < size; ++r)
                for (int k = 0; k < size; ++k)
                    for (int c = 0; c < size; ++c) want[r * size + c] += a[r * size + k] * b[k * size + c];
            DeviceBuffer<float> da(a), db(b), dc(size * size);
            bench.run("matmul", size * size / threads, threads, {da.arg(), db.arg(), dc.arg(), &size}, [] {},
                      [&] { return dc.download() == want; });
        }
        {  // block_sum: shared-memory tree reduction, one sum per block
            std::vector<int> xi = randomInts(n, 6, -10, 10);
            std::vector<float> x(xi.begin(), xi.end()), want(blocks, 0.0f);
            for (int i = 0; i < n; ++i) want[i / threads] += x[i];
            DeviceBuffer<float> dx(x), dout(blocks);
            bench.run("block_sum", blocks, threads, {dx.arg(), dout.arg(), &nArg}, [] {},
                      [&] { return dout.download() == want; });
        }
        {  // warp_reduce: warp_sum + one atomic per warp
            std::vector<int> x = randomInts(n, 7, -100, 100);
            int want = 0;
            for (int v : x) want += v;
            DeviceBuffer<int> dx(x), dout(1);
            bench.run("warp_reduce", blocks, threads, {dx.arg(), dout.arg(), &nArg}, [&] { dout.clear(); },
                      [&] { return dout.download()[0] == want; });
        }
        {  // histogram: one atomic per element
            int nbins = 1024;
            std::vector<int> data = randomInts(n, 8, 0, 1 << 20), want(nbins, 0);
            for (int v : data) want[v % nbins]++;
            DeviceBuffer<int> dd(data), dbins(nbins);
            bench.run("histogram", blocks, threads, {dd.arg(), dbins.arg(), &nArg, &nbins}, [&] { dbins.clear(); },
                      [&] { return dbins.download() == want; });
        }
        {  // adjust (device_fn.kc): two device functions per element
            float gain = 0.08f, bias = 0.5f;
            std::vector<float> x = randomFloats(n, 9), want(n);
            for (int i = 0; i < n; ++i)
                want[i] = std::fabs(x[i]) > 8.0f ? 0.0f : std::min(std::max(gain * x[i] + bias, 0.0f), 1.0f);
            DeviceBuffer<float> dx(x), dy(n);
            bench.run("adjust", blocks, threads, {dx.arg(), dy.arg(), &nArg, &gain, &bias}, [] {},
                      [&] { return close(dy.download(), want); });
        }

        bench.print();
        return 0;
    } catch (const std::exception& e) {
        std::printf("benchmark failed: %s\n", e.what());
        return 1;
    }
}
