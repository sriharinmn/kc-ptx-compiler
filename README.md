# kc: a small GPU kernel compiler that targets NVIDIA PTX

`kc` compiles a small CUDA-like language to **PTX**, NVIDIA's GPU assembly language,
and runs it on a real GPU through the CUDA Driver API. It is written from scratch in C++17:
no LLVM, no parser generators. The same compiler also runs in the browser (WebAssembly),
in a playground that shows every compiler stage live.

```
kernel block_sum(x: *f32, out: *f32, n: i32) {
    shared buf: f32[256];
    let tid = threadIdx.x;
    let i = blockIdx.x * blockDim.x + tid;
    let v = 0.0;
    if (i < n) { v = x[i]; }
    buf[tid] = v;
    barrier();
    for (let s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) { buf[tid] += buf[tid + s]; }
        barrier();
    }
    if (tid == 0) { out[blockIdx.x] = buf[0]; }
}
```

```
source ─▶ lexer ─▶ parser ─▶ type checker ─▶ SSA IR ─▶ optimizer ─▶ PTX ─▶ driver JIT (ptxas) ─▶ GPU
```

**Language:** kernels and device functions (with recursion), `i32` / `f32` / `bool` and
pointers, shared memory + `barrier()`, `atomic_add`, `if` / `while` / `for` / `break` /
`continue`, short-circuit `&&` / `||`, and `sqrt` / `abs` / `min` / `max`.

**Compiler:** hand-written lexer and recursive-descent parser, a strict type checker with
line:column errors, SSA IR built directly from the syntax tree (Braun et al. 2013), and a PTX
backend: phi elimination, the `.param` calling convention, shared-memory variables and atomics.

**Optimizer (`-O1`):** inlining, constant folding (bit-exact with the GPU), CSE over the
dominator tree, loop-invariant code motion and dead code elimination, with an IR verifier
after every pass. Measured against ptxas:

| | PTX instructions | final SASS instructions |
|---|---:|---:|
| all 9 examples, `-O0` | 307 | 288 |
| all 9 examples, `-O1` | 278 | 271 |
| device_fn.kc, `-O0` → `-O1` | 54 → 33 | 36 (2 CALLs) → 19 |

For most kernels the final machine code is byte-identical at `-O0` and `-O1`: ptxas already
does CSE, LICM and folding. The pass that changes the SASS is **inlining**, which ptxas does
not do (nvcc inlines in NVVM, before PTX), so a PTX producer has to do it itself.

**Testing:** 27 end-to-end tests run on the GPU at both `-O0` and `-O1` and compare every
result with a CPU reference; examples are checked with `ptxas`; compute-sanitizer memcheck
and racecheck are clean.

## Build and test (WSL2 / Linux, CUDA 12+)

```bash
bash test.sh            # builds, runs the GPU tests, checks PTX with ptxas, runs compute-sanitizer
~/kc-build/kc examples/saxpy.kc -o -                       # print the PTX
~/kc-build/kc examples/relu.kc --emit-ir -o -              # print the SSA IR too
~/kc-build/kc examples/device_fn.kc --passes -o /dev/null  # what each optimizer pass changed
python3 tools/measure.py                                   # PTX / SASS / registers, -O0 vs -O1
bash web/build.sh       # rebuild the playground (needs Emscripten), then open web/index.html
```
