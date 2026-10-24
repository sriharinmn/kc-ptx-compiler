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
source ─▶ lexer ─▶ parser ─▶ type checker ─▶ SSA IR ─▶ PTX ─▶ driver JIT (ptxas) ─▶ GPU
```

**Language:** kernels and device functions (with recursion), `i32` / `f32` / `bool` and
pointers, shared memory + `barrier()`, `atomic_add`, `if` / `while` / `for` / `break` /
`continue`, short-circuit `&&` / `||`, and `sqrt` / `abs` / `min` / `max`.

**Compiler:** hand-written lexer and recursive-descent parser, a strict type checker with
line:column errors, SSA IR built directly from the syntax tree (Braun et al. 2013), and a PTX
backend: phi elimination, the `.param` calling convention, shared-memory variables and atomics.

**Testing:** 22 end-to-end tests run on the GPU and compare every result with a CPU
reference; examples are checked with `ptxas`; compute-sanitizer memcheck and racecheck are clean.

## Build and test (WSL2 / Linux, CUDA 12+)

```bash
bash test.sh            # builds, runs the GPU tests, checks PTX with ptxas, runs compute-sanitizer
~/kc-build/kc examples/saxpy.kc -o -            # print the PTX
~/kc-build/kc examples/relu.kc --emit-ir -o -   # print the SSA IR too
bash web/build.sh       # rebuild the playground (needs Emscripten), then open web/index.html
```
