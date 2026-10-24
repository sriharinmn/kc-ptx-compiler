# kc: a small GPU kernel compiler that targets NVIDIA PTX

`kc` compiles a small CUDA-like language to **PTX**, NVIDIA's GPU assembly language,
and runs it on a real GPU through the CUDA Driver API. It is written from scratch in C++17:
no LLVM, no parser generators.

```
kernel saxpy(n: i32, a: f32, x: *f32, y: *f32) {
    let i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

```
source ─▶ lexer ─▶ parser ─▶ type checker ─▶ SSA IR ─▶ PTX ─▶ driver JIT (ptxas) ─▶ GPU
```

## Build and test (WSL2 / Linux, CUDA 12+)

```bash
bash test.sh            # builds, runs the GPU tests, checks PTX with ptxas, runs compute-sanitizer
~/kc-build/kc examples/saxpy.kc -o -            # print the PTX
~/kc-build/kc examples/relu.kc --emit-ir -o -   # print the SSA IR too
```
