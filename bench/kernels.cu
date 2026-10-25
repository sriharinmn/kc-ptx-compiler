// The benchmark's reference: the same kernels as kc's examples/*.kc, written
// line for line in CUDA C++ and compiled to PTX by nvcc (see CMakeLists.txt).
// extern "C" keeps the names unmangled, so both versions are found by the same name.
// No __restrict__ and no tricks: the same algorithm as the kc version.

extern "C" __global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

extern "C" __global__ void row_sum(float* m, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c += 1) {
            sum += m[r * cols + c];
        }
        out[r] = sum;
    }
}

extern "C" __global__ void matmul(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        float sum = 0.0f;
        for (int k = 0; k < n; k += 1) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[idx] = sum;
    }
}

extern "C" __global__ void block_sum(float* x, float* out, int n) {
    __shared__ float buf[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    if (i < n) {
        v = x[i];
    }
    buf[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            buf[tid] += buf[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = buf[0];
    }
}

extern "C" __global__ void warp_reduce(int* x, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = 0;
    if (i < n) {
        v = x[i];
    }
    int total = __reduce_add_sync(0xffffffff, v);  // redux.sync on sm_80+, like kc's warp_sum
    if ((threadIdx.x & 31) == 0) {                 // lane 0 (kc uses %laneid; same thing in a 1-D block)
        atomicAdd(&out[0], total);
    }
}

extern "C" __global__ void histogram(int* data, int* bins, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int b = data[i] % nbins;
        atomicAdd(&bins[b], 1);
    }
}

__device__ float clamp(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__device__ bool is_outlier(float v) { return fabsf(v) > 8.0f; }

extern "C" __global__ void adjust(float* x, float* y, int n, float gain, float bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (is_outlier(x[i])) {
        y[i] = 0.0f;
    } else {
        y[i] = clamp(gain * x[i] + bias, 0.0f, 1.0f);
    }
}
