// LayerNorm Forward.
extern "C" __global__
void LayerNormForwardKernel(
    const float* InX,
    float* OutX,
    const float* Gamma,
    const float* Beta,
    float* LNXhat,
    float* LNInvStd,
    int SeqLen,
    int ModelDim,
    float EPS)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_var = shared + blockDim.x;

    float x = 0.0f;

    if (row < SeqLen && tid < ModelDim)
        x = InX[row * ModelDim + tid];

    s_sum[tid] = (tid < ModelDim) ? x : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float mean = s_sum[0] / ModelDim;

    float diff = 0.0f;
    if (tid < ModelDim)
        diff = x - mean;

    s_var[tid] = (tid < ModelDim) ? diff * diff : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_var[tid] += s_var[tid + stride];
        __syncthreads();
    }

    float invStd = rsqrtf((s_var[0] / ModelDim) + EPS);

    if (tid == 0)
        LNInvStd[row] = invStd;

    if (row < SeqLen && tid < ModelDim) {
        float xhat = diff * invStd;
        LNXhat[row * ModelDim + tid] = xhat;
        OutX[row * ModelDim + tid] =
            xhat * Gamma[tid] + Beta[tid];
    }
}

extern "C" __declspec(dllexport)
void LaunchLayerNormForward(
    const float* InX,
    float* OutX,
    const float* Gamma,
    const float* Beta,
    float* LNXhat,
    float* LNInvStd,
    int SeqLen,
    int ModelDim)
{
    int threads = 256;
    int blocks = SeqLen;
    int sharedBytes = 2 * threads * sizeof(float);

    LayerNormForwardKernel<<<blocks, threads, sharedBytes>>>(
        InX,
        OutX,
        Gamma,
        Beta,
        LNXhat,
        LNInvStd,
        SeqLen,
        ModelDim,
        1.0e-5f
    );
}

// LayerNorm Backward.
#include <cuda_runtime.h>

extern "C" __global__
void LayerNormBackwardKernel(
    const float* dY,
    float* dX,
    const float* Gamma,
    const float* LNXhat,
    const float* LNInvStd,
    float* dGamma,
    float* dBeta,
    int SeqLen,
    int ModelDim)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_sum1 = shared;
    float* s_sum2 = shared + blockDim.x;

    float dy = 0.0f;
    float gamma = 0.0f;
    float xhat = 0.0f;
    float dhat = 0.0f;

    int idx = row * ModelDim + tid;

    if (row < SeqLen && tid < ModelDim) {
        dy = dY[idx];
        gamma = Gamma[tid];
        xhat = LNXhat[idx];
        dhat = dy * gamma;

        atomicAdd(&dGamma[tid], dy * xhat);
        atomicAdd(&dBeta[tid], dy);
    }

    s_sum1[tid] = (tid < ModelDim) ? dhat : 0.0f;
    s_sum2[tid] = (tid < ModelDim) ? dhat * xhat : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum1[tid] += s_sum1[tid + stride];
            s_sum2[tid] += s_sum2[tid + stride];
        }
        __syncthreads();
    }

    if (row < SeqLen && tid < ModelDim) {
        float sum1 = s_sum1[0];
        float sum2 = s_sum2[0];
        float scale = LNInvStd[row] / ModelDim;

        dX[idx] = scale * (ModelDim * dhat - sum1 - xhat * sum2);
    }
}

extern "C" __declspec(dllexport)
void LaunchLayerNormBackward(
    const float* dY,
    float* dX,
    const float* Gamma,
    const float* LNXhat,
    const float* LNInvStd,
    float* dGamma,
    float* dBeta,
    int SeqLen,
    int ModelDim)
{
    cudaMemset(dGamma, 0, ModelDim * sizeof(float));
    cudaMemset(dBeta, 0, ModelDim * sizeof(float));

    int threads = 256;
    int blocks = SeqLen;
    int sharedBytes = 2 * threads * sizeof(float);

    LayerNormBackwardKernel<<<blocks, threads, sharedBytes>>>(
        dY, dX, Gamma, LNXhat, LNInvStd, dGamma, dBeta, SeqLen, ModelDim);
}

// RMSNorm Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void RMSNormForwardKernel(
    const float* InX,
    float* OutX,
    const float* Gamma,
    float* RMSXHat,
    float* RMSInvStd,
    int SeqLen,
    int ModelDim,
    float EPS)
{
    extern __shared__ float s_sum[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= SeqLen)
        return;

    const float* XRow = InX + row * ModelDim;
    float* OutRow = OutX + row * ModelDim;
    float* XHatRow = RMSXHat + row * ModelDim;

    // Sum squares for this row.
    float SumSq = 0.0f;

    for (int j = tid; j < ModelDim; j += blockDim.x) {
        float x = XRow[j];
        SumSq += x * x;
    }

    s_sum[tid] = SumSq;
    __syncthreads();

    // Reduce sum of squares.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_sum[tid] += s_sum[tid + stride];

        __syncthreads();
    }

    // Compute inverse RMS.
    if (tid == 0) {
        float MeanSq = s_sum[0] / (float)ModelDim;
        RMSInvStd[row] = rsqrtf(MeanSq + EPS);
    }

    __syncthreads();

    float InvRMS = RMSInvStd[row];

    // Normalize and apply Gamma.
    for (int j = tid; j < ModelDim; j += blockDim.x) {
        float XHat = XRow[j] * InvRMS;

        XHatRow[j] = XHat;
        OutRow[j] = Gamma[j] * XHat;
    }
}

extern "C" __declspec(dllexport)
void LaunchRMSNormForward(
    const float* InX,
    float* OutX,
    const float* Gamma,
    float* RMSXHat,
    float* RMSInvStd,
    int SeqLen,
    int ModelDim)
{
    const float EPS = 1.0e-5f;

    int threads = 256;
    int blocks = SeqLen;
    int SharedBytes = threads * sizeof(float);

    RMSNormForwardKernel<<<blocks, threads, SharedBytes>>>(
        InX,
        OutX,
        Gamma,
        RMSXHat,
        RMSInvStd,
        SeqLen,
        ModelDim,
        EPS
    );
}

// RMSNorm Backward.
#include <cuda_runtime.h>

extern "C" __global__
void RMSNormBackwardKernel(
    const float* dY,
    float* dX,
    const float* Gamma,
    const float* RMSXHat,
    const float* RMSInvStd,
    float* dGamma,
    int SeqLen,
    int ModelDim)
{
    extern __shared__ float s_sum[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= SeqLen)
        return;

    const float* dYRow = dY + row * ModelDim;
    float* dXRow = dX + row * ModelDim;
    const float* XHatRow = RMSXHat + row * ModelDim;

    // Compute sum(G * XHat), where G = dY * Gamma.
    float SumGXHat = 0.0f;

    for (int j = tid; j < ModelDim; j += blockDim.x) {
        float G = dYRow[j] * Gamma[j];
        SumGXHat += G * XHatRow[j];
    }

    s_sum[tid] = SumGXHat;
    __syncthreads();

    // Reduce sum(G * XHat).
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_sum[tid] += s_sum[tid + stride];

        __syncthreads();
    }

    float MeanGXHat = s_sum[0] / (float)ModelDim;
    float InvRMS = RMSInvStd[row];

    // Compute dX and accumulate dGamma.
    for (int j = tid; j < ModelDim; j += blockDim.x) {
        float DY = dYRow[j];
        float XHat = XHatRow[j];
        float G = DY * Gamma[j];

        dXRow[j] = InvRMS * (G - XHat * MeanGXHat);

        atomicAdd(&dGamma[j], DY * XHat);
    }
}

extern "C" __declspec(dllexport)
void LaunchRMSNormBackward(
    const float* dY,
    float* dX,
    const float* Gamma,
    const float* RMSXHat,
    const float* RMSInvStd,
    float* dGamma,
    int SeqLen,
    int ModelDim)
{
    int threads = 256;
    int blocks = SeqLen;
    int SharedBytes = threads * sizeof(float);

    RMSNormBackwardKernel<<<blocks, threads, SharedBytes>>>(
        dY,
        dX,
        Gamma,
        RMSXHat,
        RMSInvStd,
        dGamma,
        SeqLen,
        ModelDim
    );
}

// AutoRegressive Mask.
#include <cuda_runtime.h>

extern "C" __global__
void AutoRegressiveMaskKernel(float* Scores, int SeqLen)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SeqLen && j < SeqLen && j > i) {
        Scores[i * SeqLen + j] = -1.0e30f;
    }
}

extern "C" __declspec(dllexport)
void LaunchAutoRegressiveMask(float* Scores, int SeqLen)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (SeqLen + threads.x - 1) / threads.x,
        (SeqLen + threads.y - 1) / threads.y
    );

    AutoRegressiveMaskKernel<<<blocks, threads>>>(Scores, SeqLen);
}

// AutoRegressive Mask Backward.
#include <cuda_runtime.h>

extern "C" __global__
void AutoRegressiveMaskBackwardKernel(
    float* ScoresGrad,
    int SeqLen)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SeqLen && j < SeqLen && j > i) {
        ScoresGrad[i * SeqLen + j] = 0.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchAutoRegressiveMaskBackward(
    float* ScoresGrad,
    int SeqLen)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (SeqLen + threads.x - 1) / threads.x,
        (SeqLen + threads.y - 1) / threads.y
    );

    AutoRegressiveMaskBackwardKernel<<<blocks, threads>>>(
        ScoresGrad,
        SeqLen
    );
}

// RoPE Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void RoPEForwardKernel(
    float* H,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    if (idx >= TotalPairs)
        return;

    // Decompose the linear index into:
    //   row  = sequence position
    //   head = attention head
    //   pair = adjacent RoPE pair within that head
    int pair = idx % PairsPerHead;

    int temp = idx / PairsPerHead;
    int head = temp % NumHeads;
    int row  = temp / NumHeads;

    float angle = static_cast<float>(row) * InvFreq[pair];
    float c = cosf(angle);
    float s = sinf(angle);

    // H is physically laid out as:
    //
    // H[row, head * HeadDim + dimension]
    //
    // RowStride will normally equal ModelDim.
    int base =
        row  * RowStride +
        head * HeadDim +
        pair * 2;

    float x0 = H[base];
    float x1 = H[base + 1];

    H[base]     = x0 * c - x1 * s;
    H[base + 1] = x0 * s + x1 * c;
}

extern "C" __declspec(dllexport)
void LaunchRoPEForward(
    float* H,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    if (H == nullptr ||
        InvFreq == nullptr ||
        SeqLen <= 0 ||
        NumHeads <= 0 ||
        HeadDim <= 0 ||
        RowStride <= 0 ||
        (HeadDim & 1) != 0)
    {
        return;
    }

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    int Threads = 256;
    int Blocks = (TotalPairs + Threads - 1) / Threads;

    RoPEForwardKernel<<<Blocks, Threads>>>(
        H,
        InvFreq,
        SeqLen,
        NumHeads,
        HeadDim,
        RowStride
    );
}

// RoPE Backward.
extern "C" __global__
void RoPEBackwardKernel(
    float* dH,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    if (idx >= TotalPairs)
        return;

    int pair = idx % PairsPerHead;

    int temp = idx / PairsPerHead;
    int head = temp % NumHeads;
    int row  = temp / NumHeads;

    float angle = static_cast<float>(row) * InvFreq[pair];

    float c = cosf(angle);
    float s = sinf(angle);

    int base =
        row  * RowStride +
        head * HeadDim +
        pair * 2;

    float g0 = dH[base];
    float g1 = dH[base + 1];

    // Backpropagation applies the transpose/inverse rotation:
    //
    // [ dx0 ]   [  cos(a)  sin(a) ] [ dg0 ]
    // [ dx1 ] = [ -sin(a)  cos(a) ] [ dg1 ]
    dH[base]     =  g0 * c + g1 * s;
    dH[base + 1] = -g0 * s + g1 * c;
}

extern "C" __declspec(dllexport)
void LaunchRoPEBackward(
    float* dH,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    if (dH == nullptr ||
        InvFreq == nullptr ||
        SeqLen <= 0 ||
        NumHeads <= 0 ||
        HeadDim <= 0 ||
        RowStride <= 0 ||
        (HeadDim & 1) != 0)
    {
        return;
    }

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    int Threads = 256;
    int Blocks = (TotalPairs + Threads - 1) / Threads;

    RoPEBackwardKernel<<<Blocks, Threads>>>(
        dH,
        InvFreq,
        SeqLen,
        NumHeads,
        HeadDim,
        RowStride
    );
}

// DropOut Forward.
#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__
void DropoutKernel(
    float* X,
    int N,
    float DropProb,
    unsigned long long Seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        curandState state;
        curand_init(Seed, idx, 0, &state);

        float r = curand_uniform(&state);

        if (r < DropProb)
            X[idx] = 0.0f;
        else
            X[idx] = X[idx] / (1.0f - DropProb);
    }
}

extern "C" __declspec(dllexport)
void LaunchDropout(
    float* X,
    int N,
    float DropProb,
    unsigned long long Seed)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    DropoutKernel<<<blocks, threads>>>(X, N, DropProb, Seed);
}

// Dropout Backward.
#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__
void DropoutBackwardKernel(
    float* dX,
    int N,
    float DropProb,
    unsigned long long Seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        curandState state;
        curand_init(Seed, idx, 0, &state);

        float r = curand_uniform(&state);

        if (r < DropProb)
            dX[idx] = 0.0f;
        else
            dX[idx] = dX[idx] / (1.0f - DropProb);
    }
}

extern "C" __declspec(dllexport)
void LaunchDropoutBackward(
    float* dX,
    int N,
    float DropProb,
    unsigned long long Seed)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    DropoutBackwardKernel<<<blocks, threads>>>(
        dX, N, DropProb, Seed);
}


#include <cuda_runtime.h>
#include <curand_kernel.h>

// Strided dropout for a rectangular region of a matrix.
// X:      Entire physical matrix.
// Rows:   Number of rows.
// Cols:   Number of columns receiving dropout.
// Pitch:  Physical number of floats per row.
// Offset: Starting column of dropout region.
//
// The random sequence uses the logical index,
// preserving compatibility with ordinary dropout.

extern "C" __global__
void DropoutStridedKernel(
    float* X,
    int Rows,
    int Cols,
    int Pitch,
    int Offset,
    float DropProb,
    unsigned long long Seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Rows * Cols;

    if (idx < N) {
        // Logical position in the dropout region.
        int row = idx / Cols;
        int col = idx % Cols;

        // Physical position in the matrix.
        int pos = row * Pitch + Offset + col;

        // Generate exactly the same random sequence
        // as the original contiguous dropout kernel.
        curandState state;
        curand_init(Seed, idx, 0, &state);

        float r = curand_uniform(&state);

        if (r < DropProb)
            X[pos] = 0.0f;
        else
            X[pos] = X[pos] / (1.0f - DropProb);
    }
}

// Strided Dropout Forward.
extern "C" __declspec(dllexport)
void LaunchDropoutStrided(
    float* X,
    int Rows,
    int Cols,
    int Pitch,
    int Offset,
    float DropProb,
    unsigned long long Seed)
{
    int N = Rows * Cols;
    if (N <= 0 || DropProb <= 0.0f) return;
    if (Pitch <= 0 || Cols <= 0 || Offset < 0 ||
        Offset + Cols > Pitch || DropProb >= 1.0f) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    DropoutStridedKernel<<<blocks, threads>>>(
        X, Rows, Cols, Pitch, Offset, DropProb, Seed);
}

// Strided Dropout Backward.
// Uses the identical mask and scaling as forward.
extern "C" __declspec(dllexport)
void LaunchDropoutBackwardStrided(
    float* dX,
    int Rows,
    int Cols,
    int Pitch,
    int Offset,
    float DropProb,
    unsigned long long Seed)
{
    int N = Rows * Cols;
    if (N <= 0 || DropProb <= 0.0f) return;
    if (Pitch <= 0 || Cols <= 0 || Offset < 0 ||
        Offset + Cols > Pitch || DropProb >= 1.0f) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    DropoutStridedKernel<<<blocks, threads>>>(
        dX, Rows, Cols, Pitch, Offset, DropProb, Seed);
}

// ReLU Forward.
#include <cuda_runtime.h>

extern "C" __global__
void ReLUForwardKernel(
    const float* A,
    float* B,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];
        B[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchReLUForward(
    const float* A,
    float* B,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ReLUForwardKernel<<<blocks, threads>>>(A, B, N);
}

// ReLU Backward.
#include <cuda_runtime.h>

extern "C" __global__
void ReLUBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        GradIn[idx] = (x > 0.0f) ? GradOut[idx] : 0.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchReLUBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ReLUBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N
    );

}

// GELU Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void GELUForwardKernel(
    const float* A,
    float* B,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];

        const float Sqrt2OverPi = 0.7978845608028654f;
        const float CubicCoeff = 0.044715f;

        float x3 = x * x * x;
        float u = Sqrt2OverPi * (x + CubicCoeff * x3);

        B[idx] = 0.5f * x * (1.0f + tanhf(u));
    }
}

extern "C" __declspec(dllexport)
void LaunchGELUForward(
    const float* A,
    float* B,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    GELUForwardKernel<<<blocks, threads>>>(A, B, N);
}

// GELU Backward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void GELUBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];

        const float Sqrt2OverPi = 0.7978845608028654f;
        const float CubicCoeff = 0.044715f;

        float x2 = x * x;
        float x3 = x2 * x;
        float u = Sqrt2OverPi * (x + CubicCoeff * x3);
        float t = tanhf(u);

        float du_dx = Sqrt2OverPi * (1.0f + 3.0f * CubicCoeff * x2);

        float dGELU =
            0.5f * (1.0f + t) +
            0.5f * x * (1.0f - t * t) * du_dx;

        GradIn[idx] = GradOut[idx] * dGELU;
    }
}

extern "C" __declspec(dllexport)
void LaunchGELUBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    GELUBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N
    );
}

// SiLU Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void SiLUForwardKernel(
    const float* A,
    float* B,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));

        B[idx] = x * sigmoid;
    }
}

extern "C" __declspec(dllexport)
void LaunchSiLUForward(
    const float* A,
    float* B,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SiLUForwardKernel<<<blocks, threads>>>(A, B, N);
}

// SiLU Backward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void SiLUBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));

        float dSiLU =
            sigmoid * (1.0f + x * (1.0f - sigmoid));

        GradIn[idx] = GradOut[idx] * dSiLU;
    }
}

extern "C" __declspec(dllexport)
void LaunchSiLUBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SiLUBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N
    );
}

// Leaky ReLU Forward.
#include <cuda_runtime.h>

extern "C" __global__
void LeakyReLUForwardKernel(
    const float* A,
    float* B,
    int N,
    float Alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];

        B[idx] = (x > 0.0f) ? x : Alpha * x;
    }
}

extern "C" __declspec(dllexport)
void LaunchLeakyReLUForward(
    const float* A,
    float* B,
    int Rows,
    int Cols,
    float Alpha)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    LeakyReLUForwardKernel<<<blocks, threads>>>(
        A,
        B,
        N,
        Alpha
    );
}

// Leaky ReLU Backward.
#include <cuda_runtime.h>

extern "C" __global__
void LeakyReLUBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N,
    float Alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        float dLeakyReLU = (x > 0.0f) ? 1.0f : Alpha;

        GradIn[idx] = GradOut[idx] * dLeakyReLU;
    }
}

extern "C" __declspec(dllexport)
void LaunchLeakyReLUBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols,
    float Alpha)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    LeakyReLUBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N,
        Alpha
    );
}

// ELU Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void ELUForwardKernel(
    const float* A,
    float* B,
    int N,
    float Alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];

        if (x > 0.0f)
            B[idx] = x;
        else
            B[idx] = Alpha * (expf(x) - 1.0f);
    }
}

extern "C" __declspec(dllexport)
void LaunchELUForward(
    const float* A,
    float* B,
    int Rows,
    int Cols,
    float Alpha)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ELUForwardKernel<<<blocks, threads>>>(
        A,
        B,
        N,
        Alpha
    );
}

// ELU Backward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void ELUBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N,
    float Alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        float dELU;

        if (x > 0.0f)
            dELU = 1.0f;
        else
            dELU = Alpha * expf(x);

        GradIn[idx] = GradOut[idx] * dELU;
    }
}

extern "C" __declspec(dllexport)
void LaunchELUBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols,
    float Alpha)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ELUBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N,
        Alpha
    );
}

// Softplus Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void SoftplusForwardKernel(
    const float* A,
    float* B,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];

        if (x > 0.0f)
            B[idx] = x + log1pf(expf(-x));
        else
            B[idx] = log1pf(expf(x));
    }
}

extern "C" __declspec(dllexport)
void LaunchSoftplusForward(
    const float* A,
    float* B,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SoftplusForwardKernel<<<blocks, threads>>>(
        A,
        B,
        N
    );
}

// Softplus Backward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void SoftplusBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        float sigmoid;

        if (x >= 0.0f) {
            float e = expf(-x);
            sigmoid = 1.0f / (1.0f + e);
        }
        else {
            float e = expf(x);
            sigmoid = e / (1.0f + e);
        }

        GradIn[idx] = GradOut[idx] * sigmoid;
    }
}

extern "C" __declspec(dllexport)
void LaunchSoftplusBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SoftplusBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N
    );
}

// Mish Forward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void MishForwardKernel(
    const float* A,
    float* B,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];
        float softplus;

        if (x > 0.0f)
            softplus = x + log1pf(expf(-x));
        else
            softplus = log1pf(expf(x));

        B[idx] = x * tanhf(softplus);
    }
}

extern "C" __declspec(dllexport)
void LaunchMishForward(
    const float* A,
    float* B,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    MishForwardKernel<<<blocks, threads>>>(
        A,
        B,
        N
    );
}

// Mish Backward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void MishBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        float softplus;
        float sigmoid;

        // Numerically stable Softplus.
        if (x > 0.0f)
            softplus = x + log1pf(expf(-x));
        else
            softplus = log1pf(expf(x));

        // Numerically stable sigmoid.
        if (x >= 0.0f) {
            float e = expf(-x);
            sigmoid = 1.0f / (1.0f + e);
        }
        else {
            float e = expf(x);
            sigmoid = e / (1.0f + e);
        }

        float t = tanhf(softplus);

        float dMish =
            t +
            x * (1.0f - t * t) * sigmoid;

        GradIn[idx] = GradOut[idx] * dMish;
    }
}

extern "C" __declspec(dllexport)
void LaunchMishBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    MishBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N
    );
}

// Softmax Forward Strided.
#include <math.h>
#include <float.h>

extern "C" __declspec(dllexport)
__global__ void SoftmaxForwardStridedKernel(
    const float* In,
    float* Out,
    int Rows,
    int Cols,       // logical vocab size: nVocab
    int RowStride,  // physical row stride: DimVocab
    float Temperature)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    if (row >= Rows)
        return;

    if (Temperature <= 0.0f)
        Temperature = 1.0f;

    int base = row * RowStride;

    // 1. Find max logit for numerical stability.
    float localMax = -FLT_MAX;

    for (int col = tid; col < Cols; col += blockDim.x) {
        float x = In[base + col];
        if (x > localMax)
            localMax = x;
    }

    shared[tid] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared[tid + stride] > shared[tid])
                shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    float maxVal = shared[0];

    // 2. Compute exp sum.
    float localSum = 0.0f;

    for (int col = tid; col < Cols; col += blockDim.x) {
        float e = expf((In[base + col] - maxVal) / Temperature);
        localSum += e;
    }

    shared[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    float sumVal = shared[0];

    // 3. Write normalized probabilities.
    for (int col = tid; col < Cols; col += blockDim.x) {
        float e = expf((In[base + col] - maxVal) / Temperature);
        Out[base + col] = e / sumVal;
    }
}

extern "C" __declspec(dllexport)
void LaunchSoftmaxForwardStrided(
    const float* dIn,
    float* dOut,
    int Rows,
    int Cols,
    int RowStride,
    float Temperature)
{
    int threads = 256;
    int blocks = Rows;
    int sharedBytes = threads * sizeof(float);

    SoftmaxForwardStridedKernel<<<blocks, threads, sharedBytes>>>(
        dIn,
        dOut,
        Rows,
        Cols,
        RowStride,
        Temperature);
}

// Softmax Backward Strided.
#include <cuda_runtime.h>

extern "C" __global__
void SoftmaxBackwardStridedKernel(
    const float* Y,
    const float* dY,
    float* dX,
    int Rows,
    int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // 1. local dot = sum(dY * Y)
    float localDot = 0.0f;

    for (int col = tid; col < D; col += blockDim.x) {
        int idx = row * D + col;
        localDot += dY[idx] * Y[idx];
    }

    shared[tid] = localDot;
    __syncthreads();

    // reduce dot
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    float dot = shared[0];

    // 2. dx = y * (dy - dot)
    for (int col = tid; col < D; col += blockDim.x) {
        int idx = row * D + col;
        dX[idx] = Y[idx] * (dY[idx] - dot);
    }
}

extern "C" __declspec(dllexport)
void LaunchSoftmaxBackward(
    const float* Y,
    const float* dY,
    float* dX,
    int Rows,
    int D)
{
    int threads = 256;
    int sharedBytes = threads * sizeof(float);

    SoftmaxBackwardStridedKernel<<<Rows, threads, sharedBytes>>>(
        Y, dY, dX, Rows, D
    );
}

// Cross Entropy Gradient Strided from Probabilities.
extern "C" __declspec(dllexport)
__global__ void CEGradientStridedKernel(
    const float* Probs,
    float* TopGradient,
    const int* TargetTokens,
    int Rows,
    int VocabSize,
    int RowStride,
    float GradScale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = Rows * VocabSize;
    if (idx >= total)
        return;

    int row = idx / VocabSize;
    int col = idx % VocabSize;

    int target = TargetTokens[row];
    int offset = row * RowStride + col;

    // Invalid target row contributes no gradient.
    if (target < 0 || target >= VocabSize) {
        TopGradient[offset] = 0.0f;
        return;
    }

    // Cross-entropy gradient after softmax:
    //
    // dLogits = Probs
    // dLogits[target] -= 1
    //
    // GradScale normally averages the gradient over Rows.
    float g = Probs[offset];

    if (col == target)
        g -= 1.0f;

    TopGradient[offset] = g * GradScale;
}

extern "C" __declspec(dllexport)
void LaunchCEGradientStrided(
    const float* dProbs,
    float* dTopGradient,
    const int* dTargetTokens,
    int Rows,
    int VocabSize,
    int RowStride,
    float GradScale)
{
    int threads = 256;
    int total = Rows * VocabSize;
    int blocks = (total + threads - 1) / threads;

    CEGradientStridedKernel<<<blocks, threads>>>(
        dProbs,
        dTopGradient,
        dTargetTokens,
        Rows,
        VocabSize,
        RowStride,
        GradScale);
}

// Cross Entropy Gradient From Probabilities.
#include <cuda_runtime.h>

extern "C" __global__
void CEGradientSubtractKernel(
    float* TopGradient,
    const int* TargetTokens,
    int SeqLen,
    int nVocab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SeqLen) {
        int target = TargetTokens[i];

        if (target >= 0 && target < nVocab)
            TopGradient[i * nVocab + target] -= 1.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchCEGradient(
    const float* Probs,
    float* TopGradient,
    const int* TargetTokens,
    int SeqLen,
    int nVocab)
{
    cudaMemcpy(
        TopGradient,
        Probs,
        SeqLen * nVocab * sizeof(float),
        cudaMemcpyDeviceToDevice
    );

    int threads = 256;
    int blocks = (SeqLen + threads - 1) / threads;

    CEGradientSubtractKernel<<<blocks, threads>>>(
        TopGradient,
        TargetTokens,
        SeqLen,
        nVocab
    );
}

// Embedding Lookup.
extern "C" __global__
void EmbeddingLookupKernel(
    const float* Embeddings,
    const int* InputTokens,
    float* X,
    int SeqLen,
    int ModelDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SeqLen * ModelDim;

    if (idx < total) {
        int row = idx / ModelDim;
        int col = idx % ModelDim;
        int tok = InputTokens[row];

        X[idx] = Embeddings[tok * ModelDim + col];
    }
}

extern "C" __declspec(dllexport)
void LaunchEmbeddingLookup(
    const float* Embeddings,
    const int* InputTokens,
    float* X,
    int SeqLen,
    int ModelDim)
{
    int threads = 256;
    int blocks = (SeqLen * ModelDim + threads - 1) / threads;

    EmbeddingLookupKernel<<<blocks, threads>>>(
        Embeddings,
        InputTokens,
        X,
        SeqLen,
        ModelDim
    );
}

// Scatter Add.
extern "C" __global__
void AddInputEmbeddingGradKernel(
    const float* XGrad,
    float* EmbGrad,
    const int* InputTokens,
    int SeqLen,
    int ModelDim,
    int nVocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SeqLen * ModelDim;

    if (idx < total) {
        int row = idx / ModelDim;
        int col = idx - row * ModelDim;
        int tok = InputTokens[row];

        if (tok >= 0 && tok < nVocab)
            atomicAdd(&EmbGrad[tok * ModelDim + col], XGrad[idx]);
    }
}

extern "C" __declspec(dllexport)
void LaunchAddInputEmbeddingGrad(
    const float* XGrad,
    float* EmbGrad,
    const int* InputTokens,
    int SeqLen,
    int ModelDim,
    int nVocab)
{
    int threads = 256;
    int blocks = (SeqLen * ModelDim + threads - 1) / threads;

    AddInputEmbeddingGradKernel<<<blocks, threads>>>(
        XGrad, EmbGrad, InputTokens, SeqLen, ModelDim, nVocab
    );
}

// Add Bias Rows Forward.
#include <cuda_runtime.h>

extern "C" __global__
void AddBiasRowsKernel(
    float* X,
    const float* Bias,
    int Rows,
    int Cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Rows * Cols;

    if (idx < total) {
        int col = idx % Cols;
        X[idx] += Bias[col];
    }
}

extern "C" __declspec(dllexport)
void LaunchAddBiasRows(
    float* X,
    const float* Bias,
    int Rows,
    int Cols)
{
    int threads = 256;
    int blocks = (Rows * Cols + threads - 1) / threads;

    AddBiasRowsKernel<<<blocks, threads>>>(
        X,
        Bias,
        Rows,
        Cols
    );
}

// Add Bias Rows Backward.
#include <cuda_runtime.h>

extern "C" __global__
void AddBiasRowsBackwardKernel(
    const float* dX,
    float* dBias,
    int Rows,
    int Cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Rows * Cols;

    if (idx < total) {
        int col = idx % Cols;
        atomicAdd(&dBias[col], dX[idx]);
    }
}

extern "C" __declspec(dllexport)
void LaunchAddBiasRowsBackward(
    const float* dX,
    float* dBias,
    int Rows,
    int Cols)
{
    int threads = 256;
    int blocks = (Rows * Cols + threads - 1) / threads;

    AddBiasRowsBackwardKernel<<<blocks, threads>>>(
        dX,
        dBias,
        Rows,
        Cols
    );
}

// Clip Vector.
extern "C" __declspec(dllexport)
__global__ void ClipVectorKernel(float* X, int N, float Limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float v = X[idx];

    if (v > Limit)
        v = Limit;
    else if (v < -Limit)
        v = -Limit;

    X[idx] = v;
}

extern "C" __declspec(dllexport)
void LaunchClipVector(float* X, int N, float Limit)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ClipVectorKernel<<<blocks, threads>>>(X, N, Limit);
}

// Cross Entopy Loss using Rows.
#include <cuda_runtime.h>
#include <math.h>

__global__ void CELossRows(const float* Probs, const int* Targets, float* RowLoss, int Rows, int VocabSize, int RowStride)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row >= Rows)
        return;

    int Target = Targets[Row];

    if ((Target < 0) || (Target >= VocabSize)) {
        RowLoss[Row] = NAN;
        return;
    }

    float P = Probs[Row * RowStride + Target];

    if (!isfinite(P)) {
        RowLoss[Row] = NAN;
        return;
    }

    if (P < 1.0e-12f)
        P = 1.0e-12f;
    else if (P > 1.0f)
        P = 1.0f;

    RowLoss[Row] = -logf(P);
}

extern "C" __declspec(dllexport) void LaunchCELossRows(const float* Probs, const int* Targets, float* RowLoss, int Rows, int VocabSize, int RowStride)
{
    const int Threads = 256;
    const int Blocks = (Rows + Threads - 1) / Threads;

    CELossRows<<<Blocks, Threads>>>(Probs, Targets, RowLoss, Rows, VocabSize, RowStride);
}

// AdamW Parameter Update.
__global__ void AdamWUpdateKernel(
    float* Param,
    const float* Grad,
    float* M,
    float* V,
    int Count,
    float LearningRate,
    float Beta1,
    float Beta2,
    float Beta1Power,
    float Beta2Power,
    float AdamEpsilon,
    float WeightDecay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Count) {
        float g;
        float m;
        float v;
        float mHat;
        float vHat;
        float p;

        g = Grad[i];

        // Update first moment.
        m = Beta1 * M[i] + (1.0f - Beta1) * g;

        // Update second moment.
        v = Beta2 * V[i] + (1.0f - Beta2) * g * g;

        M[i] = m;
        V[i] = v;

        // Bias correction.
        mHat = m / (1.0f - Beta1Power);
        vHat = v / (1.0f - Beta2Power);

        // Adam update plus decoupled weight decay.
        p = Param[i];

        Param[i] = p -
            LearningRate *
            (mHat / (sqrtf(vHat) + AdamEpsilon) +
             WeightDecay * p);
    }
}

extern "C" __declspec(dllexport)
void LaunchAdamWUpdate(
    float* Param,
    const float* Grad,
    float* M,
    float* V,
    int Count,
    float LearningRate,
    float Beta1,
    float Beta2,
    float Beta1Power,
    float Beta2Power,
    float AdamEpsilon,
    float WeightDecay)
{
    const int Threads = 256;
    int Blocks;

    Blocks = (Count + Threads - 1) / Threads;

    AdamWUpdateKernel<<<Blocks, Threads>>>(
        Param,
        Grad,
        M,
        V,
        Count,
        LearningRate,
        Beta1,
        Beta2,
        Beta1Power,
        Beta2Power,
        AdamEpsilon,
        WeightDecay);
}

// Add A to B, storing the result in B. Not using as of 10/14/2026.
#include <cuda_runtime.h>

extern "C" __global__
void AddKernel(const float* A, float* B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        B[idx] += A[idx];
}

extern "C" __declspec(dllexport)
void LaunchAdd(const float* A, float* B, int N)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    AddKernel<<<blocks, threads>>>(A, B, N);
}

// Hadamard multiplication.
extern "C" __global__
void HadamardKernel(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        C[idx] = A[idx] * B[idx];
}

extern "C" __declspec(dllexport)
void LaunchHadamard(const float* A, const float* B, float* C, int N)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    HadamardKernel<<<blocks, threads>>>(A, B, C, N);
}

// CUDA projection kernel.
extern "C" __global__
void ProjectOutDirectionKernel(float *X, const float *V,
                               int Rows, int ModelDim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= Rows)
        return;

    int base = row * ModelDim;
    float dot = 0.0f;

    // dot = X[row] . V
    for (int d = 0; d < ModelDim; d++)
        dot += X[base + d] * V[d];

    // X[row] = X[row] - dot * V
    for (int d = 0; d < ModelDim; d++)
        X[base + d] -= dot * V[d];
}

extern "C" __declspec(dllexport)
void LaunchProjectOutDirection(float *X, const float *V,
                               int Rows, int ModelDim)
{
    int Threads = 256;
    int Blocks = (Rows + Threads - 1) / Threads;

    ProjectOutDirectionKernel<<<Blocks, Threads>>>(
        X, V, Rows, ModelDim);
}

// SwiGLU Split Forward.
#include <cuda_runtime.h>
#include <math.h>

// Hidden1: Rows x HiddenWidth. Hidden2: Rows x HiddenWidth.
// For HiddenWidth = 768 and GatedDim = 512:
// Hidden1[0..511]   = Gate, Hidden1[512..767] = Value A.
// Hidden2[0..255]   = Value B, Hidden2[256..767] = Gated output.

__global__ void SwiGLUSplitKernel(
    const float* Hidden1,
    float* Hidden2,
    int N,
    int HiddenWidth,
    int GatedDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int row = idx / GatedDim;
        int col = idx % GatedDim;
        int base = row * HiddenWidth;

        int ValueAWidth = HiddenWidth - GatedDim;
        int ValueBWidth = GatedDim - ValueAWidth;

        // Gate is always in Hidden1.
        float gate = Hidden1[base + col];

        // Value is split between Hidden1 and Hidden2.
        float value;

        if (col < ValueAWidth)
            value = Hidden1[base + GatedDim + col];
        else
            value = Hidden2[base + col - ValueAWidth];

        // SwiGLU = SiLU(Gate) * Value.
        float sigmoid = 1.0f / (1.0f + expf(-gate));
        float silu = gate * sigmoid;

        // Store output after Value B in Hidden2.
        Hidden2[base + ValueBWidth + col] = silu * value;
    }
}

extern "C" __declspec(dllexport)
void LaunchSwiGLUSplit(
    const float* Hidden1,
    float* Hidden2,
    int Rows,
    int HiddenWidth,
    int GatedDim)
{
    int N = Rows * GatedDim;
    if (N <= 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SwiGLUSplitKernel<<<blocks, threads>>>(
        Hidden1, Hidden2, N, HiddenWidth, GatedDim);
}


#include <cuda_runtime.h>
#include <math.h>

// Self-Gated SiLU Backward. Hidden1 contains original preactivation.
// GradHidden2 contains incoming gradient. GradHidden1 receives the resulting gradient.

__global__ void SelfGatedSiLUBackwardKernel(
    const float* Hidden1,
    const float* GradHidden2,
    float* GradHidden1,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        float s = 1.0f / (1.0f + expf(-x));

        float derivative = x * s * (2.0f + x * (1.0f - s));

        GradHidden1[idx] = GradHidden2[idx] * derivative;
    }
}

extern "C" __declspec(dllexport)
void LaunchSelfGatedSiLUBackward(
    const float* Hidden1,
    const float* GradHidden2,
    float* GradHidden1,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;
    if (N <= 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SelfGatedSiLUBackwardKernel<<<blocks, threads>>>(
        Hidden1, GradHidden2, GradHidden1, N);
}

// SwiGLU Split Backward.
__global__ void SwiGLUSplitBackwardKernel(
    const float* Hidden1,
    const float* Hidden2,
    float* GradHidden2,
    float* GradHidden1,
    int N,
    int HiddenWidth,
    int GatedDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int row = idx / GatedDim;
        int col = idx % GatedDim;
        int base = row * HiddenWidth;

        int ValueAWidth = HiddenWidth - GatedDim;
        int ValueBWidth = GatedDim - ValueAWidth;

        // Original forward-pass inputs.
        float gate = Hidden1[base + col];

        float value;

        if (col < ValueAWidth)
            value = Hidden1[base + GatedDim + col];
        else
            value = Hidden2[base + col - ValueAWidth];

        // Incoming gradient from the gated output.
        // Dropout backward must already be complete.
        float grad = GradHidden2[base + ValueBWidth + col];

        // SiLU and its derivative.
        float sigmoid = 1.0f / (1.0f + expf(-gate));
        float silu = gate * sigmoid;

        float siluDerivative =
            sigmoid + gate * sigmoid * (1.0f - sigmoid);

        // Product-rule gradients.
        float dGate = grad * value * siluDerivative;
        float dValue = grad * silu;

        // Gate gradient: first 512 columns of Hidden1.
        GradHidden1[base + col] = dGate;

        // Distribute value gradient between the matrices.
        if (col < ValueAWidth)
            GradHidden1[base + GatedDim + col] = dValue;
        else
            GradHidden2[base + col - ValueAWidth] = dValue;
    }
}

extern "C" __declspec(dllexport)
void LaunchSwiGLUSplitBackward(
    const float* Hidden1,
    const float* Hidden2,
    float* GradHidden2,
    float* GradHidden1,
    int Rows,
    int HiddenWidth,
    int GatedDim)
{
    if (Rows <= 0 || HiddenWidth <= 0 || GatedDim <= 0)
        return;

    if (3LL * GatedDim != 2LL * HiddenWidth)
        return; // Prefer reporting an error in production.

    int N = Rows * GatedDim;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    SwiGLUSplitBackwardKernel<<<blocks, threads>>>(
        Hidden1, Hidden2, GradHidden2, GradHidden1,
        N, HiddenWidth, GatedDim);
}