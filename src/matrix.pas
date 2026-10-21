unit Matrix;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }
{ cuBLAS wrappers assume CUBLAS_POINTER_MODE_HOST }

interface

uses
  { WesChat units }
  Global;

type
  cint = LongInt;
  cublasStatus_t = cint;
  pcint = ^cint;

const
  cublasDLL = 'cublas64_13.dll';
  cudartDLL = 'cudart64_13.dll';
  WesChatKernelDLL = 'weschatkernel12.dll';
  cudaMemcpyHostToHost   = 0;
  cudaMemcpyHostToDevice = 1;
  cudaMemcpyDeviceToHost = 2;
  cudaMemcpyDeviceToDevice = 3;
  CUBLAS_OP_N = 0;
  CUBLAS_OP_T = 1;
  UNITSTRIDE = 1;
  CUBLAS_STATUS_SUCCESS         = 0;
  CUBLAS_STATUS_NOT_INITIALIZED = 1;
  CUBLAS_STATUS_ALLOC_FAILED    = 3;
  CUBLAS_STATUS_INVALID_VALUE   = 7;
  CUBLAS_STATUS_ARCH_MISMATCH   = 8;
  CUBLAS_STATUS_MAPPING_ERROR   = 11;
  CUBLAS_STATUS_EXECUTION_FAILED= 13;
  CUBLAS_STATUS_INTERNAL_ERROR  = 14;

{ CUDA and cuBLAS imports }
function cublasCreate_v2(var handle: TcublasHandle): Integer; cdecl; external cublasDLL;
function cublasDestroy_v2(handle: TcublasHandle): Integer; cdecl; external cublasDLL;
function cublasGetVersion(handle: TcublasHandle; version: pcint): cublasStatus_t; cdecl; external cublasDLL;
function CuBLAS_Init: Boolean;
function CuBLAS_Shutdown: Boolean;
function CuBLAS_Ready: Boolean;
procedure CheckCublasStatus(const Status: cublasStatus_t; const Where: string);
function cudaMalloc(devPtr: PPointer; size: NativeUInt): Integer; cdecl; external cudartDLL;
function cudaMemcpy(dst: Pointer; src: Pointer; count: NativeUInt; kind: LongInt): LongInt; cdecl; external cudartDLL;
function cudaMemset(devPtr: Pointer; value: Integer; count: NativeUInt): Integer; cdecl; external cudartDLL;
function cudaFree(devPtr: Pointer): Integer; cdecl; external cudartDLL;
function cudaDeviceReset: Integer; cdecl; external cudartDLL;
function cudaGetLastError: Integer; cdecl; external cudartDLL;
function cudaGetErrorString(error: Integer): PAnsiChar; cdecl; external cudartDLL;
function cudaDeviceSynchronize: Integer; cdecl; external cudartDLL;

{ Matrix and vector wrappers }
procedure CuMatMulFullNN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullTN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullNT(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
procedure CuMatMulFullAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
procedure CuMatMulFullAccTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
procedure CuMatMulFullScaledNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
procedure CuMatMulNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuAddScaled(handle: TcublasHandle; N: Integer; Alpha: Single; const X: PSingle; Y: PSingle);
procedure CuScale(handle: TcublasHandle; N: Integer; Alpha: Single; X: PSingle);
procedure CuMatAdd(handle: TcublasHandle; const A, B: PSingle; C: PSingle; Rows, Cols: Integer);
procedure CuCopy(handle: TcublasHandle; const Src: PSingle; Dst: PSingle; N: Integer);

{ Gradient operations }
procedure CuGradSplit(handle: TcublasHandle; const Upstream: PSingle; Left, Right: PSingle; Rows, Cols: Integer);
procedure CuAccumulateGrad(handle: TcublasHandle; const Src: PSingle; Dst: PSingle; Rows, Cols: Integer);
procedure CuMatMulAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);

{ Repurposing deimnsion }
procedure LaunchProjectOutDirection(X, V: PSingle; Rows, ModelDim: Integer); cdecl; external;

{ Raw BLAS imports }
// SGEMM.
function cublasSgemm_v2(handle: TcublasHandle;
  transa, transb: Integer;
  m, n, k: Integer;
  const alpha: PSingle;
  A: PSingle; lda: Integer;
  B: PSingle; ldb: Integer;
  const beta: PSingle;
  C: PSingle; ldc: Integer): Integer; cdecl; external cublasDLL;

// SAXPY.
function cublasSaxpy_v2(handle: TcublasHandle;
  n: LongInt;
  const alpha: PSingle;
  const x: PSingle; incx: LongInt;
  y: PSingle; incy: LongInt): LongInt; cdecl; external cublasDLL;

// SCOPY.
function cublasScopy_v2(handle: TcublasHandle;
  n: LongInt;
  const x: PSingle; incx: LongInt;
  y: PSingle; incy: LongInt): LongInt; cdecl; external cublasDLL;

// SSCAL.
function cublasSscal_v2(handle: TcublasHandle;
  n: LongInt;
  const alpha: PSingle;
  x: PSingle;
  incx: LongInt): LongInt; cdecl; external cublasDLL;

implementation

{ cuBLAS lifecycle }
// Initialize the global cuBLAS handle.
function CuBLAS_Init: Boolean;
begin
  if CuHandle = nil then
    Result := cublasCreate_v2(CuHandle) = CUBLAS_STATUS_SUCCESS
  else
    Result := True;
end;

// Destroy the global cuBLAS handle.
function CuBLAS_Shutdown: Boolean;
begin
  if CuHandle <> nil then
  begin
    Result := cublasDestroy_v2(CuHandle) = CUBLAS_STATUS_SUCCESS;
    CuHandle := nil;
  end
  else
    Result := True;
end;

// Return True if the current cuBLAS handle responds.
function CuBLAS_Ready: Boolean;
var
  ver: cint;
begin
  if CuHandle = nil then Exit(False);
  Result := cublasGetVersion(CuHandle, @ver) = CUBLAS_STATUS_SUCCESS;
end;

// Halt on a cuBLAS error.
procedure CheckCublasStatus(const Status: cublasStatus_t; const Where: string);
begin
  if Status <> CUBLAS_STATUS_SUCCESS then begin
    Writeln;
    Writeln('CUBLAS ERROR. Location: ', Where, '; Status: ', Status, '.');
    Halt;
  end;
end;

{ Gradient vector operations }
// Copy one upstream gradient into two residual paths.
procedure CuGradSplit(handle: TcublasHandle; const Upstream: PSingle; Left, Right: PSingle; Rows, Cols: Integer);
var
  n: Integer;
  Status: cublasStatus_t;
begin
  n := Rows * Cols;

  // Left := Upstream.
  Status := cublasScopy_v2(handle, n, Upstream, UNITSTRIDE, Left, UNITSTRIDE);
  CheckCublasStatus(Status, 'CuGradSplit: copy to Left');

  // Right := Upstream.
  Status := cublasScopy_v2(handle, n, Upstream, UNITSTRIDE, Right, UNITSTRIDE);
  CheckCublasStatus(Status, 'CuGradSplit: copy to Right');
end;

// Accumulate Src into Dst.
procedure CuAccumulateGrad(handle: TcublasHandle; const Src: PSingle; Dst: PSingle; Rows, Cols: Integer);
var
  n: Integer;
  alpha: Single;
  Status: cublasStatus_t;
begin
  n := Rows * Cols;
  alpha := 1.0;
  Status := cublasSaxpy_v2(handle, n, @alpha, Src, UNITSTRIDE, Dst, UNITSTRIDE);
  CheckCublasStatus(Status, 'CuAccumulateGrad');
end;

{ Matrix multiplication }
// Full row-major matrix multiplication, C = A * B.
procedure CuMatMulFullNN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
  Status: cublasStatus_t;
begin
  alpha := 1.0;
  beta  := 0.0;

  Status := cublasSgemm_v2(Handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullNN');
end;

// Full row-major matrix multiplication, C = A * B^T.
procedure CuMatMulFullNT(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
  Status: cublasStatus_t;
begin
  alpha := 1.0;
  beta  := 0.0;

  // Row-major C = A * B^T
  // Column-major equivalent: C^T = B * A^T
  Status := cublasSgemm_v2(Handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullNT');
end;

// Full row-major matrix multiplication, C = A^T * B.
procedure CuMatMulFullTN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
  Status: cublasStatus_t;
begin
  alpha := 1.0;
  beta  := 0.0;

  Status := cublasSgemm_v2(Handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullTN');
end;

// Full row-major matrix multiplication, C += A * B.
procedure CuMatMulFullAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer;
  lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
  Status: cublasStatus_t;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  Status := cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullAccNN');
end;

// Full row-major matrix multiplication, C += A * B^T.
procedure CuMatMulFullAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
  Status: cublasStatus_t;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  Status := cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullAccNT');
end;

// Full row-major matrix multiplication, C += A^T * B.
procedure CuMatMulFullAccTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
  Status: cublasStatus_t;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  Status := cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullAccTN');
end;

// Full row-major scaled multiplication, C = Alpha * A * B^T + Beta * C.
procedure CuMatMulFullScaledNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
var
  Status: cublasStatus_t;
begin
  Status := cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, @Alpha,
    B, ldb, A, lda, @Beta, C, ldc);
  CheckCublasStatus(Status, 'CuMatMulFullScaledNT');
end;

// Matrix multiplication, A no transpose, B no transpose, overwrite, row-major.
procedure CuMatMulNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  CuMatMulFullNN(handle, A, B, C, M, N, K, K, N, N);
end;

// Matrix multiplication, A no transpose, B transpose, overwrite, row-major.
procedure CuMatMulNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  CuMatMulFullNT(handle, A, B, C, M, N, K, K, K, N);
end;

// Matrix multiplication, A transpose, B no transpose, overwrite, row-major.
procedure CuMatMulTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  CuMatMulFullTN(handle, A, B, C, M, N, K, M, N, N);
end;

// Matrix multiplication, A no transpose, B no transpose, accumulate, row-major.
procedure CuMatMulAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  CuMatMulFullAccNN(handle, A, B, C, M, N, K, K, N, N);
end;

// Matrix multiplication, A no transpose, B transpose, accumulate, row-major.
procedure CuMatMulAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  CuMatMulFullAccNT(handle, A, B, C, M, N, K, K, K, N);
end;

{ Vector operations }
// Add a scaled vector: Y := Alpha * X + Y.
procedure CuAddScaled(handle: TcublasHandle; N: Integer; Alpha: Single; const X: PSingle; Y: PSingle);
var
  Status: cublasStatus_t;
begin
  Status := cublasSaxpy_v2(handle, N, @Alpha, X, UNITSTRIDE, Y, UNITSTRIDE);
  CheckCublasStatus(Status, 'CuAddScaled');
end;

// Scale a vector in place.
procedure CuScale(handle: TcublasHandle; N: Integer; Alpha: Single; X: PSingle);
var
  Status: cublasStatus_t;
begin
  Status := cublasSscal_v2(handle, N, @Alpha, X, UNITSTRIDE);
  CheckCublasStatus(Status, 'CuScale');
end;

// Add matrices elementwise: C := A + B.
procedure CuMatAdd(handle: TcublasHandle; const A, B: PSingle; C: PSingle; Rows, Cols: Integer);
var
  n: Integer;
  alpha: Single;
  Status: cublasStatus_t;
begin
  n := Rows * Cols;
  alpha := 1.0;

  if C = A then begin
    Status := cublasSaxpy_v2(handle, n, @alpha, B, UNITSTRIDE, C, UNITSTRIDE);
    CheckCublasStatus(Status, 'CuMatAdd: add B to A');
  end
  else if C = B then begin
    Status := cublasSaxpy_v2(handle, n, @alpha, A, UNITSTRIDE, C, UNITSTRIDE);
    CheckCublasStatus(Status, 'CuMatAdd: add A to B');
  end
  else begin
    Status := cublasScopy_v2(handle, n, A, UNITSTRIDE, C, UNITSTRIDE);
    CheckCublasStatus(Status, 'CuMatAdd: copy A to C');

    Status := cublasSaxpy_v2(handle, n, @alpha, B, UNITSTRIDE, C, UNITSTRIDE);
    CheckCublasStatus(Status, 'CuMatAdd: add B to C');
  end;
end;

// Copy a vector with cuBLAS.
procedure CuCopy(handle: TcublasHandle; const Src: PSingle; Dst: PSingle; N: Integer);
var
  Status: cublasStatus_t;
begin
  Status := cublasScopy_v2(handle, N, Src, UnitStride, Dst, UnitStride);
  CheckCublasStatus(Status, 'CuCopy');
end;

end.
