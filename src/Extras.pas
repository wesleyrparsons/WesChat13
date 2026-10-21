unit Extras;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

interface
implementation

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

{ Standard z-score transform: Xstd = X − μσ. The calls can still be optimized.
   μ = mean of the row. σ = standard deviation.}
procedure StandardizeRows(var A: array of Single; M, N: Integer);
var
  r, j: Integer;
  Mean, Std, InvStd: Single;
  Ones: TFVector;
begin
  // Build a vector of ones for mean calculation.
  SetLength(Ones, N);
  for j := 0 to N - 1 do
    Ones[j] := 1.0;

  for r := 0 to M - 1 do begin
    // Pointer to the start of row r.
    // Row r begins at index r * N.
    // Row is contiguous, so stride = 1.
    // 1. Compute mean = (1/N) * sum(row).
    Mean := cblas_sdot(N, @A[r * N], 1, @Ones[0], 1) / N;

    // 2. Subtract mean: row := row - mean.
    cblas_saxpy(N, -Mean, @Ones[0], 1, @A[r * N], 1);

    // 3. Compute std = sqrt(sum((row - mean)^2) / N).
    Std := cblas_snrm2(N, @A[r * N], 1) / Sqrt(N);

    // Avoid divide-by-zero.
    if Std > 1e-12 then begin
      InvStd := 1.0 / Std;

    // 4. Scale row: row := row * (1 / std).
      cblas_sscal(N, InvStd, @A[r * N], 1);
    end;
  end;
end;

// Welford addition algorithm. Not used yet.
procedure WelfordAddition(const X: TSeqMatrix; SeqLen, ModelDim: Integer; out MeanX, VarX: TFVector);
var
  t, d: Integer;
  Count: Int64;
  Entry, Delta, Delta2: Single;
begin // X[t, d]
  // Initialize.
  for d := 0 to ModelDim - 1 do begin
    MeanX[d] := 0.0;
    VarX[d] := 0.0;   // M2 accumulator.
  end;

  Count := 0;

  // Sweep through sequence.
  for t := 0 to SeqLen - 1 do begin
    Inc(Count);

    for d := 0 to ModelDim - 1 do begin
      Entry := X[t, d];
      Delta  := Entry - MeanX[d];
      MeanX[d] := MeanX[d] + Delta / Count;
      Delta2 := Entry - MeanX[d];
      VarX[d] := VarX[d] + Delta * Delta2;
    end;
  end;

  // Finalize variance.
  if Count > 1 then
    for d := 0 to ModelDim - 1 do
      VarX[d] := VarX[d] / (Count - 1)
  else
    for d := 0 to ModelDim - 1 do
      VarX[d] := 0.0;
end;

// Add B1 Vector to X-type matrix.
procedure AddB1Matrix(const A: THiddenMatrix; const B: TSeqVectorProj; var C: THiddenMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do
      C[i, j] := A[i, j] + B[j];
end;

// Add B2 Vector to X-type matrix.
procedure AddB2Matrix(const A: TSeqMatrix; const B: TSeqVector; var C: TSeqMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to SeqLen - 1 do
    for j := 0 to ModelDim - 1 do
      C[i, j] := A[i, j] + B[i];
end;

// Adds A and B matrices to get C matrix, all L x D.
procedure AddLDMatrices(const A, B: TSeqMatrix; L: Integer; var C: TSeqMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to L - 1 do
    for j := 0 to ModelDim - 1 do
      C[i, j] := A[i, j] + B[i, j];
end;

// Q * Wq is L x D, D x D, output K L x D
procedure MatMul(const A: SASHLDArray; B: SASHDDArray; out C: SASHLDArray);
var
  M, K, N: Integer;
  i, j, l: Integer;
  Sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  for i := 0 to M - 1 do begin
    for j := 0 to N - 1 do begin
      sum := 0;
      for l := 0 to K - 1 do
        sum := sum + A[i, l] * B[l, j];
      C[i, j] := Sum;
    end;
  end;
end;

// F87 optimized blocked matrix operations.
procedure BlockedMatMulLLLD(const A: SASHLLArray; B: SASHLDArray; out C: SASHLDArray);
const
  BS = 32;  // block size
var
  M, K, N: Integer;
  i, j, l: Integer;
  ii, jj, kk: Integer;
  sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  //SetLength(C, M, N);

  // initialize C
  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      C[i][j] := 0;

  ii := 0;
  while ii < M do begin
    kk := 0;
    while kk < K do begin
      jj := 0;
      while jj < N do begin
        for i := ii to Min(ii + BS - 1, M - 1) do
          for l := kk to Min(kk + BS - 1, K - 1) do begin
            sum := A[i][l];
            for j := jj to Min(jj + BS - 1, N - 1) do
              C[i][j] := C[i][j] + sum * B[l][j];
          end;

        jj := jj + BS;
      end;
      kk := kk + BS;
    end;
    ii := ii + BS;
  end;
end;

// F87 optimized blocked matrix operations.
procedure BlockedMatMulLDDD(const A: SASHLDArray; B: SASHDDArray; out C: SASHLDArray);
const
  BS = 32;  // block size
var
  M, K, N: Integer;
  i, j, l: Integer;
  ii, jj, kk: Integer;
  sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  //SetLength(C, M, N);

  // initialize C
  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      C[i][j] := 0;

  ii := 0;
  while ii < M do begin
    kk := 0;
    while kk < K do begin
      jj := 0;
      while jj < N do begin
        for i := ii to Min(ii + BS - 1, M - 1) do
          for l := kk to Min(kk + BS - 1, K - 1) do begin
            sum := A[i][l];
            for j := jj to Min(jj + BS - 1, N - 1) do
              C[i][j] := C[i][j] + sum * B[l][j];
          end;

        jj := jj + BS;
      end;
      kk := kk + BS;
    end;
    ii := ii + BS;
  end;
end;

// F87 optimized blocked matrix operations.
procedure BlockedMatMulLDDL(const A: SASHLDArray; B: SASHDLArray; out C: SASHLLArray);
const
  BS = 32;  // block size
var
  M, K, N: Integer;
  i, j, l: Integer;
  ii, jj, kk: Integer;
  sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  //SetLength(C, M, N);

  // initialize C
  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      C[i][j] := 0;

  ii := 0;
  while ii < M do begin
    kk := 0;
    while kk < K do begin
      jj := 0;
      while jj < N do begin
        for i := ii to Min(ii + BS - 1, M - 1) do
          for l := kk to Min(kk + BS - 1, K - 1) do begin
            sum := A[i][l];
            for j := jj to Min(jj + BS - 1, N - 1) do
              C[i][j] := C[i][j] + sum * B[l][j];
          end;

        jj := jj + BS;
      end;
      kk := kk + BS;
    end;
    ii := ii + BS;
  end;
end;

// Trnspose matrix L and D dimenstions.
function TransposeLD(const A: SASHLDArray): SASHDLArray;
var
  M, N: Integer;
  i, j: Integer;
begin
  M := Length(A);
  N := Length(A[0]);

  //SetLength(AT, N, M);

  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      Result[j, i] := A[i, j];
end;

end.

// Copied prior to recoding heads.
{unit Transform;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  Matrix;

const
  Scale: Single = Sqrt(6.0 / (ModelDim + ModelDim));

// Note: Vocab is dimensioned at MaxVocab, but only uses nVocab.
var
  // Not trainable parameters.
  X1, X2, X3, X4, X5,
    X6, X7, X8, U:                TSeqTensor;              // X's at all stages.
  X1Head, X2Head:                 array[0..nHead - 1] of TSeqHeadTensor;  // X partitioned into nHeads.
  X1q, X1v, X1k:                  TSeqTensor;              // X's for Q, K, V.
  X1qHead, X1vHead, X1kHead:      array[0..nHead - 1] of TSeqHeadTensor;  // Above partitioned into nHeads.
  Q, K, V:                        TSeqTensor;              // Q is X*Wq, K is X*Wk, V is X*Wv.
  QHead, KHead, VHead:            array[0..nHead - 1] of TSeqHeadTensor;  // Q, K, V partitioned into nHeads.
  Scores1, Scores2:               TScoresTensor;           // Scores is Q * K'; SoftMax on Scores.
  ScoresHead1, ScoresHead2:       array[0..nHead - 1] of TScoresHeadTensor;    // Scores partitioned into nHeads.
  Hidden1, Hidden2:               THiddenTensor;           // Neural net payer.
  WVocab:                         TVocabWeightTensor;      // ModelDim x MaxVocab.
  Logits, TopGradient:            TSeqVocabMatrix;         // Logit and Gradient.
  // Trainable parameters.
  Wq, Wk, Wv, W0:                 TWeightTensor;
  WqHead, WkHead, WvHead:         array[0..nHead - 1] of TWeightHeadTensor;
  W1:                             TWeightProjTensor;       // Weights.
  W2:                             TWeightProjTensorT;      // Weights.
  b1:                             TSeqVectorProjTensor;    // Biases.
  b2:                             TSeqVectorTensor;        // Biases.
  Gamma1, Beta1, Gamma2, Beta2:   TSeqVectorTensor;        // Weights.
  // Caches.
  LNInvStd1:  TFSVector;          // Caches for Layer-Norm.
  LNXhat1:    TSeqMatrix;
  LNInvStd2:  TFSVector;
  LNXhat2:    TSeqMatrix;
  // Other.
  InvFreq:    TFVector;           // For RoPE.

procedure InitializeTransformer;
procedure RunTransform;

implementation

// Initialize test vector.
procedure InitTestVector(var N: TFSVector);
var
  i: Integer;
begin
for i := 0 to SeqLen - 1 do
  N[i] := 0.0;
end;

// Rotary positional encoding.
// Apply at initialization of transformer.
procedure InitRoPE(var InvFreq: TFVector; HeadDim: Integer);
var
  j: Integer;
begin
  // HeadDim must be even.
  SetLength(InvFreq, HeadDim div 2);
  for j := 0 to (HeadDim div 2) - 1 do
    InvFreq[j] := Exp( - (2.0 * j) / HeadDim * Ln(10000.0) );
end;

// Apply to both Q and K. QHead[h] : [0..SeqLen-1, 0..HeadDim-1]
// Apply before or after head-splitting. But immediately after computing Q and K.
procedure ApplyRoPE(var H: TSeqHeadMatrix;  const InvFreq: TFVector; SeqLen, HeadDim: Integer);
var
  i, j: Integer;
  angle, c, s, x0, x1: Single;
begin
  for i := 0 to SeqLen - 1 do
    for j := 0 to (HeadDim div 2) - 1 do begin
      angle := i * InvFreq[j];
      c := Cos(angle);
      s := Sin(angle);

      // Original pair.
      x0 := H[i, 2 * j];
      x1 := H[i, 2 * j + 1];

      // Rotated pair.
      H[i, 2 * j]   :=  x0 * c - x1 * s;
      H[i, 2 * j + 1] :=  x0 * s + x1 * c;
    end;
end;

{procedure GradientCheckW1;
var
  i, j: Integer;
  eps, orig, loss_plus, loss_minus, numgrad, analytic: Single;
begin
  eps := 1e-4;

  // ForwardPass;
  // BackwardPass;
  // Check below.
  loss_plus := 0.0;
  loss_minus := 0.0;

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      orig := W1.Value[i, j];

      W1.Value[i, j] := orig + eps;
      // loss_plus := ForwardLoss;

      W1.Value[i, j] := orig - eps;
      // loss_minus := ForwardLoss;

      W1.Value[i, j] := orig;

      numgrad := (loss_plus - loss_minus) / (2 * eps);
      analytic := W1.Grad[i, j];

      WriteLn(i, j, ' error= ', Abs(numgrad - analytic));
    end;
end;}

// Xavier-Glorot initialization on W0 matrix.
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDim - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on WHead matrix.
procedure XGUniformWHead(var W: TWeightHeadMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to HeadDim - 1 do
    for j := 0 to HeadDim - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W1 matrix.
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W2 matrix.
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;              // 0..1.
      W[j, i] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on WVocab matrix.
procedure XGUniformWVocab(var W: TVocabWeightMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to nVocab - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Initialize the transformer stage.
procedure InitializeTransformer;
var
  h, j: Integer;
begin
  // InitTestVector(TestVector);
  // Initialize RoPE.
  InitRoPE(InvFreq, HeadDim);

  // Initialize weight matrix W0.
  XGUniformW(W0.Value, ModelDim, ModelDim);

  // Initialize the weights with Xavier-Glorot function.
  XGUniformW(Wq.Value, ModelDim, ModelDim);
  XGUniformW(Wk.Value, ModelDim, ModelDim);
  XGUniformW(Wv.Value, ModelDim, ModelDim);
  for h := 0 to nHead - 1 do begin
    XGUniformWHead(WqHead[h].Value, HeadDim, HeadDim);
    XGUniformWHead(WkHead[h].Value, HeadDim, HeadDim);
    XGUniformWHead(WvHead[h].Value, HeadDim, HeadDim);
  end;

  // Initialize W1 and W2 weight matrices.
  XGUniformW1(W1.Value, ModelDim, ModelDimProj);
  XGUniformW2(W2.Value, ModelDimProj, ModelDim);

  // Initialize WVocab weight matrices.
  XGUniformWVocab(WVocab.Value, ModelDim, nVocab);

  // Initialize b1 and b2.
  FillChar(b1.Value, SizeOf(b1.Value), 0);
  FillChar(b2.Value, SizeOf(b2.Value), 0);

  // Initialize Beta and Gamma, LN 1 and 2, with SD and mean.
  FillChar(Beta1.Value, SizeOf(Beta1.Value), 0);
  FillChar(Beta2.Value, SizeOf(Beta2.Value), 0);
  for j := 0 to ModelDim - 1 do begin
    Gamma1.Value[j] := 1.0;
    Gamma2.Value[j] := 1.0;
  end;
end;

// Zero out all gradients.
procedure ZeroGradients;
var
  h: Integer;
begin
  FillChar(X1.Grad, SizeOf(X1.Grad), 0);
  FillChar(X2.Grad, SizeOf(X2.Grad), 0);
  FillChar(X3.Grad, SizeOf(X3.Grad), 0);
  FillChar(X4.Grad, SizeOf(X4.Grad), 0);
  FillChar(X5.Grad, SizeOf(X5.Grad), 0);
  FillChar(X6.Grad, SizeOf(X6.Grad), 0);
  FillChar(X7.Grad, SizeOf(X7.Grad), 0);
  FillChar(X8.Grad, SizeOf(X8.Grad), 0);
  FillChar(W0.Grad, SizeOf(W0.Grad), 0);
  FillChar(W1.Grad, SizeOf(W1.Grad), 0);
  FillChar(W2.Grad, SizeOf(W2.Grad), 0);
  FillChar(WVocab.Grad, SizeOf(WVocab.Grad), 0);
  FillChar(b1.Grad, SizeOf(b1.Grad), 0);
  FillChar(b2.Grad, SizeOf(b2.Grad), 0);
  FillChar(Gamma1.Grad, SizeOf(Gamma1.Grad), 0);
  FillChar(Gamma2.Grad, SizeOf(Gamma2.Grad), 0);
  FillChar(Beta1.Grad, SizeOf(Beta1.Grad), 0);
  FillChar(Beta2.Grad, SizeOf(Beta2.Grad), 0);
  for h := 0 to nHead - 1 do begin
    FillChar(X1Head[h].Grad, SizeOf(X1Head[h].Grad), 0);
    FillChar(X2Head[h].Grad, SizeOf(X2Head[h].Grad), 0);
    FillChar(WqHead[h].Grad, SizeOf(WqHead[h].Grad), 0);
    FillChar(WkHead[h].Grad, SizeOf(WkHead[h].Grad), 0);
    FillChar(WvHead[h].Grad, SizeOf(WvHead[h].Grad), 0);
    FillChar(X1qHead[h].Grad, SizeOf(X1qHead[h].Grad), 0);
    FillChar(X1kHead[h].Grad, SizeOf(X1kHead[h].Grad), 0);
    FillChar(X1vHead[h].Grad, SizeOf(X1vHead[h].Grad), 0);
    FillChar(QHead[h].Grad, SizeOf(QHead[h].Grad), 0);
    FillChar(KHead[h].Grad, SizeOf(KHead[h].Grad), 0);
    FillChar(VHead[h].Grad, SizeOf(VHead[h].Grad), 0);

  end;
end;

// Simple autoregressive masking.
procedure ApplyAutoregressiveMask(var Scores: TScoresMatrix; const L: Integer);
var
  i, j: Integer;
const
  NEG_INF: Single = -1e30;
begin
  for i := 0 to L - 1 do
    for j := i + 1 to L - 1 do
      Scores[i, j] := NEG_INF;
end;

// Softmax procedure forward.
procedure SoftmaxForward(const x: TFVector; out y: array of Single);
var
  i: Integer;
  MaxVal, SumVal, InvT: Single;
begin
  // Find max for numerical stability.
  InvT := 1.0 / Temperature;
  MaxVal := x[0] * InvT;
  for i := 1 to High(x) do
    if (x[i] * InvT) > MaxVal then
      MaxVal := x[i] * InvT;

  // Compute exp(x - max).
  SumVal := 0;
  for i := 0 to High(x) do begin
    y[i] := Exp((x[i] * InvT) - MaxVal);
    SumVal := SumVal + y[i];
  end;

  // Normalize.
  SumVal := 1.0 / SumVal;
  for i := 0 to High(x) do
    y[i] := y[i] * SumVal;
end;

// Softmax procedure backward.
procedure SoftmaxBackward(const y, dy:  TFVector; out dx: array of Single);
var
  j: Integer;
  dot: Single;
  D: Integer;
begin
  D := Length(y);

  // dot = sum_j dy[j] * y[j].
  dot := 0.0;
  for j := 0 to D - 1 do
    dot := dot + dy[j] * y[j];

  // dx[j] = y[j] * (dy[j] - dot).
  for j := 0 to D - 1 do
    dx[j] := y[j] * (dy[j] - dot);
end;

// Layer-Norm a matrix.
procedure LayerNormForward(const InX: TSeqMatrix; var OutX: TSeqMatrix; SeqLen: Integer;
  const Gamma, Beta: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
var
  i, j: Integer;
  MeanL, VarL, InvStd: Single;
const
  EPS = 1e-5;
begin
  for i := 0 to SeqLen - 1 do begin
    MeanL := 0.0;
    for j := 0 to ModelDim - 1 do
      MeanL := MeanL + InX[i, j];
    MeanL := MeanL / ModelDim;

    VarL := 0.0;
    for j := 0 to ModelDim - 1 do
      VarL := VarL + Sqr(InX[i, j] - MeanL);
    VarL := VarL / ModelDim;

    InvStd := 1.0 / Sqrt(VarL + EPS);

    for j := 0 to ModelDim - 1 do
      OutX[i, j] := (InX[i, j] - MeanL) * InvStd * Gamma[j] + Beta[j];
    LNInvStd[i] := InvStd;
    for j := 0 to ModelDim - 1 do
      LNXhat[i, j] := (InX[i, j] - MeanL) * InvStd;

  end;
end;

// Layer on back propagation. dY is upstream gradient. dX is output gradient.
// dGamma, dBeta are accumulated over all rows.
procedure LayerNormBackward(const dY: TSeqMatrix; var dX: TSeqMatrix; var dGamma, dBeta: TSeqVector;
  SeqLen: Integer; const Gamma: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
var
  i, j: Integer;
  sum1, sum2, scale: Single;
  dHat: TSeqVector;
begin
  for i := 0 to SeqLen - 1 do begin
    // Step 1: dHat = dY * Gamma.
    sum1 := 0.0;
    sum2 := 0.0;
    for j := 0 to ModelDim - 1 do begin
      dHat[j] := dY[i, j] * Gamma[j];
      sum1 := sum1 + dHat[j];
      sum2 := sum2 + dHat[j] * LNXhat[i][j];
    end;

    // Step 2: compute dX.
    scale := LNInvStd[i] / ModelDim;
    for j := 0 to ModelDim - 1 do
      dX[i, j] := scale * (ModelDim * dHat[j] - sum1 - LNXhat[i, j] * sum2);

    // Step 3: accumulate dGamma and dBeta.
    for j := 0 to ModelDim - 1 do begin
      dGamma[j] := dGamma[j] + dY[i, j] * LNXhat[i][j];
      dBeta[j]  := dBeta[j]  + dY[i, j];
    end;
  end;
end;

// Calculate gradient from logits and target. ??Gradient should just be a tseqmatrix
procedure GradientFromProbabilities;
var
  i, v: Integer;
begin
  for i := 0 to SeqLen - 1 do begin
    for v := 0 to nVocab - 1 do
      TopGradient[i, v] := Logits[i, v];
    TopGradient[i, TargetTokens[i]] := Logits[i, TargetTokens[i]] - 1.0;
  end;
end;

// Back propagation addition.
procedure BackpropAdd(const dOut: TSeqMatrix; var dA, dB: TSeqMatrix; const L, D: Integer);
var
  i, j: Integer;
begin
  for i := 0 to L - 1 do
    for j := 0 to D - 1 do begin
      dA[i, j] := dA[i, j] + dOut[i, j];
      dB[i, j] := dB[i, j] + dOut[i, j];
    end;
end;

// Update the weights and biases.
procedure Optimization;
var
  h: Integer;
begin
  // Main attention/output weights.
  cblas_saxpy(ModelDim * ModelDim, -LearningRate, @W0.Grad[0, 0], 1, @W0.Value[0, 0], 1);

  // Per-head weights.
  for h := 0 to nHead - 1 do begin
    cblas_saxpy(HeadDim * HeadDim, -LearningRate,
                @WqHead[h].Grad[0, 0], 1,
                @WqHead[h].Value[0, 0], 1);

    cblas_saxpy(HeadDim * HeadDim, -LearningRate,
                @WkHead[h].Grad[0, 0], 1,
                @WkHead[h].Value[0, 0], 1);

    cblas_saxpy(HeadDim * HeadDim, -LearningRate,
                @WvHead[h].Grad[0, 0], 1,
                @WvHead[h].Value[0, 0], 1);
  end;

  // Feed-forward and vocab projection.
  cblas_saxpy(ModelDim * ModelDimProj, -LearningRate, @W1.Grad[0, 0], 1, @W1.Value[0, 0], 1);
  cblas_saxpy(ModelDimProj * ModelDim, -LearningRate, @W2.Grad[0, 0], 1, @W2.Value[0, 0], 1);
  cblas_saxpy(ModelDim * nVocab, -LearningRate, @WVocab.Grad[0, 0], 1, @WVocab.Value[0, 0], 1);

  // Biases.
  cblas_saxpy(ModelDimProj, -LearningRate, @b1.Grad[0], 1, @b1.Value[0], 1);
  cblas_saxpy(ModelDim, -LearningRate, @b2.Grad[0], 1, @b2.Value[0], 1);

  // LayerNorm parameters.
  cblas_saxpy(ModelDim, -LearningRate, @Gamma1.Grad[0], 1, @Gamma1.Value[0], 1);
  cblas_saxpy(ModelDim, -LearningRate, @Gamma2.Grad[0], 1, @Gamma2.Value[0], 1);
  cblas_saxpy(ModelDim, -LearningRate, @Beta1.Grad[0], 1, @Beta1.Value[0], 1);
  cblas_saxpy(ModelDim, -LearningRate, @Beta2.Grad[0], 1, @Beta2.Value[0], 1);
end;

// Run the transformer.
procedure RunTransform;
var
  h, i, j: Integer;
begin
  // Display entry to transform.
  writeln('Entering Transformer/FFN/Head Output');

  // Zero gradients.
  ZeroGradients;

  // Display X matrix.
  if VerboseTransform then begin
    writeln('Display X, beginning, in transform, before any action.');
    DisplayX(X, G);
    Pause;
  end;

  // BLOCK 0.

  // 1. FORWARD STAGE: ATTENTION.

    // 1A. Layer-Norm. Obtain X1 from X.

    // Layer Norm: Input X. Output X1.
    // Obtain input X from Tokenizer for Transformer stage.
    // Purpose: Normalization.
    // Equation: X1 = LayerNorm(X). X, X1 in R^{L × D}. Gamma1, Beta1 in R^{D}.
    LayerNormForward(X, X1.Value, SeqLen, Gamma1.Value, Beta1.Value, LNXhat1, LNInvStd1);

    // Display X1 matrix.
    if VerboseTransform then begin
      writeln('Display X1, beginning, after layer-norming.');
      DisplayX(X1.Value, B);
      Pause;
    end;

    // 1B. Split. Implicit split into X1 and accumulate into X4.
    //     RoPE. Also apply RoPE here.

    // 1C. Partition X1 into X1Head[1], etc.     DON'T DO THIS RECODING MULTIHEADS.
    // Equation: X1Head[1], etc. := VerticalPartitionX(X1).
    for h := 0 to nHead - 1 do begin
      VerticalPartitionX(X1.Value, h, X1Head[h].Value, SeqLen, HeadDim);
      ApplyRoPE(X1Head[h].Value, InvFreq, SeqLen, HeadDim);
    end;


    // Display X1Head[3] matrix.
    if VerboseTransform then begin
      writeln('Display X1Head[3], sample, before standardizing.');
      DisplayX(X1Head[3].Value, G);
      Pause;
    end;

    // 1D. Multiplication/Overwrite. Obtain QHead, KHead, VHead from X1.

    // Multihead Multiplication/Overwrite: Input X1Head, WqHead. Output QHead.
    // Update--Equation: Q = X1 · Wq. Q in R^{L x D}. X1 in R^{L · D}. Wq in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    for h := 0 to nHead - 1 do
     MatMulNN(@X1Head[h].Value[0, 0], @WqHead[h].Value[0, 0], @QHead[h].Value[0, 0], SeqLen, HeadDim, HeadDim);

    {// Multiplication/Overwrite: Input X1, Wq. Output Q.
    // Equation: Q = X1 · Wq. Q in R^{L x D}. X1 in R^{L · D}. Wq in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    MatMulNN(@X1.Value[0, 0], @Wq.Value[0, 0], @Q.Value[0, 0], SeqLen, ModelDim, ModelDim);}

    // Display QHead[3] matrix.
    if VerboseTransform then begin
      writeln('Display QHead[3], sample, in transform.');
      DisplayX(QHead[3].Value, G);
      Pause;
    end;

    // Multihead Multiplication/Overwrite: Input X1Head, WkHead. Output KHead.
    // Update-
    for h := 0 to nHead - 1 do
      MatMulNN(@X1Head[h].Value[0, 0], @WkHead[h].Value[0, 0], @KHead[h].Value[0, 0], SeqLen, HeadDim, HeadDim);

    // Display KHead[3] matrix.
    if VerboseTransform then begin
      writeln('Display KHead[3], endbeginning, in transform.');
      DisplayX(KHead[3].Value, E);
      Pause;
    end;

    {// Multiplication/Overwrite: Input X1, Wk. Output K.
    // Equation: K = X1 · Wk. K in R^{L x D}. X1 in R^{L · D}. Wk in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    MatMulNN(@X1.Value[0, 0], @Wk.Value[0, 0], @K.Value[0, 0], SeqLen, ModelDim, ModelDim);}

    // Multihead Multiplication/Overwrite: Input X1Head, WvHead. Output VHead.
    // Update--
    for h := 0 to nHead - 1 do
      MatMulNN(@X1Head[h].Value[0, 0], @WvHead[h].Value[0, 0], @VHead[h].Value[0, 0], SeqLen, HeadDim, HeadDim);

    {// Multiplication/Overwrite: Input X1, Wv. Output V.
    // Equation: V = X1 · Wv. V in R^{L x D}. X1 in R^{L · D}. Wv in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    MatMulNN(@X1.Value[0, 0], @Wv.Value[0, 0], @V.Value[0, 0], SeqLen, ModelDim, ModelDim);}

    // 1D. Nil.

    // 1E. Multiplication. Obtain Scores1.

    // Multihead Multiplication/Overwrite: Input QHead, KᵀHead. Output: ScoresHead1.
    // That is, the Queries * Tansposed(Keys) are the attention scores.
    // Update Equation: Scores1 = Q · Kᵀ. Scores1 in R^{L · L}. Q in R^{L x D}. Kᵀ in R^{D x L}. M=SeqLen N=SeqLen K=HeadDim
    for h := 0 to nHead - 1 do
      MatMulNT(@QHead[h].Value[0, 0], @KHead[h].Value[0, 0], @ScoresHead1[h].Value[0,0], SeqLen, SeqLen, HeadDim);

    if VerboseTransform then begin
      writeln('Display ScoresHead1[0], beginning, before standardizing.');
      DisplayX(ScoresHead1[0].Value, B);
      Pause;
    end;
    if VerboseTransform then begin
      writeln('Display ScoresHead1[4], sample, before standardizing.');
      DisplayX(ScoresHead1[4].Value, G);
      Pause;
    end;

    {// Multiplication/Overwrite: Input Q, Kᵀ. Output: Scores1.
    // That is, the Queries * Tansposed(Keys) are the attention scores.
    // Equation: Scores1 = Q · Kᵀ. Scores1 in R^{L · L}. Q in R^{L x D}. Kᵀ in R^{D x L}. M=SeqLen N=SeqLen K=SeqLen
    MatMulNT(@Q.Value[0, 0], @K.Value[0, 0], @Scores1.Value[0,0], SeqLen, SeqLen, ModelDim);}

    // Display Scores1Head1[3] matrix.
    if VerboseTransform then begin
      writeln('Display ScoresHead1[3], beginning, before standardizing.');
      DisplayX(ScoresHead1[3].Value, B);
      Pause;
    end;

    // 1F. Standardize, Mask & Softmax. Obtain Scores2.
    // Standardization: Input ScoresHead1. Output ScoresHead1.
    // Equation: ScoresHead1 = Sqrt(1 / HeadDim). ScoresHead1 in R^{L x L}. Using HeadDim not ModelDim.
    for h := 0 to nHead - 1 do
      cblas_sscal(SeqLen * SeqLen,  1 / Sqrt(HeadDim),  @ScoresHead1[h].Value[0, 0], 1);

    // Masking: Input ScoresHead1. Output ScoresHead1.
    // Equation: ScoresHead1 = Mask(ScoresHead1). ScoresHead1 in R^{L x L}.
    for h := 0 to nHead - 1 do
      ApplyAutoRegressiveMask(ScoresHead1[h].Value, SeqLen);

    // Softmax: Input ScoresHead1. Output ScoresHead2.
    // Equation: ScoresHead2 = Softmax(ScoresHead1). ScoresHead in R^{L x L}.
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        SoftmaxForward(ScoresHead1[h].Value[i], ScoresHead2[h].Value[i]);

    // Display Scores1Head2[1] matrix.
    if VerboseTransform then begin
      writeln('Display ScoresHead2[1], sample, after softmax.');
      DisplayX(ScoresHead2[1].Value, G);
      Pause;
    end;

    // Do attention dropout.
    if Training then
      for h := 0 to nHead - 1 do
        for i := 0 to SeqLen - 1 do
          for j := 0 to SeqLen - 1 do
            if Random < ADropout then
              ScoresHead2[h].Value[i, j] := 0.0
            else
              ScoresHead2[h].Value[i, j] := ScoresHead2[h].Value[i, j] / (1.0 - ADropOut);

    // 1G. Multiplication/Overwrite. Obtain X2Head from ScoresHead2.
    // Scoring: Input ScoresHead2, VHead. Output: X2Head.
    // Equation: X2 = Scores2 · V. X2 in R^{L · D}. Scores2 in R^{L x L}. V in R^{L x D}. M=SeqLen N=ModelDim K=SeqLen
    for h := 0 to nHead - 1 do
      MatMulNN(@ScoresHead2[h].Value[0, 0], @VHead[h].Value[0, 0], @X2Head[h].Value[0, 0], SeqLen, HeadDim, SeqLen);

    // Display X2Head[0] matrix.
    if VerboseTransform then begin
      writeln('Display X2Head[0], sample, after softmax.');
      DisplayX(X2Head[0].Value, G);
      Pause;
    end;

    // Concatenate XHead[1], etc. into X2.
    // Equation: X2 := VerticalConcatX(X1).
    for h := 0 to nHead - 1 do
      VerticalConcatX(X1Head[h].Value, h, X2.Value, SeqLen, HeadDim);

    // Display X2 matrix.
    if VerboseTransform then begin
      writeln('Display X2, beginning, in transform, after Softmax, and concatenation.');
      DisplayX(X2.Value, B);
      Pause;
    end;

    // 1H. Mutiplication/Overwrite. Obtain X3 by weighting X2 by W0.
    // Weighting: Input X2, W0. Output X3.
    // Equation: X3 = X2 · W0. X3 in R^{L · D}. W0 in R^{D x D}. X2 in R^{L x D}.
    MatMulNN(@X2.Value[0, 0], @W0.Value[0, 0], @X3.Value[0, 0], SeqLen, ModelDim, ModelDim);

    // Display X3 matrix.
    if VerboseTransform then begin
      writeln('Display X3, beginning, in transform.');
      DisplayX(X3.Value, B);
      Pause;
    end;

    // 1I. Merge. Obtain X4 from X1 and X3.
    // Merge Addition: Input X1, X3. Output X4.
    // Equation: X4 = X1 + X3. X4 in R^{L · D}. X1 in R^{L · D}. X2 in R^{L x D}.
    MatAdd(X1.Value, X3.Value, X4.Value, SeqLen, ModelDim);

    // Display X4 matrix.
    if VerboseTransform then begin
      writeln('Display X4, sample, in transform, after residual added to X3.');
      DisplayX(X4.Value, G);
      Pause;
    end;

    // 1J. Layer-Norm. Obtain X5 from X4.
    // Layer Norm: Input X4. Output X5.
    // Equation: X5 = LayerNorm(X4). X4 in R^{L × D}. X5 in R^{L × D}. Gamma2, Beta2 in R^{D}.
    LayerNormForward(X4.Value, X5.Value, SeqLen, Gamma2.Value, Beta2.Value, LNXhat2, LNInvStd2);

    // Display X5 matrix.
    if VerboseTransform then begin
      writeln('Display X5, beginning, in transform, before FFN.');
      DisplayX(X5.Value, G);
      Pause;
    end;

      // 2. STAGE FORWARD FFN.

      // 2A. Multiplication/Overwrite. Obtain Hidden 1 from X5 and W1.
      // Expansion: Input X5, W1. Output Hidden1.
      // Equation: Hidden1 = X5 · W1. Hidden1 in R^{L x DB}. X5 in R^{L x D}. W1 in R^{D x DB}.
      MatMulNN(@X5.Value[0, 0], @W1.Value[0, 0], @Hidden1.Value[0, 0], SeqLen, ModelDimProj, ModelDim);

      // 2B. Addition/Accumulate. Obtain Hidden 1 from Hidden1 and b1.
      // Addition: Input Hidden1, b1. Output Hidden1.
      // Equation: Hidden1 = Hidden1 * b1. Hidden1 in R^{L x DB}. b1 in R^{DB}.
      // AddMatVec(@Hidden1.Value, b1.Value, SeqLen, ModelDimProj);
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDimProj,  1.0,  @b1.Value[0], 1,  @Hidden1.Value[i,0], 1);

      // Display Hidden1 matrix.
      if VerboseTransform then begin
        writeln('Display Hidden1, grid, in transform, after adding b1, and before ReLU.');
        DisplayX(Hidden1.Value, G);
        Pause;
      end;

      // Not necessary because ReLU overwrite Hidden1.Value.
      // Copy pre-ReLU Hidden into Hidden2.
      // cblas_scopy(SeqLen * ModelDimProj, @Hidden1.Value, 1, @Hidden2.Value, 1);

      // 2C. ReLU. Obtain Hidden2 from Hidden1.
      // Activation: Input Hidden1. Output Hidden2.
      // Equation: Hidden2 = ReLU(Hidden1).
      ReLUMaskForward(Hidden1.Value, Hidden2.Value);

      // Do attention dropout.
      if Training then
        for i := 0 to SeqLen - 1 do
          for j := 0 to ModelDimProj - 1 do
            if Random < RDropout then
              Hidden2.Value[i, j] := 0.0
            else
              Hidden2.Value[i, j] := Hidden2.Value[i, j] / (1.0 - RDropOut);

      // 2D. Multiplication/Overwrite. Obtain X6 from Hidden2.
      // Contraction: Input Hidden2, W2. Output X6.
      // Equation: X6 = Hidden2 · W2. Hidden2 in R^{L x DB}. W2 in R^{DB x D}. X6 in R^{L x D}.
      MatMulNN(@Hidden2.Value[0, 0], @W2.Value[0, 0], @X6.Value[0, 0], SeqLen, ModelDim, ModelDimProj);

      // 2E. Addition/Accumulation. Obtain X6 from Hidden2 and b2.
      // Addition: Input Hidden2, b2. Output X6.
      // Equation: X6 = X6 + b2. X6 in R^{L x D}. b2 in R^{L x D}.
      //AddMatVec(@X6.Value, @b2.Value, SeqLen, ModelDim);
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDim,  1.0,  @b2.Value[0], 1,  @X6.Value[i,0], 1);

      // Display X6 matrix.
      if VerboseTransform then begin
        writeln('Display X6, beginning, in transform, after contraction.');
        DisplayX(X6.Value, B);
        Pause;
      end;

      // 2F. Addition/Merge. Obtain X7 from X5 and X6.
      // Backprop Merge Addition: Input Residual X6, X5. Output X7.
      // Equation: X7 = X5 + X6. X7 in R^{L · D}. X5 in R^{L · D}. X6 in R^{L x D}.
      MatAdd(X5.Value, X6.Value, X7.Value, SeqLen, ModelDim);

      // Display X7 matrix.
      if VerboseTransform then begin
        writeln('Display X7, beginning, in transform, after residual added to X6.');
        DisplayX(X7.Value, B);
        Pause;
      end;

      // 3. FORWARD HEAD OUTPUT STAGE.

        // 3A. Multiplication/Overwrite. Obtain Logits from X7 and Vocab.
        // Multiplication: Input X7, Vocab. Output Logits.
        // Equation: Logits = X7 · WVocab. Logits in R^{L x nVocab}. X in R^{L x D}.  WVocab in R^{D x nVocab}.
        MatMulNN(@X7.Value[0, 0], @WVocab.Value[0, 0], @Logits[0, 0], SeqLen, nVocab, ModelDim);

        // Display Logits matrix.
        if VerboseTransform then begin
          writeln('Display Logits, beginning, in transform, before softmax.');
          DisplayX(Logits, B);
          Pause;
        end;

        // Display WVocab matrix.
        if VerboseTransform then begin
          writeln('Display WVocab, beginning, in transform, before computing Logit.');
          DisplayX(WVocab.Value, B);
          Pause;
        end;

        // 3B. Softmax. Obtain Logits from Logits.
        // Softmax: Input Logit. Output Logit.
        // Equation: Logit = Softmax(Logit).
        for i := 0 to SeqLen - 1 do
          SoftmaxForward(Logits[i], Logits[i]);

        // Display Logits matrix.
        if VerboseTransform then begin
          writeln('Display Logits, beginning, in transform, after softmax.');
          DisplayX(Logits, B);
          Pause;
        end;

        // 3C. Cross-Entropy Loss. Obtain TopGradient from Logits.
        // Gradient: Input Logits. Output TopGradient.
        // Equation: TopGradient in R^{L x nVocab}. Logits in R^{L x nVocab}.
        GradientFromProbabilities;

        // Display TopGradient matrix.
        if VerboseTransform then begin
          writeln('Display TopGradient, beginning, in transform, after Logit calculation.');
          DisplayX(TopGradient, B);
          Pause;
        end;

      // BACK PROPAGATION. FEED BACKWARD NETWORK.

      // 2F. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.
      // Equation: X7.Grad = TopGradient · WVocabᵀ.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. WVocabᵀ in R^{nVocab x D}.
      writeln('Stage 2F');
      cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, nVocab,  1.0,   // 112 = Transposed.
        @TopGradient[0, 0], MaxVocab,  @WVocab.Value[0, 0], MaxVocab, 0.0,  @X7.Grad[0, 0], ModelDim);

      // Backprop TopGradient modifies/overwrites WVocab: Input X7ᵀ, TopGradient. Output WVocab.Grad.
      // Equation: WVocab.Grad = X7ᵀ · TopGradient. WVocab.Grad in R^{D x nVocab}. X7ᵀ in R^{D x L}. TopGradient in R^{L x nVocab}.
      cblas_sgemm(101, 112, 111,  ModelDim, nVocab, SeqLen,  1.0,  @X7.Value[0,0], ModelDim,
        @TopGradient[0,0], MaxVocab,  1.0,  @WVocab.Grad[0,0], MaxVocab);

      // Backprop Split X7 Grad into X5 and X6: Input X5.Grad, X7.Grad. Output dX.Grad.
      // Equation: X5.Grad = X5.Grad + X7.Grad. All in R^{L x D}.
      GradSplit(X7.Grad, X5.Grad, X6.Grad, SeqLen, ModelDim);

      if VerboseTransform then begin
        writeln('Display X7.Grad, sample, in transform, after stage 2D.');
        DisplayX(X5.Grad, G);
        Pause;
      end;

      // 2E. Backprop Addition/Accumulation. Obtain b2 from X6.
      writeln('Stage 2E');
      // Backprop X6 Grad creates b2 Grad. Input X6.Grad. Output b2.Grad.
      // Equation: b2.Grad = sum of X6.Grad. b2.Grad is R^{L x D}. X6.Grad in R^{L x D}.
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDim,  1.0,  @X6.Grad[i, 0], 1,  @b2.Grad[0], 1);

      // 2D. Backprop Multiplication/Overwrite. Obtain W2 from Hidden2 and X6.
      writeln('Stage 2D');
      // Backprop X6 Grad creates W2 Grad: Input Hidden2ᵀ.Value, X6.Grad. Output W2.Grad.
      // Equation: W2.Grad = Hidden2ᵀ.Value · X6.Grad. W2.Grad is R^{DB x D}. Hidden2ᵀ.Value is R^{DB x L}. X6.Grad in R^{L x D}.
      MatMulTN(@Hidden2.Value, @X6.Grad, @W2.Grad, ModelDimProj, ModelDim, SeqLen);
      //cblas_sgemm(101, 112, 111,  ModelDimProj, ModelDim, SeqLen,  1.0,
        //@Hidden2.Value[0, 0], ModelDimProj,  @X6.Grad[0, 0], ModelDim, 1.0,  @W2.Grad[0, 0], ModelDim);

      // Backprop X6 Grad creates Hidden2 Grad: Input
      // Equation: Hidden2.Grad = X6.Grad * W2ᵀ.Value. X6.Grad in R^{L x D}. W2ᵀ.Value is R^{D x DB}. Hidden2.Grad is R^{L x DB}.
      MatMulNT(@X6.Grad, @W2.Value, @Hidden2.Grad, SeqLen, ModelDimProj, ModelDim);

      // 2C. Backprop ReLU. Obtain Hidden1 from Hidden2.
      writeln('Stage 2C');
      // Backprop BackReLU activation on Hidden: Input Hidden2.Grad. Output Hidden1.Grad.
      // Equation: Hidden1.Grad = ReLUMaskBackward(Hidden2.Grad). Hidden1.Grad is R^{L x DB}. Hidden2.Value is R^{L x DB}.
      for i := 0 to SeqLen - 1 do                        // Where is ReLUMaskBackward?????????????????
        for j := 0 to ModelDimProj - 1 do
          if Hidden1.Value[i, j] > 0.0 then
            Hidden1.Grad[i, j] := Hidden2.Grad[i, j]
          else
            Hidden1.Grad[i, j] := 0.0;

      // 2B. Backprop Addition/Accumulate. Obtain b1 from Hidden1.
      writeln('Stage 2B');
      // Backprop Hidden Grad creates b1 Grad: Input Hidden1.Grad. Output b1.Grad.
      // Equation: b1.Grad = sum of Hidden1.Grad. b1.Grad is R^{L x DB}. Hidden1.Grad in R^{L x DB}.
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDimProj,  1.0,  @Hidden1.Grad[i, 0], 1,  @b1.Grad[0], 1);

      // 2A. Backprop Multiplication/Overwrite. Obtain W1 from X5ᵀ and Hidden1.
      //     Obtain X5 from Hidden1 and W1.
      writeln('Stage 2A');
      // Backprop Hidden1 Grad creates W1 Grad. Input: X5ᵀ.Value, Hidden1.Grad. Output: W1.Grad.
      // Equation: W1.Grad = X5ᵀ.Value · Hidden1.Grad. W1.Grad is R^{D x DB}. X5ᵀ.Value is R^{L x D}. Hidden1.Grad is R^{D x DB).
      MatMulTN(@X5.Value, @Hidden1.Grad, @W1.Grad, SeqLen, ModelDimProj, ModelDim);
      //cblas_sgemm(101, 112, 111,  ModelDim, ModelDimProj, SeqLen,  1.0,
        //@X5[0, 0], ModelDim,  @dHidden1[0, 0], ModelDimProj, 1.0,  @dW1[0, 0], ModelDimProj);

      // Backprop Hidden1 Grad accumulates into X5 Grad. Input: Hidden1.Grad, W1ᵀ.Value. Output: X5.Grad.
      // Equation: X5.Grad = Hidden1.Grad · W1ᵀ.Value. Hidden1.Grad is R^{D x DB). W1ᵀ.Value is R^{DB x D}. X5.Grad is R^{L x D}.
      MatMulAccNT(@Hidden1.Grad, @W1.Value, @X5.Grad, SeqLen, ModelDim, ModelDimProj);
      //cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, ModelDimProj,  1.0,
        //@dHidden1[0,0], ModelDimProj,  @W1[0,0], ModelDimProj,  1.0,  @dX5[0,0], ModelDim);

    // 1. BACKPROP STAGE TRANSFORMER.

    // 1J. Backprop Layer-Norm. Obtain X5 from X4.
    writeln('Stage 1J');
    // Backprop Layer-Norm: Input X5, dX5. Output X4.Grad, Gamma2.Grad, Beta2.Grad.
    // Equation: X4.Grad, Gamma2.Grad, Beta2.Grad = LayerNorm(X5, X5.Grad, Gamma2, Beta2). X4.Grad, X5.Grad in R^{L x D}. Gamma2.Grad, Beta2.Grad in R^{D}.
    LayerNormBackward(X5.Grad, X4.Grad, Gamma2.Grad, Beta2.Grad, SeqLen, Gamma2.Value, LNXhat2, LNInvStd2);

    if VerboseTransform then begin
      writeln('Display X4.Grad, sample, in transform, after stage 1J, layer-norm.');
      DisplayX(X4.Grad, G);
      Pause;
    end;

    // 1I. Backprop Split. Input: X1.Grad. Output: X3.Grad. Output X4.Grad,
    writeln('Stage 1I');
    // Equation: X3.Grad, X4.Grad = X1.Grad. All in R^{L x D}.
    //cblas_saxpy(SeqLen * ModelDim,  1.0,  @dX4[0, 0], 1,  @dX1[0, 0], 1);
      //cblas_saxpy(SeqLen * ModelDim,  1.0,  @dX4[0, 0], 1,  @dX3[0, 0], 1);
    GradSplit(X4.Grad, X1.Grad, X3.Grad, SeqLen, ModelDim);
    { Example for i := 0 to SeqLen - 1 do
    cblas_saxpy(ModelDim, 1.0, @dX7[i,0], 1, @dX6[i,0], 1);}

    // Guide: To find the change for the weights: dW0 = X6ᵀ ·  dX7.
    //        To find the error for the input: dX6 = dX7  · W0ᵀ.
    // Guide: To find the change for the multiplication: dScores = dX2 · Vᵀ.
    //        To find the error for the input: dV = Sᵀ · dX2.

    // 1H. Backprop Mutiplication/Overwrite. Obtain W0 Grad from X3 Grad: Input: X2ᵀ.Value, X3.Grad. Output: W0.Grad.
    // Equations: W0.Grad = X2ᵀ.Value · X3.Grad. W0.Grad is R^{L x D}. X3.Grad is R^{L x D}.
    //cblas_sgemm(101, 112, 111,  ModelDim, ModelDim, SeqLen,  1.0,
      //@X2[0, 0], ModelDim,  @dX3[0, 0], ModelDim, 1.0,  @dW0[0, 0], ModelDim);
    MatMulTN(@X2.Value, @X3.Grad, @W0.Grad, ModelDim, SeqLen, ModelDim);

    // Backprop Create X2 Grad from X3 Grad: Input: X3.Grad, W0ᵀ.Value. Output: X2.Grad.
    // Equations: X2.Grad = X3.Grad · W0ᵀ. W0.Grad is R^{L x D}. X2.Grad, X3.Grad is R^{L x D}. W0ᵀ.Value is R^{D x L}.
    MatMulNT(@X3.Grad, @W0.Value, @X2.Grad, SeqLen, ModelDim, ModelDim);
    //cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, ModelDim,  1.0,
      //@X3.Grad[0, 0], ModelDim,  @W0.Value[0, 0], ModelDim, 0.0,  @X2.Grad[0, 0], ModelDim);

    if VerboseTransform then begin
      writeln('Display X3.Grad, grid, in transform, before stage 1G, obtain scores2.');
      DisplayX(X3.Grad, G);
      Pause;
    end;

    // 1G. Backprop Multiplication/Overwrite. Obtain Scores2.Grad from X2.Grad: Input X2.Grad, Vᵀ.Value. Output: Scores2.Grad.
    writeln('Stage 1G');
    // Equations: Scores2.Grad = X2.Grad · Vᵀ.Value. Scores2.Grad is R^{L x L}. X2.Grad is R^{L x D}. Vᵀ.Value is R^{D x L}.
    MatMulNT(@X2.Grad, @V.Value, @Scores2.Grad, SeqLen, ModelDim, SeqLen);
    //cblas_sgemm(101, 111, 112,  SeqLen, SeqLen, ModelDim,  1.0,     // 112 = Transposed.
    //@X2.Grad[0, 0], ModelDim,  @V.Value[0, 0], ModelDim,  0.0,  @Scores2.Grad[0, 0], SeqLen);

    // Partition X2 into X1Head[1], etc.
    // Equation: X1Head[2], etc. := VerticalPartitionX(X2).
    for h := 0 to nHead - 1 do
      VerticalPartitionX(X2.Value, h, X1Head[h].Value, SeqLen, HeadDim);

    // Backprop Create VHead Grad from X2Head Grad: Input ScoresHead2ᵀ.Value, X2Head.Grad. Output: VHead.Grad.
    // Equations: VHead.Grad = ScoresHead2ᵀ.Value · X2Head.Grad. VHead.Grad is R^{L x D}. ScoresHead2ᵀ.Value is R^{L x L}. X2Head.Grad is R^{L x D}.
    for h := 0 to nHead - 1 do
      MatMulTN(@ScoresHead2[h].Value, @X2Head[h].Grad, @VHead[h].Grad, SeqLen, SeqLen, HeadDim);
    //cblas_sgemm(101, 112, 111,  SeqLen, ModelDim, SeqLen,  1.0,
      //@Scores2.Value[0, 0], SeqLen,  @X2.Grad[0, 0], ModelDim,  0.0,  @V.Grad[0, 0], ModelDim);

    // 1F. Backprop Standardize, Mask & Softmax. Obtain ScoresHead1.
    writeln('Stage 1F');
    // Insure ScoresHead1.Grad is empty.
    for h := 0 to nHead - 1 do
      FillChar(ScoresHead1[h].Grad, SizeOf(ScoresHead1[h].Grad), 0);
    // Backprop Softmax: Input ScoresHead2.Value ScoresHead2.Grad. Output ScoresHead1.Grad.
    // Equation: ScoresHead1.Grad = SoftMaxBackwards(ScoresHead2.Value, ScoresHead2.Grad).
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        SoftmaxBackward(ScoresHead2[h].Value[i], ScoresHead2[h].Grad[i], ScoresHead1[h].Grad[i]);

    // Backprop AutoRegression.
    // Equation: ScoresHead1.Grad = Unmask(ScoresHead1.Grad).
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        for j := i + 1 to SeqLen - 1 do
          ScoresHead1[h].Grad[i, j] := 0.0;

    // Backprop standardization. Input: ScoresHead1.Grad. Output: ScoresHead1.Grad.
    // Equation: ScoresHead1.Grad = Sqrt(1 / ModelDim). ScoresHead1.Grad in R^{L x L}.
    for h := 0 to nHead - 1 do
      cblas_sscal(SeqLen * SeqLen,  1 / SqrtHD,  @ScoresHead1[h].Grad[0, 0], 1);

    if VerboseTransform then begin
      writeln('ScoresHead1[0].Grad, grid, in transform, before stage 1E, Q and K-transform.');
      DisplayX(ScoresHead1[0].Grad, G);
      Pause;
    end;

    // 1E. Backprop multiplication. Obtain QHead.Grad and KHead.Grad.
    writeln('Stage 1E');
    {QHead.Grad = ScoresHead1.Grad · KHead.Value
     KHead.Grad = QHeadᵀ.Value · ScoresHead1.Grad}
    // Backprop Multiplication: Input ScoresHead1.Grad, KHead.Value. Output QHead.Grad.
    // Equation: QHead.Grad = ScoresHead1.Grad · KHead.Value. QHead.Grad, ScoresHead1.Grad in R^{L x L}. KHead.Value in R^{L x D}.
    for h := 0 to nHead - 1 do
      MatMulNN(@ScoresHead1[h].Grad, @KHead[h].Value, @QHead[h].Grad, SeqLen, HeadDim, SeqLen);
    //cblas_sgemm(101, 111, 111,  SeqLen, ModelDim, SeqLen,  1.0,
      //@Scores1.Grad[0, 0], SeqLen,  @K.Value[0, 0], ModelDim,  0.0,  @Q.Grad[0, 0], ModelDim);

    // Backprop Multiplication: Input ScoresHead1.Gradᵀ, QHead.Value. Output KHead.Grad.
    // Equation: KHead.Grad = ScoresHead1.Gradᵀ · QHead.Value. KHead.Grad in R^{L x D}. ScoresHead1.Gradᵀ in R^{L · L}. QHead.Value in R^{L x D}.
    for h := 0 to nHead - 1 do
      MatMulTN(@ScoresHead1[h].Grad, @QHead[h].Value, @KHead[h].Grad, SeqLen, HeadDim, SeqLen);
    //cblas_sgemm(101, 112, 111,  SeqLen, ModelDim, SeqLen,  1.0,
      //@Scores1.Grad[0, 0], SeqLen,  @Q.Value[0, 0], ModelDim,  0.0,  @K.Grad[0, 0], ModelDim);

    // 1D. Backprop multiplication/overwrite. Obtain W_.Grad and X1_q.Grad for q, k, v.
    writeln('Stage 1D');
    // Obtain X1qHead, X1kHead, X1vHead, from X1Head.
    {WqHead.Grad = X1Headᵀ.Value · QHead.Grad
     X1qHead.Grad = QHead.Grad · WqHeadᵀ.Value}
    // Backprop Create WqHead Grad from QHead Grad: Input X1Headᵀ.Value · QHead.Grad. Output WqHead.Grad.
    // Equation: WqHead.Grad = X1ᵀ · QHead.Grad. WqHead.Grad in R^{D x D}. X1Headᵀ in R^{D x L}. QHead.Grad in R^{L x D}.
    for h := 0 to nHead - 1 do
      MatMulTN(@X1Head[h].Value, @QHead[h].Grad, @WqHead[h].Grad, HeadDim, HeadDim, SeqLen);
    //cblas_sgemm(101, 112, 111,  ModelDim, ModelDim, SeqLen,  1.0,
      //@X1[0, 0], ModelDim,  @Q.Grad[0, 0], ModelDim, 1.0,  @Wq.Grad[0, 0], ModelDim);

    // Backprop Create X1qHead from QHead Grad: Input QHead.Grad, WqHeadᵀ.Value. Output X1qHead.Grad.
    // Equation: X1qHead.Grad = QHead.Grad · WqHeadᵀ. X1qHead.Grad in R^{L x D}. QHead.Grad in R^{L x D}. WqHeadᵀ.Value in R^{D · D}.
    for h := 0 to nHead - 1 do
      MatMulNT(@QHead[h].Grad, @WqHead[h].Value, @X1qHead[h].Grad, SeqLen, HeadDim, HeadDim);
    //cblas_sgemm(101, 111, 112,  SeqLen, HeadDim, HeadDim,  1.0,
      //@Q.Grad[0, 0], ModelDim,  @Wq[0, 0], ModelDim, 1.0,  @X1q.Grad[0, 0], ModelDim);

    {WHeadk.Grad = X1Headᵀ.Value · KHead.Grad
     X1Headk.Grad = KHead.Grad · WHeadkᵀ.Value}
    // Backprop Create WkHead Grad from KHead Grad: Input X1Headᵀ.Value · KHead.Grad. Output WkHead.Grad.
    // Equation:  WkHead.Grad = X1Headᵀ.Value · KHead.Grad. WkHead.Grad in R^{D x D}. X1Headᵀ.Value in R^{D x L}. KHead.Grad in R^{L x D}.
    for h := 0 to nHead - 1 do
      MatMulTN(@X1Head[h].Value, @KHead[h].Grad, @WkHead[h].Grad[h], HeadDim, HeadDim, SeqLen);
    //cblas_sgemm(101, 112, 111,  ModelDim, ModelDim, SeqLen,  1.0,
      //@X1.Value[0, 0], ModelDim,  @K.Grad[0, 0], ModelDim, 1.0,  @Wk.Grad[0, 0], ModelDim);

    // Backprop Create X1kHeadk Grad from KHead Grad. Input KHead.Grad, WkHeadᵀ.Value. Output X1kHead.Grad.
    // Equation: X1kHead.Grad = KHead.Grad · WkHeadᵀ.Value. X1kHead.Grad in R^{L x D}. KHead.Grad in R^{L x D}. WkHeadᵀ.Value in R^{D · D}.
    for h := 0 to nHead - 1 do
      MatMulNT(@KHead[h].Grad, @WkHead[h].Value, @X1kHead[h].Grad, SeqLen, HeadDim, HeadDim);
    //cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, ModelDim,  1.0,
      //@K.Grad[0, 0], ModelDim,  @Wk.Value[0, 0], ModelDim, 1.0,  @X1k.Grad[0, 0], ModelDim);

    {WvHead.Grad = X1Headᵀ · VHead.Grad
     X1vHead.Grad = VHead.Grad · WvHeadᵀ.Value}
    // Backprop Create WvHead Grad from VHead Grad: Input X1Headᵀ.Value · VHead.Grad. Output WvHead.Grad.
    // Equation: WvHead.Grad = X1Headᵀ.Value · V.Grad. WvHead.Grad in R^{D x D}. X1Headᵀ in R^{D x L}. VHead.Grad in R^{L x D}.
    for h := 0 to nHead - 1 do
      MatMulTN(@X1Head[h].Value, @VHead[h].Value, @WvHead[h].Grad, HeadDim, HeadDim, SeqLen);
    //cblas_sgemm(101, 112, 111,  ModelDim, ModelDim, SeqLen,  1.0,
      //@X1.Value[0, 0], ModelDim,  @V.Grad[0, 0], ModelDim, 1.0,  @Wv.Grad[0, 0], ModelDim);

    // Backprop Create X1vHead Grad from VHead Grad. Input VHead.Grad, WvHeadᵀ. Value. Output X1vHead.Grad.
    // Equation: X1vHead.Grad = VHead.Grad times WvHeadᵀ.Value. X1vHead.Grad = V.Grad · WVHeadᵀ.Value. VHead.Grad in R^{L x D}. WvHeadᵀ.Value in R^{D · D}.
    for h := 0 to nHead - 1 do
      MatMulNT(@VHead[h].Grad, @WvHead[h].Value, @X1vHead[h].Grad, SeqLen, HeadDim, HeadDim);
    //cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, ModelDim,  1.0,
      //@V.Grad[0, 0], ModelDim,  @Wv.Value[0, 0], ModelDim, 1.0,  @X1v.Grad[0, 0], ModelDim);

    // 1C. Backprop Concatenate. Concat X1Head[1], etc. into X1.
    // Equation: X1, etc. := VerticalConcatX(X1Head).
    for h := 0 to nHead - 1do
      VerticalConcatX( X1Head[h].Value, h, X1.Value, SeqLen, HeadDim);

    if VerboseTransform then begin
      writeln('Display X1.Grad, grid, in transform, after concatenation.');
      DisplayX(X1.Grad, G);
      Pause;
    end;

    // 1B. Backprop Merge: Obtain X1 Grad as sum of Grads. Input X1q.Grad, X1k.Grad, and X1v.Grad. Output X1.Grad.
    writeln('Stage 1B');
    // Equation:  X1.Grad = X1q.Grad + X1k.Grad + X1v.Grad. All in R^{L x D}.
    for i := 0 to SeqLen - 1 do
      for j := 0 to ModelDim - 1 do
        X1.Grad[i, j] := X1q.Grad[i, j] + X1k.Grad[i, j] + X1v.Grad[i, j];

    // Backprop Accumulate: Input X1.Grad, X4.Grad. Output X1.Grad.
    // Equation: X1.Grad = X1.Grad + X4.Grad. All R^{L x D}.
    AccumulateGrad(X4.Grad, X1.Grad, SeqLen, ModelDim);

    // 1A. Backprop Layer-Norm: Input X1.Value, X1.Grad. Output X.Grad, Gamma1.Grad, Beta1.Grad.
    writeln('Stage 1A');
    // Equation: X1.Grad, Gamma1.Grad, Beta1.Grad = LayerNorm(X1.Value, X1.Grad, Gamma1.Value, Beta1.Value). X.Grad, X1.Grad in R^{L x D}. Gamma1.Grad, Beta1.Grad in R^{D}.
    LayerNormBackward(X1.Grad, X1.Grad, Gamma1.Grad, Beta1.Grad, SeqLen, Gamma1.Value, LNXhat1, LNInvStd1);
                  //X1 or X Output grad above?
    if VerboseTransform then begin
      writeln('Display X1.Grad, grid, in transform, at end.');
      DisplayX(X1.Grad, G);
      Pause;
    end;

  // Modify weights and biases.
  Optimization;

  // Place X1 in X for next block.
  CopyXMatrix(X1.Value, X, SeqLen, ModelDim);
  If VerboseTransform then begin
    writeln('End of tranformer block.');
    Pause;
  end;
end;

end.

end.
{
// BACK PROPAGATION. FEED BACKWARD NETWORK.

// 2F. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.
Writeln('            Transform Backprop Stage 2F');

// Equation: X7.Grad = TopGradient · Embeddings.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. Embeddings.Value in R^{nVocab x D}.
MatMulFullNN(@TopGradient[0, 0], @Embeddings.Value[0, 0], @X7.Grad[0, 0], SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);
{cblas_sgemm(101, 111, 111, SeqLen, ModelDim, nVocab, 1.0, @TopGradient[0, 0], DimVocab,
@Embeddings.Value[0, 0], ModelDim, 0.0, @X7.Grad[0, 0], ModelDim);}

Writeln('Finished MatMul X7.Grad loop.');

// Backprop TopGradient modifies/overwrites Embeddingsᵀ: Input X7ᵀ, TopGradient. Output Embeddingsᵀ.Grad.
// Equation: Embeddingsᵀ.Grad = X7ᵀ · TopGradient. Embeddingsᵀ.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L}. TopGradient in R^{L x nVocab}.
// Problem here was I had NT rather than TN.
MatMulFullAccTN(@TopGradient[0,0], @X7.Value[0,0], @Embeddings.Grad[0,0], nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);
Writeln('Finished Embeddings.Grad GEMM.');

// Backprop Split X7 Grad into X5 and X6: Input X5.Grad, X7.Grad. Output dX.Grad.
// Equation: X5.Grad = X5.Grad + X7.Grad. All in R^{L x D}.
GradSplit(X7.Grad, X5.Grad, X6.Grad, SeqLen, ModelDim);

// Display X7.Grad matrix.
VTPDisplayX('Display X7.Grad, in transform, after stage 2D.', X7.Grad, G);

// 2E. Backprop Addition/Accumulation. Obtain b2 from X6.
Writeln('            Transform Backprop Stage 2E');

// Backprop X6 Grad creates b2 Grad. Input X6.Grad. Output b2.Grad.
// Equation: b2.Grad = sum of X6.Grad. b2.Grad is R^{L x D}. X6.Grad in R^{L x D}.
for i := 0 to SeqLen - 1 do
  AddScaled(ModelDim, 1.0, @X6.Grad[i,0], @b2.Grad[0]);

// 2D. Backprop Multiplication/Overwrite. Obtain W2 from Hidden2 and X6.
Writeln('            Transform Backprop Stage 2D');

// Backprop X6 Grad creates W2 Grad: Input Hidden2ᵀ.Value, X6.Grad. Output W2.Grad.
// Equation: W2.Grad = Hidden2ᵀ.Value · X6.Grad. W2.Grad is R^{DB x D}. Hidden2ᵀ.Value is R^{DB x L}. X6.Grad in R^{L x D}.
MatMulTN(@Hidden2.Value, @X6.Grad, @W2.Grad, ModelDimProj, ModelDim, SeqLen);

// Backprop X6 Grad creates Hidden2 Grad: Input
// Equation: Hidden2.Grad = X6.Grad * W2ᵀ.Value. X6.Grad in R^{L x D}. W2ᵀ.Value is R^{D x DB}. Hidden2.Grad is R^{L x DB}.
MatMulNT(@X6.Grad, @W2.Value, @Hidden2.Grad, SeqLen, ModelDimProj, ModelDim);

// 2C. Backprop ReLU. Obtain Hidden1 from Hidden2.
Writeln('            Transform Backprop Stage 2C');

// Backprop BackReLU activation on Hidden: Input Hidden2.Grad. Output Hidden1.Grad.
// Equation: Hidden1.Grad = ReLUMaskBackward(Hidden2.Grad). Hidden1.Grad is R^{L x DB}. Hidden2.Value is R^{L x DB}.
for i := 0 to SeqLen - 1 do
  for j := 0 to ModelDimProj - 1 do
    if Hidden1.Value[i, j] > 0.0 then
      Hidden1.Grad[i, j] := Hidden2.Grad[i, j]
    else
      Hidden1.Grad[i, j] := 0.0;

// 2B. Backprop Addition/Accumulate. Obtain b1 from Hidden1.
Writeln('            Transform Backprop Stage 2B');

// Backprop Hidden Grad creates b1 Grad: Input Hidden1.Grad. Output b1.Grad.
// Equation: b1.Grad = sum of Hidden1.Grad. b1.Grad is R^{L x DB}. Hidden1.Grad in R^{L x DB}.
for i := 0 to SeqLen - 1 do
  AddScaled(ModelDimProj, 1.0, @Hidden1.Grad[i,0], @b1.Grad[0]);

// 2A. Backprop Multiplication/Overwrite. Obtain W1 from X5ᵀ and Hidden1.
Writeln('            Transform Backprop Stage 2A');

// Obtain X5 from Hidden1 and W1.
// Backprop Hidden1 Grad creates W1 Grad. Input: X5ᵀ.Value, Hidden1.Grad. Output: W1.Grad.
// Equation: W1.Grad = X5ᵀ.Value · Hidden1.Grad. W1.Grad is R^{D x DB}. X5ᵀ.Value is R^{L x D}. Hidden1.Grad is R^{D x DB).
MatMulTN(@X5.Value, @Hidden1.Grad, @W1.Grad, SeqLen, ModelDimProj, ModelDim);

// Backprop Hidden1 Grad accumulates into X5 Grad. Input: Hidden1.Grad, W1ᵀ.Value. Output: X5.Grad.
// Equation: X5.Grad = Hidden1.Grad · W1ᵀ.Value. Hidden1.Grad is R^{D x DB). W1ᵀ.Value is R^{DB x D}. X5.Grad is R^{L x D}.
MatMulAccNT(@Hidden1.Grad, @W1.Value, @X5.Grad, SeqLen, ModelDim, ModelDimProj);

// 1. BACKPROP STAGE TRANSFORMER.

// 1J. Backprop Layer-Norm. Obtain X5 from X4.
Writeln('          Transform Backprop Stage 1J');

// Backprop Layer-Norm: Input X5, dX5. Output X4.Grad, Gamma2.Grad, Beta2.Grad.
// Equation: X4.Grad, Gamma2.Grad, Beta2.Grad = LayerNorm(X5, X5.Grad, Gamma2, Beta2). X4.Grad, X5.Grad in R^{L x D}. Gamma2.Grad, Beta2.Grad in R^{D}.
LayerNormBackward(X5.Grad, X4.Grad, Gamma2.Grad, Beta2.Grad, SeqLen, Gamma2.Value, LNXhat2, LNInvStd2);

// Display X4.Grad matrix.
VTPDisplayX('Display X4.Grad, in transform, after stage 1J, layer-norm.', X4.Grad, G);

// 1I. Backprop Split. Input: X1.Grad. Output: X3.Grad. Output X4.Grad,
Writeln('          Transform Bacprop Stage 1I');

// Equation: X3.Grad, X4.Grad = X1.Grad. All in R^{L x D}.
GradSplit(X4.Grad, X1.Grad, X3.Grad, SeqLen, ModelDim);

// Guide: To find the change for the weights: dW0 = X6ᵀ ·  dX7.
//        To find the error for the input: dX6 = dX7  · W0ᵀ.
// Guide: To find the change for the multiplication: dScores = dX2 · Vᵀ.
//        To find the error for the input: dV = Sᵀ · dX2.

// 1H. Backprop Mutiplication/Overwrite. Obtain W0 Grad from X3 Grad: Input: X2ᵀ.Value, X3.Grad. Output: W0.Grad.

// Equations: W0.Grad = X2ᵀ.Value · X3.Grad. W0.Grad is R^{L x D}. X3.Grad is R^{L x D}.
MatMulTN(@X2.Value, @X3.Grad, @W0.Grad, ModelDim, SeqLen, ModelDim);

// Backprop Create X2 Grad from X3 Grad: Input: X3.Grad, W0ᵀ.Value. Output: X2.Grad.
// Equations: X2.Grad = X3.Grad · W0ᵀ. W0.Grad is R^{L x D}. X2.Grad, X3.Grad is R^{L x D}. W0ᵀ.Value is R^{D x L}.
MatMulNT(@X3.Grad, @W0.Value, @X2.Grad, SeqLen, ModelDim, ModelDim);

// Display X3.Grad matrix.
VTPDisplayX('Display X3.Grad, in transform, before stage 1G.', X3.Grad, G);

// 1G. Backprop Multiplication/Overwrite. Obtain Scores2.Grad from X2.Grad: Input X2.Grad, Vᵀ.Value. Output: Scores2.Grad.
Writeln('          Transform Backprop Stage 1G');

// Equations: Scores2.Grad = X2.Grad · Vᵀ.Value. Scores2.Grad is R^{L x L}. X2.Grad is R^{L x D}. Vᵀ.Value is R^{D x L}.
for h := 0 to nHead - 1 do begin
HeadOffset := h * HeadDim;
MatMulFullTN(@X2.Grad[0, HeadOffset], @V.Value[0, HeadOffset], @ScoresHead2[h].Grad[0,0], SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen);
end;

// Backprop Create VHead Grad from X2Head Grad: Input ScoresHead2ᵀ.Value, X2Head.Grad. Output: VHead.Grad.
// Equations: VHead.Grad = ScoresHead2ᵀ.Value · X2Head.Grad. VHead.Grad is R^{L x D}. ScoresHead2ᵀ.Value is R^{L x L}. X2Head.Grad is R^{L x D}.
for h := 0 to nHead - 1 do begin
HeadOffset := h * HeadDim;
MatMulFullNT(@ScoresHead2[h].Value[0,0], @X2.Grad[0, HeadOffset], @V.Grad[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
end;

// 1F. Backprop Standardize, Mask & Softmax. Obtain ScoresHead1.
Writeln('          Transform Backprop Stage 1F');

// Insure ScoresHead1.Grad is empty.
for h := 0 to nHead - 1 do
FillChar(ScoresHead1[h].Grad, SizeOf(ScoresHead1[h].Grad), 0);

// Backprop Softmax: Input ScoresHead2.Value ScoresHead2.Grad. Output ScoresHead1.Grad.
// Equation: ScoresHead1.Grad = SoftMaxBackwards(ScoresHead2.Value, ScoresHead2.Grad).
for h := 0 to nHead - 1 do
for i := 0 to SeqLen - 1 do
  SoftmaxBackward(ScoresHead2[h].Value[i], ScoresHead2[h].Grad[i], ScoresHead1[h].Grad[i]);

// Scaling after Softmax.
for h := 0 to nHead - 1 do
Scale(SeqLen * SeqLen, InvSqrtHeadDim, @ScoresHead1[h].Grad[0,0]);
//cblas_sscal(SeqLen * SeqLen, InvSqrtHeadDim, @ScoresHead1[h].Grad[0,0], 1);

// Backprop AutoRegression.
// Equation: ScoresHead1.Grad = Unmask(ScoresHead1.Grad).
for h := 0 to nHead - 1 do
for i := 0 to SeqLen - 1 do
  for j := i + 1 to SeqLen - 1 do
    ScoresHead1[h].Grad[i, j] := 0.0;

// Backprop standardization. Input: ScoresHead1.Grad. Output: ScoresHead1.Grad.
// Equation: ScoresHead1.Grad = Sqrt(1 / ModelDim). ScoresHead1.Grad in R^{L x L}. Done above.

// Display ScoresHead.Grad matrix.
VTPDisplayX('ScoresHead1[0].Grad, transform, before stage 1E, Q and K-transform.', ScoresHead1[0].Grad, G);

// 1E. Backprop multiplication. Obtain QHead.Grad and KHead.Grad.
Writeln('          Transform Backprop Stage 1E');

// Backprop Multiplication: Input ScoresHead1.Grad, KHead.Value. Output QHead.Grad.
// Equation: QHead.Grad = ScoresHead1.Grad · KHead.Value. QHead.Grad, ScoresHead1.Grad in R^{L x L}. KHead.Value in R^{L x D}.
for h := 0 to nHead - 1 do begin
HeadOffset := h * HeadDim;
MatMulFullNN(@ScoresHead1[h].Grad[0,0], @K.Value[0, HeadOffset], @Q.Grad[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
end;

// Backprop Multiplication: Input ScoresHead1.Gradᵀ, Q.Value. Output K.Grad.
// Equation: K.Grad = ScoresHead1.Gradᵀ · Q.Value. K.Grad in R^{L x D}. ScoresHead1.Gradᵀ in R^{L · L}. Q.Value in R^{L x D}.
for h := 0 to nHead - 1 do begin
HeadOffset := h * HeadDim;
MatMulFullNT(@ScoresHead1[h].Grad[0,0], @Q.Value[0, HeadOffset], @K.Grad[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
end;

// 1D. RoPE. Nil.
Writeln('          Transform Backprop Stage 1D Nil');

// 1C. Backprop multiplication/overwrite. Obtain W_.Grad and X1_q.Grad for Q, K, and V.
Writeln('          Transform Backprop Stage 1C');

// Obtain X1q, X1k, X1v, from X1.
{Wq.Grad = X1ᵀ.Value · Q.Grad
X1q.Grad = Q.Grad · Wqᵀ.Value}
// Backprop Create Wq Grad from Q Grad: Input X1ᵀ.Value · Q.Grad. Output Wq.Grad.
// Equation: Wq.Grad = X1ᵀ · Q.Grad. Wq.Grad in R^{D x D}. X1ᵀ in R^{D x L}. Q.Grad in R^{L x D}.
MatMulTN(@X1.Value, @Q.Grad, @Wq.Grad, ModelDim, ModelDim, SeqLen);

// Backprop Create X1q from Q Grad: Input Q.Grad, Wqᵀ.Value. Output X1q.Grad.
// Equation: X1q.Grad = Q.Grad · Wqᵀ. X1q.Grad in R^{L x D}. Q.Grad in R^{L x D}. Wqᵀ.Value in R^{D · D}.
MatMulNT(@Q.Grad, @Wq.Value, @X1q.Grad, SeqLen, ModelDim, ModelDim);

{Wk.Grad = X1ᵀ.Value · K.Grad
X1k.Grad = K.Grad · Wkᵀ.Value}
// Backprop Create Wk Grad from K Grad: Input X1ᵀ.Value · K.Grad. Output Wk.Grad.
// Equation:  Wk.Grad = X1ᵀ.Value · K.Grad. Wk.Grad in R^{D x D}. X1ᵀ.Value in R^{D x L}. K.Grad in R^{L x D}.
MatMulTN(@X1.Value, @K.Grad, @Wk.Grad, ModelDim, ModelDim, SeqLen);

// Backprop Create X1kk Grad from K Grad. Input K.Grad, Wkᵀ.Value. Output X1k.Grad.
// Equation: X1k.Grad = K.Grad · Wkᵀ.Value. X1k.Grad in R^{L x D}. K.Grad in R^{L x D}. Wkᵀ.Value in R^{D · D}.
MatMulNT(@K.Grad, @Wk.Value, @X1k.Grad, SeqLen, ModelDim, ModelDim);

{Wv.Grad = X1ᵀ · V.Grad
X1v.Grad = V.Grad · Wvᵀ.Value}
// Backprop Create Wv Grad from V Grad: Input X1ᵀ.Value · V.Grad. Output Wv.Grad.
// Equation: Wv.Grad = X1ᵀ.Value · V.Grad. Wv.Grad in R^{D x D}. X1ᵀ in R^{D x L}. V.Grad in R^{L x D}.
MatMulTN(@X1.Value, @V.Grad, @Wv.Grad, ModelDim, ModelDim, SeqLen);

// Backprop Create X1v Grad from V Grad. Input V.Grad, Wvᵀ. Value. Output X1v.Grad.
// Equation: X1v.Grad = V.Grad times Wvᵀ.Value. X1v.Grad = V.Grad · WVᵀ.Value. V.Grad in R^{L x D}. Wvᵀ.Value in R^{D · D}.
MatMulNT(@V.Grad, @Wv.Value, @X1v.Grad, SeqLen, ModelDim, ModelDim);

// 1B. Backprop Merge: Obtain X1 Grad as sum of Grads. Input X1q.Grad, X1k.Grad, and X1v.Grad. Output X1.Grad.
Writeln('          Transform Backprop Stage 1B');

// Equation:  X1.Grad = X1q.Grad + X1k.Grad + X1v.Grad. All in R^{L x D}.
for i := 0 to SeqLen - 1 do
for j := 0 to ModelDim - 1 do
  X1.Grad[i, j] := X1q.Grad[i, j] + X1k.Grad[i, j] + X1v.Grad[i, j];

// Backprop Accumulate: Input X1.Grad, X4.Grad. Output X1.Grad.
// Equation: X1.Grad = X1.Grad + X4.Grad. All R^{L x D}.
AccumulateGrad(X4.Grad, X1.Grad, SeqLen, ModelDim);

// Display X3.Grad matrix.
VTPDisplayX('Display X1.Grad, in transform, after concatenation.', X1.Grad, G);

// 1A. Backprop Layer-Norm: Input X1.Value, X1.Grad. Output X.Grad, Gamma1.Grad, Beta1.Grad.
Writeln('          Transform Backprop Stage 1A');

// Equation: X.Grad, Gamma1.Grad, Beta1.Grad = LayerNorm(X1.Value, X1.Grad, Gamma1.Value, Beta1.Value). X.Grad, X1.Grad in R^{L x D}. Gamma1.Grad, Beta1.Grad in R^{D}.
LayerNormBackward(X1.Grad, X.Grad, Gamma1.Grad, Beta1.Grad, SeqLen, Gamma1.Value, LNXhat1, LNInvStd1);

// Display X.Grad matrix.
VTPDisplayX('Display X.Grad, in transform, at end.', X.Grad, G);

// Place X7 in X for next .
Writeln('          Transform Backprop Copy X matrix');
CopyXTensor(X7, X);
If VerboseTransform then begin
Writeln('End of tranformer .');
Pause;
end;
}

