unit Train;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  Math, SysUtils,
  { WesChat units }
  Display, Global, IOHandler, Matrix, OutputHead, TransformForward, TransformBackprop, Util;

{ Training interface }
{ TokenizedCorpus contains token IDs used to build InputTokens and TargetTokens }
{ nVocab is the active vocabulary size; ModelDim is the embedding/model dimension }

procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState; const TokenizedCorpus: TIVector);

implementation

{ Training data }
type
  TRowLossVector = array[0..SeqLen - 1] of Single;    // Row losses returned by the CUDA CE-loss kernel.
  TLossBreakdown = record                             // UD performance.
    TotalLossSum: Double;
    LexicalLossSum: Double;
    UDLossSum: Double;
    TotalCount: Int64;
    LexicalCount: Int64;
    UDCount: Int64;
  end;

{// X7 dim repurposing. This is turned off.
  TX7CorrPair = record
    Dim1, Dim2: Integer;
    Corr: Double;
  end;
  TX7CorrPairs = array of TX7CorrPair;

  TX7BlockPair = record
    Dim1, Dim2: Integer;
    MinAbsCorr: Double;
    MeanAbsCorr: Double;
    RMS1, RMS2: Double;
    SignConsistent: Boolean;
  end;
  TX7BlockPairs = array of TX7BlockPair;
  TDoubleVector = array of Double;
  TDoubleMatrix = array of Double;
  TX7EigenVector = array of Single;}

var
  AutoSaveEpoch: Integer;                             // First epoch eligible for automatic best-model saving.
  MinSaveGap: Integer;                                // Minimum epochs between automatic best-model saves.
  MinSaveDelta: Double;                               // Minimum loss improvement required for another best-model save.
  RowLoss: TRowLossVector;                            // Per-row cross-entropy losses copied from the GPU.
  Beta1Power, Beta2Power: Single;                     // AdamW bias-correction powers.
  LossBreakdown: TLossBreakdown;                      // UD performance.
  LexicalLoss, UDLoss, TotalLexicalLossSum, TotalUDLossSum: Double;
  TotalLexicalCount, TotalUDCount: Int64;

{{ Repurposing dimensions This is turned off }
// Display the most correlated pairs from X7 matrix.
procedure ReportMostCorrelatedX7Dims(const Stats: TX7SimilarityStats; ModelDim, TopCount: Integer);
var
  Cross, Sum: array of Single;
  Best: TX7CorrPairs;
  i, j, k, InsertAt: Integer;
  N: Int64;
  SumI, SumJ, CrossIJ: Double;
  VarI, VarJ, CovIJ, Corr, Score: Double;
begin
  N := Stats.SampleCount;

  if N < 2 then begin
    Writeln('Not enough X7 samples to calculate correlations.');
    Exit;
  end;

  SetLength(Cross, ModelDim * ModelDim);
  SetLength(Sum, ModelDim);
  SetLength(Best, TopCount);

  cudaMemcpy(@Cross[0], Stats.dCross,
    ModelDim * ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);
  cudaMemcpy(@Sum[0], Stats.dSum,
    ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);

  // Initialize the list.
  for k := 0 to TopCount - 1 do begin
    Best[k].Dim1 := -1;
    Best[k].Dim2 := -1;
    Best[k].Corr := 0.0;
  end;

  // Examine each pair once. The diagonal is deliberately skipped.
  for i := 0 to ModelDim - 2 do begin

    SumI := Sum[i];

    // Centered sum of squares for dimension i.
    VarI := Cross[i * ModelDim + i] - (SumI * SumI / N);
    if VarI <= 0.0 then Continue;

    for j := i + 1 to ModelDim - 1 do begin

      SumJ := Sum[j];

      VarJ := Cross[j * ModelDim + j] - (SumJ * SumJ / N);

      if VarJ <= 0.0 then Continue;

      CrossIJ := Cross[i * ModelDim + j];

      // Centered cross product.
      CovIJ := CrossIJ - (SumI * SumJ / N);

      Corr := CovIJ / Sqrt(VarI * VarJ);

      // Allow for small floating-point excursions.
      if Corr > 1.0 then
        Corr := 1.0
      else if Corr < -1.0 then
        Corr := -1.0;

      Score := Abs(Corr);

      // Find where this pair belongs in the TopCount list.
      InsertAt := -1;

      for k := 0 to TopCount - 1 do
        if Score > Abs(Best[k].Corr) then begin
          InsertAt := k;
          Break;
        end;

      if InsertAt >= 0 then begin
        // Move lower-ranked pairs down one position.
        for k := TopCount - 1 downto InsertAt + 1 do
          Best[k] := Best[k - 1];
        Best[InsertAt].Dim1 := i;
        Best[InsertAt].Dim2 := j;
        Best[InsertAt].Corr := Corr;
      end;
    end;
  end;

  Writeln('Most highly correlated X7 dimensions, Samples = ', N);
  Writeln(' Rank    Dim 1    Dim 2       Corr       |Corr|');

  for k := 0 to TopCount - 1 do
    if Best[k].Dim1 >= 0 then
      Writeln(k + 1:5, Best[k].Dim1:9, Best[k].Dim2:9,
        Best[k].Corr:12:6, Abs(Best[k].Corr):12:6);

  Writeln;
end;

procedure ReportMostCorrelatedX7Pairs(const X7Stats: TX7SimilarityArray; ModelDim, TopCount: Integer);
var
  Cross, Sum: array of array of Single;
  Best: TX7BlockPairs;
  Blk, i, j, k, InsertAt: Integer;
  N, TotalN: Int64;
  SumI, SumJ, VarI, VarJ, CovIJ, Corr: Double;
  AbsCorr, MinAbsCorr, SumAbsCorr, MeanAbsCorr: Double;
  SumSqI, SumSqJ, RMS1, RMS2: Double;
  FirstSign, ThisSign: Integer;
  SignConsistent, ValidPair: Boolean;
begin
  if Length(X7Stats) = 0 then begin
    Writeln('No X7 correlation statistics available.');
    Exit;
  end;

  if TopCount <= 0 then
    Exit;

  // Copy each block's accumulated statistics from GPU to CPU.
  SetLength(Cross, Length(X7Stats));
  SetLength(Sum, Length(X7Stats));

  for Blk := 0 to High(X7Stats) do begin
    SetLength(Cross[Blk], ModelDim * ModelDim);
    SetLength(Sum[Blk], ModelDim);

    cudaMemcpy(@Cross[Blk][0], X7Stats[Blk].dCross,
      ModelDim * ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);

    cudaMemcpy(@Sum[Blk][0], X7Stats[Blk].dSum,
      ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);
  end;

  SetLength(Best, TopCount);

  for k := 0 to TopCount - 1 do begin
    Best[k].Dim1 := -1;
    Best[k].Dim2 := -1;
    Best[k].MinAbsCorr := 0.0;
    Best[k].MeanAbsCorr := 0.0;
    Best[k].RMS1 := 0.0;
    Best[k].RMS2 := 0.0;
    Best[k].SignConsistent := False;
  end;

  // Examine every unique pair of ModelDim dimensions.
  for i := 0 to ModelDim - 2 do
    for j := i + 1 to ModelDim - 1 do begin

      MinAbsCorr := 2.0;
      SumAbsCorr := 0.0;
      SumSqI := 0.0;
      SumSqJ := 0.0;
      TotalN := 0;
      FirstSign := 0;
      SignConsistent := True;
      ValidPair := True;

      // Find the correlation of this pair in every block.
      for Blk := 0 to High(X7Stats) do begin
        N := X7Stats[Blk].SampleCount;

        if N < 2 then begin
          ValidPair := False;
          Break;
        end;

        SumI := Sum[Blk][i];
        SumJ := Sum[Blk][j];

        VarI := Cross[Blk][i * ModelDim + i] - (SumI * SumI / N);
        VarJ := Cross[Blk][j * ModelDim + j] - (SumJ * SumJ / N);

        if (VarI <= 0.0) or (VarJ <= 0.0) then begin
          ValidPair := False;
          Break;
        end;

        CovIJ := Cross[Blk][i * ModelDim + j] - (SumI * SumJ / N);
        Corr := CovIJ / Sqrt(VarI * VarJ);

        if Corr > 1.0 then
          Corr := 1.0
        else if Corr < -1.0 then
          Corr := -1.0;

        AbsCorr := Abs(Corr);

        if AbsCorr < MinAbsCorr then
          MinAbsCorr := AbsCorr;

        SumAbsCorr := SumAbsCorr + AbsCorr;

        // Accumulate raw activation sums of squares across all blocks.
        SumSqI := SumSqI + Cross[Blk][i * ModelDim + i];
        SumSqJ := SumSqJ + Cross[Blk][j * ModelDim + j];
        Inc(TotalN, N);

        // Check whether correlation sign remains consistent.
        if Corr >= 0.0 then
          ThisSign := 1
        else
          ThisSign := -1;

        if FirstSign = 0 then
          FirstSign := ThisSign
        else if ThisSign <> FirstSign then
          SignConsistent := False;
      end;

      if not ValidPair then
        Continue;

      MeanAbsCorr := SumAbsCorr / Length(X7Stats);

      if TotalN > 0 then begin
        RMS1 := Sqrt(SumSqI / TotalN);
        RMS2 := Sqrt(SumSqJ / TotalN);
      end
      else begin
        RMS1 := 0.0;
        RMS2 := 0.0;
      end;

      // Find insertion point, ranked primarily by minimum |correlation|.
      InsertAt := -1;

      for k := 0 to TopCount - 1 do
        if (Best[k].Dim1 < 0) or
           (MinAbsCorr > Best[k].MinAbsCorr) or
           ((MinAbsCorr = Best[k].MinAbsCorr) and
            (MeanAbsCorr > Best[k].MeanAbsCorr)) then begin
          InsertAt := k;
          Break;
        end;

      if InsertAt >= 0 then begin
        for k := TopCount - 1 downto InsertAt + 1 do
          Best[k] := Best[k - 1];

        Best[InsertAt].Dim1 := i;
        Best[InsertAt].Dim2 := j;
        Best[InsertAt].MinAbsCorr := MinAbsCorr;
        Best[InsertAt].MeanAbsCorr := MeanAbsCorr;
        Best[InsertAt].RMS1 := RMS1;
        Best[InsertAt].RMS2 := RMS2;
        Best[InsertAt].SignConsistent := SignConsistent;
      end;
    end;

  Writeln('Most redundant X7 dimension pairs across all blocks');
  Writeln(' Rank   Dim1   Dim2    Min|Corr|   Mean|Corr|      RMS1      RMS2   Sign');

  for k := 0 to TopCount - 1 do
    if Best[k].Dim1 >= 0 then begin
      Write(k + 1:5, Best[k].Dim1:7, Best[k].Dim2:7,
        Best[k].MinAbsCorr:13:6, Best[k].MeanAbsCorr:13:6,
        Best[k].RMS1:10:5, Best[k].RMS2:10:5);

      if Best[k].SignConsistent then
        Writeln('   Same')
      else
        Writeln('   Mixed');
    end;

  Writeln;
end;

procedure SortEigenvaluesDescending(var Eigen: TDoubleVector);
var
  i, j: Integer;
  Temp: Double;
begin
  for i := 0 to High(Eigen) - 1 do
    for j := i + 1 to High(Eigen) do
      if Eigen[j] > Eigen[i] then begin
        Temp := Eigen[i];
        Eigen[i] := Eigen[j];
        Eigen[j] := Temp;
      end;
end;

procedure JacobiEigenvalues(var A: TDoubleMatrix; N: Integer; var Eigen: TDoubleVector);
const
  Tolerance = 1.0e-10;
  MaxSweeps = 100;
var
  Sweep, p, q, k: Integer;
  App, Aqq, Apq, Tau, T, C, S: Double;
  Akp, Akq, MaxOff: Double;
begin
  SetLength(Eigen, N);

  for Sweep := 1 to MaxSweeps do begin
    MaxOff := 0.0;

    for p := 0 to N - 2 do
      for q := p + 1 to N - 1 do begin
        Apq := A[p * N + q];

        if Abs(Apq) > MaxOff then
          MaxOff := Abs(Apq);

        if Abs(Apq) <= Tolerance then
          Continue;

        App := A[p * N + p];
        Aqq := A[q * N + q];

        Tau := (Aqq - App) / (2.0 * Apq);

        if Tau >= 0.0 then
          T := 1.0 / (Tau + Sqrt(1.0 + Tau * Tau))
        else
          T := -1.0 / (-Tau + Sqrt(1.0 + Tau * Tau));

        C := 1.0 / Sqrt(1.0 + T * T);
        S := T * C;

        // Update rows/columns p and q while preserving symmetry.
        for k := 0 to N - 1 do
          if (k <> p) and (k <> q) then begin
            Akp := A[k * N + p];
            Akq := A[k * N + q];

            A[k * N + p] := C * Akp - S * Akq;
            A[p * N + k] := A[k * N + p];

            A[k * N + q] := S * Akp + C * Akq;
            A[q * N + k] := A[k * N + q];
          end;

        // Update the diagonal.
        A[p * N + p] := App - T * Apq;
        A[q * N + q] := Aqq + T * Apq;

        A[p * N + q] := 0.0;
        A[q * N + p] := 0.0;
      end;

    if MaxOff < Tolerance then
      Break;
  end;

  for p := 0 to N - 1 do
    Eigen[p] := A[p * N + p];

  SortEigenvaluesDescending(Eigen);
end;

procedure SortEigenpairsDescending(var Eigen, EigenVectors: TDoubleVector; N: Integer);
var
  i, j, k: Integer;
  Temp: Double;
begin
  for i := 0 to N - 2 do
    for j := i + 1 to N - 1 do
      if Eigen[j] > Eigen[i] then begin

        // Swap eigenvalues.
        Temp := Eigen[i];
        Eigen[i] := Eigen[j];
        Eigen[j] := Temp;

        // Swap the corresponding eigenvector columns.
        for k := 0 to N - 1 do begin
          Temp := EigenVectors[k * N + i];
          EigenVectors[k * N + i] := EigenVectors[k * N + j];
          EigenVectors[k * N + j] := Temp;
        end;
      end;
end;

procedure JacobiEigen(var A: TDoubleMatrix; N: Integer; var Eigen, EigenVectors: TDoubleVector);
const
  Tolerance = 1.0e-10;
  MaxSweeps = 100;
var
  Sweep, p, q, i, k: Integer;
  App, Aqq, Apq, Tau, T, C, S: Double;
  Akp, Akq, Vkp, Vkq, MaxOff: Double;
  Temp: Double;
begin
  SetLength(Eigen, N);
  SetLength(EigenVectors, N * N);

  // Start eigenvectors as the identity matrix.
  for p := 0 to N - 1 do
    for q := 0 to N - 1 do
      if p = q then
        EigenVectors[p * N + q] := 1.0
      else
        EigenVectors[p * N + q] := 0.0;

  for Sweep := 1 to MaxSweeps do begin
    MaxOff := 0.0;

    for p := 0 to N - 2 do
      for q := p + 1 to N - 1 do begin
        Apq := A[p * N + q];

        if Abs(Apq) > MaxOff then
          MaxOff := Abs(Apq);

        if Abs(Apq) <= Tolerance then
          Continue;

        App := A[p * N + p];
        Aqq := A[q * N + q];

        Tau := (Aqq - App) / (2.0 * Apq);

        if Tau >= 0.0 then
          T := 1.0 / (Tau + Sqrt(1.0 + Tau * Tau))
        else
          T := -1.0 / (-Tau + Sqrt(1.0 + Tau * Tau));

        C := 1.0 / Sqrt(1.0 + T * T);
        S := T * C;

        // Rotate covariance matrix rows/columns p and q.
        for k := 0 to N - 1 do
          if (k <> p) and (k <> q) then begin
            Akp := A[k * N + p];
            Akq := A[k * N + q];

            A[k * N + p] := C * Akp - S * Akq;
            A[p * N + k] := A[k * N + p];

            A[k * N + q] := S * Akp + C * Akq;
            A[q * N + k] := A[k * N + q];
          end;

        // Update diagonal elements.
        A[p * N + p] := App - T * Apq;
        A[q * N + q] := Aqq + T * Apq;

        A[p * N + q] := 0.0;
        A[q * N + p] := 0.0;

        // Accumulate the same rotation into the eigenvectors.
        for k := 0 to N - 1 do begin
          Vkp := EigenVectors[k * N + p];
          Vkq := EigenVectors[k * N + q];

          EigenVectors[k * N + p] := C * Vkp - S * Vkq;
          EigenVectors[k * N + q] := S * Vkp + C * Vkq;
        end;
      end;

    if MaxOff < Tolerance then
      Break;
  end;

  // Diagonal of A now contains the eigenvalues.
  for p := 0 to N - 1 do
    Eigen[p] := A[p * N + p];

  // Keep eigenvectors paired with their eigenvalues.
  SortEigenpairsDescending(Eigen, EigenVectors, N);

  {Writeln;
  Writeln('Weakest X7 eigenvector. Eigenvalue = ',
    Eigen[ModelDim - 1]:12:6);

  for i := 0 to ModelDim - 1 do begin
    Write(EigenVectors[i * ModelDim + ModelDim - 1]:10:6, ' ');

    if ((i + 1) mod 8) = 0 then
      Writeln;
  end;}

  Temp := 0.0;

  for i := 0 to ModelDim - 1 do
    Temp := Temp + Sqr(EigenVectors[i * ModelDim + ModelDim - 1]);

  Writeln('Weakest eigenvector norm = ', Sqrt(Temp):12:8);
end;

procedure GetWeakestX7Eigenvector(const Stats: TX7SimilarityStats; ModelDim: Integer;
  var WeakVector: TX7EigenVector; var WeakEigenvalue: Double);
var
  HostCross, HostSum: array of Single;
  Covar, Eigen, EigenVectors: TDoubleVector;
  i, j: Integer;
  N: Int64;
begin
  N := Stats.SampleCount;

  if N < 2 then
    raise Exception.Create('Not enough X7 samples for eigenvector.');

  SetLength(HostCross, ModelDim * ModelDim);
  SetLength(HostSum, ModelDim);
  SetLength(Covar, ModelDim * ModelDim);

  cudaMemcpy(@HostCross[0], Stats.dCross,
    ModelDim * ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);

  cudaMemcpy(@HostSum[0], Stats.dSum,
    ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);

  // Covariance for this block only.
  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDim - 1 do
      Covar[i * ModelDim + j] :=
        (Double(HostCross[i * ModelDim + j]) -
        (Double(HostSum[i]) * Double(HostSum[j]) / N)) / (N - 1);

  JacobiEigen(Covar, ModelDim, Eigen, EigenVectors);

  WeakEigenvalue := Eigen[ModelDim - 1];

  SetLength(WeakVector, ModelDim);

  // Eigenvectors are columns.
  for i := 0 to ModelDim - 1 do
    WeakVector[i] :=
      Single(EigenVectors[i * ModelDim + ModelDim - 1]);
end;

procedure ReportX7CovarianceSpectrum(const X7Stats: TX7SimilarityArray; ModelDim: Integer);
var
  HostCross, HostSum: array of Single;
  Covar, Eigen, EigenVectors: TDoubleVector;
  Blk, i, j, ValidBlocks: Integer;
  N, TotalSamples: Int64;
  BlockCov, TotalVariance, CumVariance, Pct, CumPct: Double;
begin
  if Length(X7Stats) = 0 then begin
    Writeln('No X7 statistics available.');
    Exit;
  end;

  SetLength(HostCross, ModelDim * ModelDim);
  SetLength(HostSum, ModelDim);
  SetLength(Covar, ModelDim * ModelDim);

  // Covar will accumulate the separately centered covariance
  // matrices from each transformer block.
  for i := 0 to High(Covar) do
    Covar[i] := 0.0;

  ValidBlocks := 0;
  TotalSamples := 0;

  for Blk := 0 to High(X7Stats) do begin
    N := X7Stats[Blk].SampleCount;

    if N < 2 then
      Continue;

    cudaMemcpy(@HostCross[0], X7Stats[Blk].dCross,
      ModelDim * ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);

    cudaMemcpy(@HostSum[0], X7Stats[Blk].dSum,
      ModelDim * SizeOf(Single), cudaMemcpyDeviceToHost);

    // Construct this block's centered covariance matrix and
    // accumulate it into Covar.
    for i := 0 to ModelDim - 1 do
      for j := 0 to ModelDim - 1 do begin

        BlockCov :=
          (Double(HostCross[i * ModelDim + j]) -
          (Double(HostSum[i]) * Double(HostSum[j]) / N)) / (N - 1);

        Covar[i * ModelDim + j] :=
          Covar[i * ModelDim + j] + BlockCov;
      end;

    Inc(ValidBlocks);
    Inc(TotalSamples, N);
  end;

  if ValidBlocks = 0 then begin
    Writeln('No valid X7 blocks for covariance spectrum.');
    Exit;
  end;

  // Average the separately centered covariance matrices.
  for i := 0 to High(Covar) do
    Covar[i] := Covar[i] / ValidBlocks;

  //JacobiEigenvalues(Covar, ModelDim, Eigen);
  JacobiEigen(Covar, ModelDim, Eigen, EigenVectors);

  // Tiny negative eigenvalues can occur from floating-point rounding.
  for i := 0 to High(Eigen) do
    if (Eigen[i] < 0.0) and (Eigen[i] > -1.0e-8) then
      Eigen[i] := 0.0;

  TotalVariance := 0.0;

  for i := 0 to High(Eigen) do
    if Eigen[i] > 0.0 then
      TotalVariance := TotalVariance + Eigen[i];

  Writeln;
  Writeln('X7 covariance eigenvalue spectrum');
  Writeln('Blocks = ', ValidBlocks, '; Samples = ', TotalSamples);
  Writeln(' Dim       Eigenvalue     Variance %   Cumulative %');

  CumVariance := 0.0;

  for i := 0 to High(Eigen) do begin
    if Eigen[i] > 0.0 then
      CumVariance := CumVariance + Eigen[i];

    if TotalVariance > 0.0 then begin
      Pct := 100.0 * Eigen[i] / TotalVariance;
      CumPct := 100.0 * CumVariance / TotalVariance;
    end
    else begin
      Pct := 0.0;
      CumPct := 0.0;
    end;

    if (i < 5) or (i > High(Eigen) - 5) then
      Writeln(i + 1:4, Eigen[i]:17:7, Pct:13:5, CumPct:15:5);
  end;

  Writeln('Largest eigenvalue  = ', Eigen[0]:12:6);
  Writeln('Median eigenvalue   = ', Eigen[ModelDim div 2]:12:6);
  Writeln('Smallest eigenvalue = ', Eigen[ModelDim - 1]:12:6);

  if Eigen[0] > 0.0 then
    Writeln('Smallest/Largest    = ', Eigen[ModelDim - 1] / Eigen[0]:12:8);

  if TotalVariance > 0.0 then
    Writeln('Smallest variance % = ', 100.0 * Eigen[ModelDim - 1] / TotalVariance:12:8);
end;

procedure ReportX7ProjectionChange(dOriginal, dProjected: PSingle; SeqLen, ModelDim: Integer);
var
  Original, Projected: array of Single;
  i, N: Integer;
  Delta, SumSq, MaxDelta: Double;
begin
  N := SeqLen * ModelDim;

  SetLength(Original, N);
  SetLength(Projected, N);

  cudaMemcpy(@Original[0], dOriginal,
    N * SizeOf(Single), cudaMemcpyDeviceToHost);

  CheckCudaError('Copy original X7 for projection test');

  cudaMemcpy(@Projected[0], dProjected,
    N * SizeOf(Single), cudaMemcpyDeviceToHost);

  CheckCudaError('Copy projected X7 for projection test');

  SumSq := 0.0;
  MaxDelta := 0.0;

  for i := 0 to N - 1 do begin
    Delta := Double(Projected[i]) - Double(Original[i]);

    SumSq := SumSq + Delta * Delta;

    if Abs(Delta) > MaxDelta then
      MaxDelta := Abs(Delta);
  end;

  Writeln('Projection RMS change = ', Sqrt(SumSq / N):12:8);
  Writeln('Projection max change = ', MaxDelta:12:8);
end;

procedure ReportWeakVectorGPU(dV: PSingle; ModelDim: Integer);
var
  V: array of Single;
  i: Integer;
  SumSq, MaxAbs: Double;
begin
  SetLength(V, ModelDim);

  cudaMemcpy(@V[0], dV, ModelDim * SizeOf(Single),
    cudaMemcpyDeviceToHost);

  CheckCudaError('Copy weak X7 vector back to host');

  SumSq := 0.0;
  MaxAbs := 0.0;

  for i := 0 to ModelDim - 1 do begin
    SumSq := SumSq + Sqr(V[i]);

    if Abs(V[i]) > MaxAbs then
      MaxAbs := Abs(V[i]);
  end;

  Writeln('GPU weak-vector norm = ', Sqrt(SumSq):12:8);
  Writeln('GPU weak-vector max  = ', MaxAbs:12:8);
end;}

{ Training log }
// Close the training log.
procedure CloseTrainLog;
begin
  if not TrainLogOpen then Exit;

  Writeln(TrainLogFile, 'Training session ended: ', DateTimeToStr(Now));
  Writeln(TrainLogFile);
  Flush(TrainLogFile);

  CloseFile(TrainLogFile);
  TrainLogOpen := False;
end;

{ Training log }
// Open the training log.
procedure OpenTrainLog(const FileName: string);
begin
  // Close a log left open by a preceding training session.
  if TrainLogOpen then
    CloseTrainLog;

  TrainLogFileName := FileName;

  AssignFile(TrainLogFile, TrainLogFileName);

  if FileExists(TrainLogFileName) then
    Append(TrainLogFile)
  else
    Rewrite(TrainLogFile);

  TrainLogOpen := True;

  Writeln(TrainLogFile);
  Writeln(TrainLogFile, '============================================================');
  Writeln(TrainLogFile, 'Training session started: ', DateTimeToStr(Now));
  Writeln(TrainLogFile, '============================================================');
  Flush(TrainLogFile);
end;

{procedure OpenTrainLog(const FileName: string);
begin
  TrainLogFileName := FileName;

  AssignFile(TrainLogFile, TrainLogFileName);

  if FileExists(TrainLogFileName) then
    Append(TrainLogFile)
  else
    Rewrite(TrainLogFile);

  TrainLogOpen := True;

  Writeln(TrainLogFile);
  Writeln(TrainLogFile, '============================================================');
  Writeln(TrainLogFile, 'Training session started: ', DateTimeToStr(Now));
  Writeln(TrainLogFile, '============================================================');
  Flush(TrainLogFile);
end;}

// Write one timestamped line to the training log.
procedure TrainLog(const S: string);
begin
  if not TrainLogOpen then Exit;

  Writeln(TrainLogFile, DateTimeToStr(Now), '  ', S);
  Flush(TrainLogFile);
end;

// Write the starting model and training settings to the training log.
procedure LogTrainingStart(const ModelFileName: string);
begin
  if not TrainLogOpen then Exit;
  Writeln(TrainLogFile, 'Model file: ', ModelFileName);
  Writeln(TrainLogFile, 'Corpus ID: ', CorpusID);
  Writeln(TrainLogFile, 'Corpus bytes: ', nCorpus);
  Writeln(TrainLogFile, 'Raw tokens: ', RawTokenCount);
  Writeln(TrainLogFile, 'Tokenizer kind: ', Ord(TokenizerKind));
  Writeln(TrainLogFile, 'ModelDim: ', ModelDim);
  Writeln(TrainLogFile, 'ModelDimProj: ', ModelDimProj);
  Writeln(TrainLogFile, 'nBlock: ', nBlock);
  Writeln(TrainLogFile, 'nHead: ', nHead);
  Writeln(TrainLogFile, 'SeqLen: ', SeqLen);
  Writeln(TrainLogFile, 'nVocab: ', nVocab);
  Writeln(TrainLogFile, 'DimVocab: ', DimVocab);
  Writeln(TrainLogFile, 'Completed epochs: ', CompletedEpochs);
  Writeln(TrainLogFile, 'Global step: ', GlobalStep);
  Writeln(TrainLogFile, 'AdamW step: ', AdamWStep);
  Writeln(TrainLogFile, 'Learning rate: ', LearningRate:0:8);
  Writeln(TrainLogFile, 'Adaptive LR: ', AdaptiveLR);
  Writeln(TrainLogFile, 'Weight decay: ', WeightDecay:0:8);
  Writeln(TrainLogFile, 'Clip limit: ', ClipLimit:0:6);
  Writeln(TrainLogFile, 'Stride: ', Stride);
  Writeln(TrainLogFile, 'Attention dropout: ', ADropOut:0:4);
  Writeln(TrainLogFile, 'Residual dropout: ', RDropOut:0:4);
  Writeln(TrainLogFile, 'MLP dropout: ', MLPDropOut:0:4);
  if MinLoss < MaxDouble then begin
    Writeln(TrainLogFile, 'Historical minimum loss: ', MinLoss:0:6);
    Writeln(TrainLogFile, 'Historical minimum epoch: ', MinLossEpoch);
  end;
  if BestSavedLoss < MaxDouble then begin
    Writeln(TrainLogFile, 'Best saved loss: ', BestSavedLoss:0:6);
    Writeln(TrainLogFile, 'Best saved epoch: ', LastBestSaveEpoch);
  end;
  Writeln(TrainLogFile);
  Flush(TrainLogFile);
end;

// Write one completed epoch to the training log.
procedure LogTrainingEpoch(const Epoch: Integer; const MEL, BPB, DiffLoss, LR: Double);
begin
  if not TrainLogOpen then Exit;
  Write(TrainLogFile, 'Epoch ', Epoch, ': MEL=', MEL: 0: 6, '; BPB=', BPB: 0: 6, '; DiffLoss=', DiffLoss: 0: 6, '; LR=', LR: 0: 8);
  if MinLoss < MaxDouble then
    Write(TrainLogFile, '; Best=', MinLoss: 0: 6, ' @ ', MinLossEpoch);
  Writeln(TrainLogFile);
  Flush(TrainLogFile);
end;

{ Loss computation }
// Compute mean cross-entropy loss with the CUDA row-loss kernel.
{ Loss computation }

// Compute mean cross-entropy loss and separate lexical and UD-token losses.
function ComputeCELossGPU(dProbs: PSingle; dTargetTokens: PInteger; dRowLoss: PSingle;
  const TargetTokens: TIDimVector; var RowLoss: TRowLossVector; var LossBreakdown: TLossBreakdown): Double;
var
  i, Tok: Integer;
begin
  LaunchCELossRows(dProbs, dTargetTokens, dRowLoss, SeqLen, nVocab, DimVocab);

  if DebugCudaChecks then
    CheckCudaError('Launch CELossRows.');

  cudaMemcpy(@RowLoss[0], dRowLoss, SeqLen * SizeOf(Single), cudaMemcpyDeviceToHost);

  if DebugCudaChecks then
    CheckCudaError('Copy CE row losses.');

  FillChar(LossBreakdown, SizeOf(LossBreakdown), 0);

  for i := 0 to SeqLen - 1 do begin
    if IsNan(RowLoss[i]) or IsInfinite(RowLoss[i]) then
      raise Exception.CreateFmt('ComputeCELossGPU: invalid row loss at position %d.', [i]);

    Tok := TargetTokens[i];

    LossBreakdown.TotalLossSum := LossBreakdown.TotalLossSum + RowLoss[i];
    Inc(LossBreakdown.TotalCount);

    // UD grammatical tokens.
    if (Tok >= 260) and (Tok <= 399) then begin
      LossBreakdown.UDLossSum := LossBreakdown.UDLossSum + RowLoss[i];
      Inc(LossBreakdown.UDCount);
    end

    // Exclude BOS, EOS, PAD, and UNK from lexical loss.
    else if (Tok < 256) or (Tok > 259) then begin
      LossBreakdown.LexicalLossSum := LossBreakdown.LexicalLossSum + RowLoss[i];
      Inc(LossBreakdown.LexicalCount);
    end;
  end;

  if LossBreakdown.TotalCount > 0 then
    Result := LossBreakdown.TotalLossSum / LossBreakdown.TotalCount
  else
    Result := 0.0;
end;

{function ComputeCELossGPU(dProbs: PSingle; dTargetTokens: PInteger; dRowLoss: PSingle; var RowLoss: TRowLossVector): Double;
var
  i: Integer;
begin
  LaunchCELossRows(dProbs, dTargetTokens, dRowLoss, SeqLen, nVocab, DimVocab);

  if DebugCudaChecks then
    CheckCudaError('Launch CELossRows.');

  cudaMemcpy(@RowLoss[0], dRowLoss, SeqLen * SizeOf(Single), cudaMemcpyDeviceToHost);

  if DebugCudaChecks then
    CheckCudaError('Copy CE row losses.');

  Result := 0.0;

  for i := 0 to SeqLen - 1 do begin
    if IsNan(RowLoss[i]) or IsInfinite(RowLoss[i]) then
      raise Exception.CreateFmt('ComputeCELossGPU: invalid row loss at position %d.', [i]);

    Result := Result + RowLoss[i];
  end;

  Result := Result / SeqLen;
end;}

// Run a forward pass and return cross-entropy loss without backpropagation.
function ComputeLossOnly(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const InputTokens, TargetTokens: TIDimVector; var LossBreakdown: TLossBreakdown): Double;
var
  Blk: Integer;
  RowLoss: TRowLossVector;
begin
  with WModelParams do with WModelState do begin

    // Copy the supplied diagnostic input tokens to the GPU.
    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    if DebugCudaChecks then
      CheckCudaError('Copy diagnostic input tokens.');

    LaunchEmbeddingLookup(Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

    for Blk := 0 to nBlock - 1 do begin
      RunTransformForward(WModelParams, WModelState, Blk);

      if Blk < nBlock - 1 then
        cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    if DebugCudaChecks then
      CheckCudaError('Copy diagnostic target tokens.');

    RunOutputForward(WModelParams, WModelState);

    Result := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss, TargetTokens, RowLoss, LossBreakdown);
  end;
end;

{function ComputeLossOnly(var WModelParams: TWModelParams; var WModelState: TWModelState; const InputTokens, TargetTokens: TIDimVector): Double;
var
  Blk: Integer;
  RowLoss: TRowLossVector;
begin
  with WModelParams do with WModelState do begin

    // Copy the supplied diagnostic input tokens to the GPU.
    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);
    if DebugCudaChecks then
      CheckCudaError('Copy diagnostic input tokens.');

    LaunchEmbeddingLookup(Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

    for Blk := 0 to nBlock - 1 do begin
      RunTransformForward(WModelParams, WModelState, Blk);

      if Blk < nBlock - 1 then
        cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    if DebugCudaChecks then
      CheckCudaError('Copy diagnostic target tokens.');

    RunOutputForward(WModelParams, WModelState);

    Result := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss, RowLoss);
  end;
end;}

{ Diagnostics }
// Check the first values of a device vector for NaN or extreme values.
procedure CheckFirstValuesFinite(const Name: string; Ptr: PSingle; Count: Integer);
var
  Temp: array[0..31] of Single;
  i, n: Integer;
begin
  n := Count;
  if n > 32 then n := 32;

  cudaMemcpy(@Temp[0], Ptr, n * SizeOf(Single), cudaMemcpyDeviceToHost);
  if DebugCudaChecks then
    CheckCudaError('copy ' + Name);

  for i := 0 to n - 1 do begin
    if (Temp[i] <> Temp[i]) or (Temp[i] > 1.0e20) or (Temp[i] < -1.0e20) then
      raise Exception.CreateFmt('BAD VALUE in %s[%d] = %g', [Name, i, Temp[i]]);
  end;
end;

{ Training windows }
// Build the target vector one token ahead of the input vector.
procedure BuildTargetVector(var Target: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Target[i] := TokenizedCorpus[StartIndex + i + 1];
end;

// Build the input vector for one training window.
procedure BuildInputVector(var Input: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Input[i] := TokenizedCorpus[StartIndex + i];
end;

// Build valid starting positions for all windows in an epoch.
procedure BuildWindowStarts(const TokenCount, SeqLen, Stride, Epoch, StartStride: Integer; var Starts: TIVector);
var
  Start, Count: Integer;
begin
  SetLength(Starts, 0);

  Start := (Epoch * StartStride) mod Stride;
  Count := 0;

  while (Start + SeqLen + 1) <= TokenCount do begin
    SetLength(Starts, Count + 1);
    Starts[Count] := Start;
    Inc(Count);

    Inc(Start, Stride);
  end;
end;

// Compute sampled mean CE loss before training begins.
function ComputeSampledMeanLoss(var WModelParams: TWModelParams; var WModelState: TWModelState; const TokenizedCorpus: TIVector;
  const EpochForStarts, SampleCount: Integer): Double;
var
  i, Index, Start, Count: Integer;
  LossSum: Double;
  Starts: TIVector;
begin
  BuildWindowStarts(Length(TokenizedCorpus), SeqLen, Stride, EpochForStarts, StartStride, Starts);

  if Length(Starts) = 0 then begin
    Result := 0.0;
    Exit;
  end;

  LossSum := 0.0;
  Count := SampleCount;

  if Count > Length(Starts) then
    Count := Length(Starts);

  if Count = 1 then begin
    Start := Starts[0];

    BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);

    Result := ComputeLossOnly(WModelParams, WModelState, InputTokens, TargetTokens, LossBreakdown);
    Exit;
  end;

  for i := 0 to Count - 1 do begin
    Index := Round(Double(i) * High(Starts) / (Count - 1));
    Start := Starts[Index];

    BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);

    LossSum := LossSum +
      ComputeLossOnly(WModelParams, WModelState, InputTokens, TargetTokens, LossBreakdown);
  end;

  Result := LossSum / Count;
end;

{ Window shuffling }
// Swap two integer values.
procedure Swap(var A, B: Integer);
var
  C: Integer;
begin
  C := A;
  A := B;
  B := C;
end;

// Shuffle the training-window starting positions.
procedure ShuffleStarts(var Starts: TIVector);
var
  i, j: Integer;
begin
  if Length(Starts) <= 1 then Exit;

  for i := High(Starts) downto 1 do begin
    j := Random(i + 1);   // 0..i
    Swap(Starts[i], Starts[j]);
  end;
end;

{ Training initialization }
// Initialize counters and learning settings for a new model.
procedure InitializeNewTrainingSettings;
begin
  GlobalStep := 0;
  AdamWStep := 0;
  CompletedEpochs := 1;
  GlobalSeed := 123456789;

  LearningStyle := SlowLearning;
  LearningRate := 0.000100;
  OverrideLearningRate := -1.0;
end;

{ Main training workflow }
// Train a new or resumed model on the supplied tokenized corpus.
procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState; const TokenizedCorpus: TIVector);
const
  ln2 = 0.693147180559945309;
  MinWindowsPerEpoch = 100;
  BaselineSampleCount = 256;
var
  i, j, k: Integer;
  Blk, Epoch, FirstEpoch, OldLastBestSaveEpoch, Start, WindowCount, MaxStride: Integer;
  Loss, DiffLoss, BitsPerByte, EpochLoss, MeanEpochLoss, LastMeanEpochLoss, OldBestSavedLoss: Double;
  TokensPerSecond, ElapsedEpochTime: Double;
  TargetLearningRate: Double;
  Starts: TIVector;
  NeedParamCopy, WasNewModel: Boolean;
  AdamParamRMS, AdamUpdateRMS, AdamUpdateRatio, AdamMRMS, AdamSqrtVRMS: Double;
  EpochTime: TDateTime;
  EpochInterval: Integer = 10;
  CompactStats: TCompactTensorStats;
  WarmupSteps : Integer;
  AdaptiveLRReason: string = 'No change in adaptive learning yet.';
  {WeakX7Vector: TX7EigenVector;       For dim repurposing.
  WeakEigenvalue: Double;
  NormalLoss, ProjectedLoss: Double;}

  // Report the current model and training specifications.
  procedure ReportModelSpecsInfo;   // Report alpha param if any for ELUs.
  begin
    Writeln('>>Model ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), ': nTC = ', Length(TokenizedCorpus), '; nCorpus =  ', nCorpus, '; nVocab = ', nVocab,
      '; DimVocab = ', DimVocab, '; Seqlen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock = ', nBlock, '; Proj = ', ProjText,
      '; Shuffle = ', ShuffleWindows, ';');
    Write('  Tokenizer = ', TokenizerKindName(TokenizerKind), '; Gating = ', MLPGatingKindName(MLPGatingKind), '; Activation = ', EffectiveMLPActivationName,
    '; Normalization = ', NormKindName(NormKind), '; DropOut = ', Training);
    if Training then
      Write(' (', ADropOut: 4: 3, ' ', MLPDropOut: 4: 3, ' ', RDropOut: 4: 3, ')');
    Writeln('.');
  end;

  // Report current AdamW statistics.
  procedure AdamWReportInfo;
  begin
    Writeln('>>AdamW: Step = ', AdamWStep, '; Param RMS = ', AdamParamRMS: 0: 8, '; Update RMS = ', AdamUpdateRMS: 0: 10,
    '; Update = ', 100.0 * AdamUpdateRatio: 0: 5, '%; M RMS = ', AdamMRMS: 0: 8, '; Sqrt(V) RMS = ', AdamSqrtVRMS: 0: 8, '.');
  end;

  // Display interactive training commands.
  procedure ReportOptions;
  begin
    Writeln('>>', DateTimeToStr(Now), '  D = set Dropout rate. I = get program Info. L = set Learning rate. N = go to iNference. P = Pause. S = Save model. ',
      'T = set sTride. V = toggle Verbose mode. W = set Weight decay. X = eXit training.');
  end;

  // Report the completed epoch loss and learning rate.
  procedure ReportEpochInfo;
  begin
    Write(TimeToStr(Now), ' Epoch ', Epoch, ' ended. Steps = ', GlobalStep - WindowCount + 1, '..', GlobalStep, '. Best epoch = ', MinLossEpoch, '. LR = ', LearningRate: 9: 7,
      '. Mean loss = ',  MeanEpochLoss: 8: 6, '; Perplexity = ', exp(MeanEpochLoss): 8: 6, '; Bits per byte = ', BitsPerByte: 0: 4);
    Write('; Loss ');
    if DiffLoss = 0 then
      Writeln('same.')
    else if DiffLoss > 0 then
      Writeln('better by ', DiffLoss: 8: 6, '.')
    else
      Writeln('worse by ', -DiffLoss: 8: 6, '.');

    // UD performance.
    if TokenizerKind = UDTokenizer then begin
      Write('UD performance: ');
      if TotalLexicalCount > 0 then
        Write('Lexical loss: ', LexicalLoss: 0: 6, ', Tokens: ', TotalLexicalCount);
      if TotalUDCount > 0 then
        Write('; UD loss: ', UDLoss: 0: 6, ', Tokens: ', TotalUDCount);
      if (TotalLexicalCount + TotalUDCount) > 0 then
        Write('; UD targets: ', 100.0 * TotalUDCount / (TotalLexicalCount + TotalUDCount): 0: 2, '%');
      Writeln('.');
    end;
  end;

  // Report interval timing and training throughput.
  procedure EpochIntervalReportInfo;
  begin
    Write('>>Window # = ', WindowCount);
    if Epoch > 0 then
      Writeln('; Epoch time = ', Trunc(ElapsedEpochTime) div 3600,
        ' hours, ', (Trunc(ElapsedEpochTime) mod 3600) div 60, ' minutes, ', (ElapsedEpochTime mod 60): 0: 4,
        ' seconds; Training speed = ', TokensPerSecond: 0: 0, ' tokens/second.')
    else
      Writeln('.');
  end;

  // Process one interactive training command if a control key was pressed.
  function TrainReadIfKeyPressed: Boolean;
  var
    key: char;
    j: Integer;
    Dropout: Single;
  begin
    Result := False;
    key := CheckForControlKey;
    case key of
      'd', 'D': begin        // Change dropout rates.
        Write('Enter dropout rates: ');
        Readln(DropOut);
        ADropout := DropOut;
        MLPDropout := DropOut;
        RDropout := DropOut;
        Writeln('New dropout rates = ', Dropout: 4: 3, '. Training...');
      end;
      'i', 'I': begin        // Display training info.
        ReportProgramInfo;
        EpochIntervalReportInfo;
        ReportOptions;
        PauseNNL;
      end;
      'l', 'L': begin        // Override learning rate.
        Write('Enter override learning rate: ');
        Readln(OverrideLearningRate);
        Writeln('New learning rate = ', OverrideLearningRate: 9: 7, '. Training...');
      end;
      'n', 'N': begin        // Exit training. Go to inference.
        Writeln('Stopping training. Going to inference...');
        TrainSuccess := True;
        Result := True;
        StopTraining := True;
      end;
      'p', 'P': begin        // Pause work.
        Write('Paused... ');
        PauseNNL;
      end;
      's', 'S': begin        // Save model.
        Write('Enter model filename, blank for automatic checkpoint: ');
        Readln(ModelFileName);

        ModelFileName := Trim(ModelFileName);

        if ModelFileName = '' then begin
          if Trim(WorkingName) = '' then WorkingName := 'weschat';

          ModelFileName := ModelDir + WorkingName + '_epoch' + IntToStr(Epoch) +
            '_step' + IntToStr(GlobalStep) + '_' + FormatDateTime('yyyy-mm-dd_hhnss', Now) + '.model';
        end
        else begin
          if ExtractFileExt(ModelFileName) = '' then
            ModelFileName := ModelFileName + '.model';

          if not IsAbsolutePath(ModelFileName) then
            ModelFileName := ModelDir + ModelFileName
          else
            ModelFileName := ExpandFileName(ModelFileName);
        end;

        if SaveModel(ModelFileName, WModelParams, WAdamWState) then begin
          ModelPresent := True;
          Writeln('Model saved. File = ', ModelFileName, '.');
        end
        else
          Writeln('File not saved. Training...');
      end;
      't', 'T': begin
        Write('Current stride = ', Stride, '. Enter new stride: ');
        Readln(j);
        MaxStride := (Length(TokenizedCorpus) - SeqLen) div MinWindowsPerEpoch;

        if j <= 0 then
          Writeln('Stride must be greater than zero.')
        else if j > MaxStride then
          Writeln('Stride is too large. Maximum stride for at least ',
            MinWindowsPerEpoch, ' windows per epoch is ', MaxStride, '.')
        else begin
          Stride := j;
          Writeln('New stride = ', Stride, '. It will take effect next epoch.');
        end;
      end;
      'v', 'V': begin        // Enable verbose transform.
        VerboseTransform := not VerboseTransform;
        Writeln('Verbose transform mode: ', VerboseTransform, '. Training...');
      end;
      'w', 'W': begin        // Override weight decay.
        Write('Enter new weight decay: ');
        Readln(WeightDecay);
        Writeln('New weight decay = ', WeightDecay: 8: 6, '. Training...');
      end;
      'x', 'X': begin        // Exit training. Success, go to main menu.
        Writeln('Exit requested. Stopping training.');
        TrainSuccess := False;
        Result := True;
        StopTraining := True;
      end;
    end;
  end;

begin
  { Training session initialization }
  // Configure automatic best-model saving.
  LastBestSaveEpoch := -1000000000;    // Sentinel for no best model saved.
  AutoSaveEpoch := 10;                 // Start automatic saving of the best model at this epoch.
  BestSavedLoss := MaxDouble;          // The prior best saved loss.
  MinSaveGap := 5;                     // Wait this many epochs to save a best model again.
  MinSaveDelta := 0.00025;             // Smallest loss improvement needed for saving best model.

  // Set gating text to display.
  if MLPGatingKind = SplitSwiGLU then
    ProjText := '8/3'
  else
    ProjText := '4';

  // Getting working name right.
  if Trim(WorkingName) = '' then begin
    Writeln('WorkingName is blank. Using "weschat".');
    WorkingName := 'weschat';
  end;

  { Training log startup }
  // Start logging.
  OpenTrainLog(LogDir + WorkingName + '_train.log');
  LogTrainingStart(BestModelFileName);

  // Initialize loss and timing state.
  MeanEpochLoss := 0;
  EpochTime := Now;
  ElapsedEpochTime := 0.0;
  MinLoss := 1000000;
  MinLossEpoch := -1;

  // Initialize speed statistics.
  TokensPerSecond := 0.0;

  // Initialize training control state.
  StopTraining := False;
  Training := True;
  WasNewModel := NewModel;
  if WasNewModel then begin
    InitializeAdaptiveLRState(AdaptiveLRState);
    InitializeNewTrainingSettings;
  end;

  // Guard against an unset corpus byte count.
  if nCorpus = 0 then begin
    nCorpus := 1000000;
    Writeln('Empty corpus. nCorpus set at 1000000.');
    Pause;
  end;

  // Verify that the physical vocabulary capacity is large enough.
  if nVocab > DimVocab then begin
    Writeln('nVocab > DimVocab. Aborting training...');
    TrainSuccess := False;
    Exit;
  end;

  { Model and CUDA initialization }

  // A new model must not reuse the preceding model's CUDA storage.
  if WasNewModel and (CudaAllocated or (CuHandle <> nil)) then
    EndCuda(WModelParams, WModelState, WAdamWState);

  // Initialize transformer state.
  InitializeTransformerState(WModelState);

  // Initialize parameters for a new model and determine whether host parameters must be copied.
  NeedParamCopy := (not CudaAllocated) or ParamsNeedCopyToDevice or WasNewModel;
  if WasNewModel then
    InitializeTransformerParams(WModelParams);
  NewModel := False;

  // Initizlize target learning.
  TargetLearningRate := LearningRate;

  // Start CUDA and allocate model, state, and AdamW device buffers.
  StartCuda(WModelParams, WModelState, WAdamWState);
  Writeln('--- Training ---');

  try
    if NeedParamCopy then begin
      CopyParamsToDevice(WModelParams);
      ParamsNeedCopyToDevice := False;
    end;

    // Copy RoPE inverse frequencies to the device.
    CopyInvFreqToDevice(WModelState);

    // Initialize or restore AdamW after its device buffers exist.
    if WasNewModel then begin
      InitializeWAdamWState(WAdamWState);
      AdamWStep := 0;
      AdamWStateLoaded := True;
    end
    else if AdamWStateLoaded then begin
      // WES2 model containing saved AdamW state.
      CopyAdamWStateToDevice(WAdamWState);
    end
    else begin
      // WES1 model: loaded weights, but no AdamW history.
      InitializeWAdamWState(WAdamWState);
      AdamWStep := 0;
      AdamWStateLoaded := True;
    end;

    // Initialize the epoch loop.
    Start := 0;
    FirstEpoch := CompletedEpochs;

    // Determine warmup length from the number of updates in the first epoch.
    BuildWindowStarts(Length(TokenizedCorpus), SeqLen, Stride, FirstEpoch, StartStride, Starts);
    WarmupSteps := Max(1, Length(Starts) div 2);
    Write('Warmup steps = ', WarmupSteps, ' (half of ', Length(Starts), ' training windows in first epoch). ');
    Writeln('Learning-rate warmup will use ', WarmupSteps, ' optimizer steps.');

    { Baseline evaluation }
    // Estimate baseline loss from evenly spaced windows across the corpus.
    LastMeanEpochLoss := ComputeSampledMeanLoss(WModelParams, WModelState, TokenizedCorpus, FirstEpoch, BaselineSampleCount);

    // Report the baseline and training settings.
    Writeln('>>Estimated baseline loss before training = ', LastMeanEpochLoss: 8: 6, '; Perplexity = ', Exp(LastMeanEpochLoss): 0: 7,
      '; Bits per byte = ', LastMeanEpochLoss * RawTokenCount / nCorpus / Ln2: 0: 7, '.');
    ReportModelSpecsInfo;
    ReportOptions;
    Writeln('Training started...');

    // Optionally display embeddings before training.
    if VerboseTransform then begin
      Writeln('Display Embeddings.Value prior to Transform.');
      DisplayX(WModelParams.Embeddings.Value, B);
      Pause;
    end;

    // x7 correlation initialization.  For dim repurposing.
    {with WModelState do begin
      SetLength(X7Stats, nBlock);
      for k := 0 to High(X7Stats) do begin
        cudaMemset(X7Stats[k].dCross, 0, WeightSize);
        cudaMemset(X7Stats[k].dSum, 0, ModelSize);
        X7Stats[k].SampleCount := 0;
      end;
    end;}

    { Epoch training loop }
    // Loop through epochs.
    for Epoch := FirstEpoch to MaxEpochs - 1 do with WModelState do begin
      EpochLoss := 0;
      WindowCount := 0;

      // Build all valid window starting positions for this epoch.
      BuildWindowStarts(Length(TokenizedCorpus), SeqLen, Stride, Epoch, StartStride, Starts);

      // Initialize UD performance vars.
      TotalLexicalLossSum := 0;
      TotalUDLossSum := 0;
      TotalLexicalCount := 0;
      TotalUDCount := 0;

      // Shuffle window order when enabled.
      if ShuffleWindows then
        ShuffleStarts(Starts);

      // Loop through the training windows.
      for i := 0 to High(Starts) do begin
        Start := Starts[i];

        if TrainReadIfKeyPressed then begin
          StopTraining := True;
          Break;
        end;

        // Optionally display the current training window.
        if DisplayWindow then
          Writeln('Epoch is ', Epoch, ' SeqLen window is from ', Start, ' to ', Start + SeqLen, ' on loop ', WindowCount);

        // Build the target vector, one ahead, for the loss stage.
        BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);
        cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

        // Build the input vector.
        BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
        cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

        // Optionally inspect input tokens and the first embedding row.
        if VerboseTransform then begin
          Writeln('Checking tokens');
          for j := 0 to SeqLen - 1 do
            Write(j: 4, ' token=', InputTokens[j], ' ');
          Writeln('Values for row zero.');
          k := InputTokens[0];
          for j := 0 to 15 do
            Write(WModelParams.Embeddings.Value[k, j]: 8: 5, ' ');
         Pause;
        end;

        // Look up input-token embeddings into block 0 X.
        LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);

        // Display X.Value matrix.
        if VerboseTransform then begin
          cudaMemcpy(@WModelState.StateBlock[0].X.Value[0,0], StateBlock[0].X.dValue, XSize, cudaMemcpyDeviceToHost);
          VTPDisplayX('Display X.Value before transform.', StateBlock[0].X.Value, G);
          VTPDisplayX('Display X.Value before transform.', StateBlock[0].X.Value, B);
        end;

        // Optional transformer-style embedding scaling by sqrt(d_model).
        CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

        // Zero gradients.
        for k := 0 to nBlock - 1 do
          ZeroGradients(WModelParams, WModelState, k);
        if DebugCudaChecks then
          CheckCudaError('Zero transformer state and parameter gradients.');

        // Forward pass through all transformer blocks.
        for Blk := 0 to nBlock - 1 do begin
          RunTransformForward(WModelParams, WModelState, Blk);

          if Blk < nBlock - 1 then
            cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
        end;

        // Run the output-head forward pass.
        RunOutputForward(WModelParams, WModelState);

        // Compute mean cross-entropy loss on the GPU and copy only SeqLen losses.
        Loss := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss, TargetTokens, RowLoss, LossBreakdown);

        // Accumulate UD and lexical loss over epoch.
        TotalLexicalLossSum := TotalLexicalLossSum + LossBreakdown.LexicalLossSum;
        TotalUDLossSum := TotalUDLossSum + LossBreakdown.UDLossSum;

        Inc(TotalLexicalCount, LossBreakdown.LexicalCount);
        Inc(TotalUDCount, LossBreakdown.UDCount);

        // Compute the logit gradient, X7 gradient, and output-side tied-embedding gradient.
        RunOutputBackward(WModelParams, WModelState);

        if VerboseTransform then
          Writeln('            Switch from Forward to Backprop');

        // Backpropagate through all transformer blocks.
        for Blk := nBlock - 1 downto 0 do with WModelState do begin
          RunTransformBackprop(WModelParams, WModelState, Blk);

          if Blk > 0 then
            cudaMemcpy(StateBlock[Blk - 1].X7.dGrad, StateBlock[Blk].X.dGrad, XSize, cudaMemcpyDeviceToDevice);
        end;

        // Clip parameter gradients before optimization.
        for k := 0 to nBlock - 1 do
          with WModelParams.ParamBlock[k] do begin
            LaunchClipVector(Wq.dGrad, ModelDim * ModelDim, ClipLimit);
            LaunchClipVector(Wk.dGrad, ModelDim * ModelDim, ClipLimit);
            LaunchClipVector(Wv.dGrad, ModelDim * ModelDim, ClipLimit);
            LaunchClipVector(W0.dGrad, ModelDim * ModelDim, ClipLimit);

            LaunchClipVector(W1.dGrad, ModelDim * ModelDimProj, ClipLimit);
            LaunchClipVector(W2.dGrad, ModelDimProj * ModelDim, ClipLimit);

            LaunchClipVector(b1.dGrad, ModelDimProj, ClipLimit);
            LaunchClipVector(b2.dGrad, ModelDim, ClipLimit);

            LaunchClipVector(Gamma1.dGrad, ModelDim, ClipLimit);
            LaunchClipVector(Beta1.dGrad, ModelDim, ClipLimit);
            LaunchClipVector(Gamma2.dGrad, ModelDim, ClipLimit);
            LaunchClipVector(Beta2.dGrad, ModelDim, ClipLimit);
          end;
        if DebugCudaChecks then
          CheckCudaError('Gradient clipping.');

        if VerboseTransform then begin
          cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);

          if DebugCudaChecks then
            CheckCudaError('Copy probabilities for verbose display.');

          VTPDisplayX('Display Probs after softmax.', Probs, B);
        end;

        // Select the target learning rate for this update.
        // Manual override has highest priority and bypasses warmup.
        if OverrideLearningRate <> -1.0 then begin
          TargetLearningRate := OverrideLearningRate;
          LearningRate := TargetLearningRate;
        end
        else begin

          // Adaptive learning retains its current target learning rate.
          if AdaptiveLR then
          // TargetLearningRate is maintained by ApplyAdaptiveLR.
          // Otherwise use the selected schedule.
          else begin
            case LearningStyle of
              FlatLearning: TargetLearningRate := 0.0003;

              SlowLearning:
                case Epoch of
                  1..2:       TargetLearningRate := 0.000100;
                  3..7:       TargetLearningRate := 0.000075;
                  8..15:      TargetLearningRate := 0.000050;
                  16..30:     TargetLearningRate := 0.000025;
                  31..60:     TargetLearningRate := 0.000015;
                  61..100:    TargetLearningRate := 0.000010;
                  else         TargetLearningRate := 0.000005;
                end;

              FastLearning:
                case Epoch of
                  1..2:       TargetLearningRate := 0.00030;
                  3..10:      TargetLearningRate := 0.00020;
                  11..30:     TargetLearningRate := 0.00010;
                  31..100:    TargetLearningRate := 0.000050;
                  101..400:   TargetLearningRate := 0.000025;
                  401..800:   TargetLearningRate := 0.000010;
                  else         TargetLearningRate := 0.000005;
                end;

              RolledOffLearning:
                TargetLearningRate := FloorLearningRate + (BaseLearningRate - FloorLearningRate) * Power(RollOff, GlobalStep);
            end;
          end;

          // Linear warmup for a newly initialized model.
          if WasNewModel and (AdamWStep < WarmupSteps) then
            LearningRate := TargetLearningRate * (AdamWStep + 1) / WarmupSteps
          else
            LearningRate := TargetLearningRate;
        end;

        // Perform AdamW optimization.
        Beta1Power := Single(Power(AdamBeta1, AdamWStep + 1));
        Beta2Power := Single(Power(AdamBeta2, AdamWStep + 1));

        // Update embedding gradients with AdamW.
        UpdateEmbeddingGradient(WModelParams, WModelState);

        // Update transformer block parameters with AdamW.
        for k := 0 to nBlock - 1 do
          AdamWOptimizeBlock(WModelParams, WAdamWState, k, Beta1Power, Beta2Power);

        // Update tied embeddings.
        AdamWOptimizeEmbeddings(WModelParams, WAdamWState, Beta1Power, Beta2Power);
        Inc(AdamWStep);

        // Optionally check updated parameters for invalid values.
        if DebugCudaChecks then begin
          CheckFirstValuesFinite('Embeddings', WModelParams.Embeddings.dValue, nVocab * ModelDim);
          for k := 0 to nBlock - 1 do with WModelParams.ParamBlock[k] do begin
            CheckFirstValuesFinite('Wq', Wq.dValue, ModelDim * ModelDim);
            CheckFirstValuesFinite('W1', W1.dValue, ModelDim * ModelDimProj);
            CheckFirstValuesFinite('W2', W2.dValue, ModelDimProj * ModelDim);
            CheckFirstValuesFinite('Gamma1', Gamma1.dValue, ModelDim);
            CheckFirstValuesFinite('Gamma2', Gamma2.dValue, ModelDim);
          end;
          CheckCudaError('Training parameter update.');
        end;

        // Accumulate window loss and advance the global step.
        EpochLoss := EpochLoss + Loss;
        Inc(WindowCount);
        Inc(GlobalStep);
      end;    // End sequence loop.

      if StopTraining then begin
        CloseTrainLog;
        Exit;
      end;

      // Compute mean lexical and UD losses for the entire epoch.
      if TotalLexicalCount > 0 then
        LexicalLoss := TotalLexicalLossSum / TotalLexicalCount
      else
        LexicalLoss := 0.0;
      if TotalUDCount > 0 then
        UDLoss := TotalUDLossSum / TotalUDCount
      else
        UDLoss := 0.0;

      // Compute mean loss and derived epoch statistics.
      if WindowCount = 0 then begin
        Writeln('Epoch ', Epoch, ' contained no training windows.');
        Continue;
      end;
      MeanEpochLoss := EpochLoss / WindowCount;
      DiffLoss := LastMeanEpochLoss - MeanEpochLoss;
      LastMeanEpochLoss := MeanEpochLoss;

      CompletedEpochs := Epoch + 1;
      BitsPerByte := MeanEpochLoss * RawTokenCount / nCorpus / Ln2;

      { Epoch completion }
      // Save a new best model when the configured conditions are met.
      if MeanEpochLoss < MinLoss then begin
        MinLoss := MeanEpochLoss;
        MinLossEpoch := Epoch;

        if (Epoch >= AutoSaveEpoch) and
          ((Epoch - LastBestSaveEpoch) >= MinSaveGap) and ((BestSavedLoss = MaxDouble) or ((BestSavedLoss - MeanEpochLoss) >= MinSaveDelta)) then begin

          // Report first and subsequent best-model saves.
          if LastBestSaveEpoch < 0 then
            Write('--Saving first best model in epoch ', Epoch)
          else
            Write('--Saving new best model. Previous saved best = ',
              BestSavedLoss:9:7, ' in epoch ', LastBestSaveEpoch,
              '; new best = ', MeanEpochLoss:9:7, ' in epoch ', Epoch);

          OldLastBestSaveEpoch := LastBestSaveEpoch;
          OldBestSavedLoss := BestSavedLoss;

          // Set before SaveModel so these values are included in the checkpoint.
          LastBestSaveEpoch := Epoch;
          BestSavedLoss := MeanEpochLoss;

          if SaveModel(BestModelFileName, WModelParams, WAdamWState) then begin
            Writeln('. Best model saved: ', BestModelFileName, '.');

            TrainLog('Best model saved. Epoch=' + IntToStr(Epoch) +
              '; MEL=' + FloatToStrF(MeanEpochLoss, ffFixed, 10, 6) +
              '; File=' + BestModelFileName);
          end
          else begin
            // Save failed, so restore the previous saved-best state.
            LastBestSaveEpoch := OldLastBestSaveEpoch;
            BestSavedLoss := OldBestSavedLoss;

            Writeln('. Best model not saved.');
            TrainLog('Best model save FAILED. Epoch=' + IntToStr(Epoch) +
              '; MEL=' + FloatToStrF(MeanEpochLoss, ffFixed, 10, 6) +
              '; File=' + BestModelFileName);
          end;
        end;
      end;

      // Mark whether the epoch loss improved.
      if DiffLoss >= 0 then       // Loss gets better or stays same.
        Write('^^')
      else if DiffLoss < 0 then
        Write('vv');              // Loss gets worse.

      // Report the completed epoch.
      ReportEpochInfo;

      // Report detailed progress every EpochInterval epochs.
      if (Epoch mod EpochInterval) = 0 then begin

        // Compute average epoch time over the reporting interval.
        ElapsedEpochTime := (Now - EpochTime) * 86400.0 / EpochInterval;
        EpochTime := Now;

        ReportModelSpecsInfo;

        // Compute and report AdamW statistics.
        GetAdamWStatistics(WModelParams, WAdamWState, AdamParamRMS, AdamUpdateRMS, AdamUpdateRatio, AdamMRMS, AdamSqrtVRMS);
        AdamWReportInfo;

        // Report the active learning-rate mode.
        if OverrideLearningRate <> -1.0 then begin
          Writeln('>>Learning rate: Override = ', LearningRate: 8: 6, '.');
        end
        else if not AdaptiveLR then begin
          case LearningStyle of
            FlatLearning: Write('>>Learning rate: Flat = ', LearningRate: 9: 7, '.');

            SlowLearning: Write('>>Learning rate: Slow = ', LearningRate: 9: 7, ' with 1..2: 0.000100; 3..7: 0.000075; ',
              '8..15: 0.000050; 16..30: 0.000025; 31..60: 0.000015; 61..100: 0.000010; else 0.000005.');

            FastLearning: Write('>>Learning rate: Fast = ', LearningRate: 9: 7, ' with 1..2: 0.00030; 3..10: 0.00020; ',
              '11..30: 0.00010; 31..100: 0.000050; 101..400: 0.000025; 401..800: 0.000010; else 0.000005.');

            RolledOffLearning: Write('>>Learning rate: Rolloff = ', LearningRate: 9: 7,
              ' Floor LR = ', FloorLearningRate: 9: 7, ' Base LR = ', BaseLearningRate: 9: 7, ' LR rolloff = ', RollOff: 9: 7, '.');
          end;
          Writeln(' WeightDecay = ', WeightDecay: 6: 4, '; Clip = ', ClipLimit: 6: 4, '.');
        end;

        // Compute training throughput.
        if ElapsedEpochTime > 0.0 then begin
          TokensPerSecond := (WindowCount * SeqLen) / ElapsedEpochTime;
        end;

        // Report interval timing and throughput.
        EpochIntervalReportInfo;

        // Report full tensor statistics when verbose.
        if VerboseTransform then
          ReportAdamWTensorStatistics(WModelParams, WAdamWState);
        // Report compact tensor statistics.
        ReportCompactTensorStatistics(WModelParams, CompactStats);

        // Compute new adaptive LR for the next epoch.
        with CompactStats do
          if AdaptiveLR and (OverrideLearningRate = -1.0) and ((not WasNewModel) or (AdamWStep >= WarmupSteps)) then begin
            ApplyAdaptiveLR(AdaptiveLRState, TargetLearningRate, FloorLearningRate, MeanEpochLoss, MinLoss, DiffLoss,
              AdamParamRMS, AdamUpdateRatio, AdamMRMS, AdamSqrtVRMS, MaxGammaRMS, Epoch, AdaptiveLRReason);
            Writeln('>>Learning rate: Adaptive target = ', TargetLearningRate: 9: 7, ' for next epoch. ', AdaptiveLRReason);
          end;
      end;      // End interval report.

      // Log the completed epoch.
      LogTrainingEpoch(CompletedEpochs, MeanEpochLoss, BitsPerByte, DiffLoss, LearningRate);

      // Display the X7 correlations.
      {if RepurposeDimsOn then begin
        {for Blk := 0 to High(X7Stats) do begin
          Writeln('Block ', Blk);
          ReportMostCorrelatedX7Dims(X7Stats[Blk], ModelDim, 5);
        end;}
        // ReportMostCorrelatedX7Pairs(WModelState.X7Stats, ModelDim, 10);
        // ReportX7CovarianceSpectrum(X7Stats, ModelDim); Most recent before weakest stuff.
        GetWeakestX7Eigenvector(X7Stats[nBlock - 1], ModelDim,
          WeakX7Vector, WeakEigenvalue);

        Writeln('Final-block weakest eigenvalue = ', WeakEigenvalue:12:6);

        {Writeln('Length WeakX7Vector = ', Length(WeakX7Vector));
        Writeln('dWeakX7Vector       = ', PtrUInt(dWeakX7Vector));
        Writeln('ModelDim            = ', ModelDim);
        Writeln('Bytes               = ', ModelDim * SizeOf(Single));}
        cudaMemcpy(dWeakX7Vector, @WeakX7Vector[0],    // poroblem
          ModelDim * SizeOf(Single), cudaMemcpyHostToDevice);

        CheckCudaError('Copy weak X7 vector to GPU');

        ReportWeakVectorGPU(dWeakX7Vector, ModelDim);

        // Save the original X7.
        cudaMemcpy(dX7Backup, StateBlock[nBlock - 1].X7.dValue,
          XSize, cudaMemcpyDeviceToDevice);

        // Recompute output from ORIGINAL X7.
        RunOutputForward(WModelParams, WModelState);

        NormalLoss := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss,
          TargetTokens, RowLoss, LossBreakdown);

        // Project out weakest direction.
        LaunchProjectOutDirection(StateBlock[nBlock - 1].X7.dValue,
          dWeakX7Vector, SeqLen, ModelDim);

        CheckCudaError('LaunchProjectOutDirection launch');

        cudaDeviceSynchronize;
        CheckCudaError('LaunchProjectOutDirection execution');

        ReportX7ProjectionChange(dX7Backup,
          StateBlock[nBlock - 1].X7.dValue, SeqLen, ModelDim);

        // Recompute output from PROJECTED X7.
        RunOutputForward(WModelParams, WModelState);

        ProjectedLoss := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss,
          TargetTokens, RowLoss, LossBreakdown);

        // Restore X7.
        cudaMemcpy(StateBlock[nBlock - 1].X7.dValue, dX7Backup,
          XSize, cudaMemcpyDeviceToDevice);

        // Restore output probabilities too.
        RunOutputForward(WModelParams, WModelState);

        Writeln('Normal loss           = ', NormalLoss:10:6);
        Writeln('Projected loss        = ', ProjectedLoss:10:6);
        Writeln('Loss change           = ', ProjectedLoss - NormalLoss:10:6);

      end;}
    end    // End epoch loop.

  except
    on E: Exception do begin
      TrainSuccess := False;
      Writeln('Training Error: ', E.ClassName, '; ', E.Message, '.');
      CloseTrainLog;
      Pause;
      Exit;
    end;
  end;

  Writeln;
  CloseTrainLog;
  Writeln('Training ended.');
  TrainSuccess := True;
  Pause;
end;

end.
