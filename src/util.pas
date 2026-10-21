unit Util;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  Math, SysUtils,
  { WesChat units }
  Display, Global, Matrix;

const
  WeightSize: Integer = ModelDim * ModelDim * SizeOf(Single);
  WeightProjectedSize: Integer = ModelDim * ModelDimProj * SizeOf(Single);
  ProjectedSize: Integer = ModelDimProj * SizeOf(Single);
  SeqSize: Integer = SeqLen * SizeOf(Single);
  ModelSize: Integer = ModelDim * SizeOf(Single);
  XSize: Integer = SeqLen * ModelDim * SizeOf(Single);
  HiddenSize: Integer = SeqLen * ModelDimProj * SizeOf(Single);
  ScoresSize: Integer = SeqLen * SeqLen * SizeOf(Single);
  EmbeddingsSize: Integer = DimVocab * ModelDim * SizeOf(Single);
  InvFreqSize: Integer = (HeadDim div 2) * SizeOf(Single);
  ProbsSize: Integer = SeqLen * DimVocab * SizeOf(Single);

type
  // Compact tensor report.
  TCompactTensorStats = record
    EmbParamRMS: Double;
    EmbGradRMS: Double;
    EmbMaxParam: Double;
    EmbMaxGrad: Double;
    EmbClippedPercent: Double;
    EmbClippedCount: Integer;
    MaxWeightGradRMS: Double;
    MaxGammaRMS: Double;
  end;

{ CUDA and cuBLAS lifecycle }
procedure InitializeCublas;
procedure CheckCudaError(const Where: string);
procedure StartCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
procedure EndCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);

{ Work-folder paths }
function IsAbsolutePath(const FolderName: string): Boolean;
procedure InitWorkFolders(const Root: string);
function CleanBaseName(const FileName: string): string;
function WorkSymbolFile(const BaseName: string): string;
function WorkTokenFile(const BaseName: string): string;
function WorkModelFile(const BaseName: string): string;
function WorkLogFile(const BaseName: string): string;
function WorkRunFile(const BaseName: string): string;

{ Token and loss utilities }
procedure PadToSeqMultiple(var TokenVectorToPad: TIVector; const Seq: Integer);
function ComputeLoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;

{ Model initialization and CUDA storage }
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
procedure InitializeTransformerState(var WModelState: TWModelState);
procedure InitializeTransformerParams(var WModelParams: TWModelParams);
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
procedure CopyParamsToHost(var WModelParams: TWModelParams);
procedure CopyAdamWStateToHost(var WAdamWState: TWAdamWState);
procedure CopyAdamWStateToDevice(var WAdamWState: TWAdamWState);
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
procedure InitializeWAdamWState(var WAdamWState: TWAdamWState);
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
procedure ResetTrainingStateForEnhancement(var WAdamWState: TWAdamWState);

{ AdamW optimization and diagnostics }
procedure AdamWOptimizeBlock(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Blk: Integer;
  const Beta1Power, Beta2Power: Single);
procedure ReinitializeBeta(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState);
procedure ResetAdamWStateForEnhancement(var WAdamWState: TWAdamWState);
procedure AdamWOptimizeEmbeddings(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Beta1Power, Beta2Power: Single);
procedure UpdateEmbeddingGradient(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure GetAdamWStatistics(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState;
  out ParamRMS, UpdateRMS, UpdateRatio, MRMS, SqrtVRMS: Double);
procedure ReportCompactTensorStatistics(var WModelParams: TWModelParams; out Stats: TCompactTensorStats);  // Do not need Epoch param.
procedure ReportAdamWTensorStatistics(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState);
procedure GetClippedGradientPercent(dGrad: PSingle; const Count: Integer;
  const ClipLimit: Single; out ClippedCount: Integer; out ClippedPercent: Double);

{ Adaptive learning rate }
procedure InitializeAdaptiveLRState(var LRState: TAdaptiveLRState);
procedure ApplyAdaptiveLR(var LRState: TAdaptiveLRState; var LearningRate: Double;  const FloorLR, CurrentLoss, BestLoss, RollingImprovement,
  ParamRMS, UpdateRatio, MRMS, SqrtVRMS, MaxGammaRMS: Double; const Epoch: Integer; out Reason: string);

{ CUDA kernel entry points }
procedure LaunchReLUForward(A: PSingle; B: PSingle; Rows: Integer; Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchReLUBackward(Hidden1: PSingle; GradOut: PSingle; GradIn: PSingle; Rows: Integer; Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchGELUForward(A: PSingle; B: PSingle; Rows: Integer; Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchGELUBackward(Hidden1: PSingle; GradOut: PSingle; GradIn: PSingle; Rows: Integer; Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchSiLUForward(A: PSingle; B: PSingle; Rows: Integer; Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchSiLUBackward(Hidden1: PSingle; GradOut: PSingle; GradIn: PSingle; Rows: Integer; Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchSelfGatedSiLUBackward(Hidden1, GradHidden2, GradHidden1: PSingle; Rows, Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchLeakyReLUForward(A, B: PSingle; Rows, Cols: Integer; Alpha: Single); cdecl; external WesChatKernelDLL;
procedure LaunchLeakyReLUBackward(Hidden1, GradOut, GradIn: PSingle; Rows, Cols: Integer; Alpha: Single); cdecl; external WesChatKernelDLL;
procedure LaunchELUForward(A, B: PSingle; Rows, Cols: Integer; Alpha: Single); cdecl; external WesChatKernelDLL;
procedure LaunchELUBackward(Hidden1, GradOut, GradIn: PSingle; Rows, Cols: Integer; Alpha: Single); cdecl; external WesChatKernelDLL;
procedure LaunchSoftplusForward(A, B: PSingle; Rows, Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchSoftplusBackward(Hidden1, GradOut, GradIn: PSingle; Rows, Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchMishForward(A, B: PSingle; Rows, Cols: Integer); cdecl; external WesChatKernelDLL;
procedure LaunchMishBackward(Hidden1, GradOut, GradIn: PSingle; Rows, Cols: Integer); cdecl; external WesChatKernelDLL;

procedure LaunchClipVector(X: PSingle; N: Integer; Limit: Single);
  cdecl; external WesChatKernelDLL;
procedure LaunchEmbeddingLookup(Embeddings: PSingle; InputTokens: PInteger; X: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchAutoRegressiveMask(Scores: PSingle; SeqLen: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchAutoRegressiveMaskBackward(ScoresGrad: PSingle; SeqLen: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchDropout(X: PSingle; N: Integer; DropProb: Single; Seed: UInt64);
  cdecl; external WesChatKernelDLL;
procedure LaunchDropoutBackward(dX: PSingle; N: Integer; DropProb: Single; Seed: UInt64);
  cdecl; external WesChatKernelDLL;
procedure LaunchDropoutStrided(X: PSingle; Rows, Cols, Pitch, Offset: Integer; DropProb: Single; Seed: UInt64);
  cdecl; external WesChatKernelDLL;
procedure LaunchDropoutBackwardStrided(dX: PSingle; Rows, Cols, Pitch, Offset: Integer; DropProb: Single; Seed: UInt64);
  cdecl; external WesChatKernelDLL;
procedure LaunchSoftmaxForwardStrided(dIn: PSingle; dOut: PSingle; Rows: Integer; Cols: Integer; RowStride: Integer; Temperature: Single);
  cdecl; external WesChatKernelDLL;
procedure LaunchSoftmaxForward(dIn: PSingle; dOut: PSingle; Rows: Integer; N: Integer; Temperature: Single);
procedure LaunchSoftmaxBackward(Y: PSingle; dY: PSingle; dX: PSingle; Rows: Integer; D: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchLayerNormForward(InX, OutX, Gamma, Beta, LNXhat, LNInvStd: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchLayerNormBackward(dY, dX, Gamma, LNXhat, LNInvStd, dGamma, dBeta: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchRMSNormForward(InX, OutX, Gamma, RMSXHat, RMSInvStd: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchRMSNormBackward(dY, dX, Gamma, RMSXHat, RMSInvStd, dGamma: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchRoPEForward(H: PSingle; const InvFreq: PSingle; SeqLen: Integer; NumHeads: Integer; HeadDim: Integer; RowStride: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchRoPEBackward(dH: PSingle; const InvFreq: PSingle; SeqLen: Integer; NumHeads: Integer; HeadDim: Integer; RowStride: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchCEGradientStrided(Probs: PSingle; TopGradient: PSingle; TargetTokens: PInteger; Rows: Integer; VocabSize: Integer; RowStride: Integer; GradScale: Single);
  cdecl; external WesChatKernelDLL;
procedure LaunchAddInputEmbeddingGrad(XGrad: PSingle; EmbGrad: PSingle; InputTokens: PInteger; SeqLen: Integer; ModelDim: Integer; nVocab: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchAddBiasRows(X: PSingle; Bias: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchAddBiasRowsBackward(dX: PSingle; dBias: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchCELossRows(Probs: PSingle; Targets: PInteger; RowLoss: PSingle; Rows, VocabSize, RowStride: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchAdamWUpdate(Param, Grad, M, V: PSingle; Count: Integer; LearningRate, Beta1, Beta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay: Single);
  cdecl; external WesChatKernelDLL;
procedure LaunchAdd(A, B: PSingle; N: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchSwiGLUSplit(Hidden1, Hidden2: PSingle; Rows, HiddenWidth, GatedDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchHadamard(A, B, C: PSingle; N: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchProjectOutDirection(X, V: PSingle; Rows, ModelDim: Integer);
  cdecl; external WesChatKernelDLL;
procedure LaunchSwiGLUSplitBackward(Hidden1, Hidden2, GradHidden2, GradHidden1: PSingle; Rows, HiddenWidth, GatedDim: Integer);
  cdecl; external WesChatKernelDLL;

implementation

type
  TStatBuffer = array of Single;

{ DLL availability }
// Return True if a DLL can be loaded.
function CheckDLL(const LibName: string): Boolean;
var
  DLLHandle: THandle;
begin
  // Attempt to load the library.
  DLLHandle := LoadLibrary(PChar(LibName));

  // If the handle is non-zero, it loaded successfully.
  Result := (DLLHandle <> 0);

  // Clean up if it was successfully loaded.
  if Result then
    FreeLibrary(DLLHandle);
end;

// Verify that the CUDA, cuBLAS, and WesChat kernel DLLs are available.
procedure CheckAllDLLs;
begin
  CublasPresent := CheckDLL(cublasDLL);
  CudartPresent := CheckDLL(cudartDLL);
  WesChatKernelPresent := CheckDLL(WesChatKernelDLL);
  if not CublasPresent or not CudartPresent or not WesChatkernelPresent then begin
      Writeln('One of the required DLLs is not present: ', cublasDLL, ', ', cudartDLL, ', or ', WesChatKernelDLL, '.');
      Pause;
      Halt;
  end;
end;

{ CUDA and cuBLAS lifecycle }
// Initialize cuBLAS after verifying the required DLLs.
procedure InitializeCublas;
begin
  CheckAllDLLs;
  if CuBLAS_Init then
      Writeln('CuBLAS successfully initiated.')
  else begin
    Writeln('Error initiating CuBLAS. Halting....');
    Pause;
    Halt;
  end;
end;

// Check both the pending CUDA launch error and synchronized execution status.
procedure CheckCudaError(const Where: string);
var
  Err: Integer;
begin
  Err := cudaGetLastError;
  if Err <> 0 then begin
    Writeln;
    Writeln('CUDA LAUNCH ERROR. Location: ', Where, '; Error # : ', Err, '; Message : ', StrPas(cudaGetErrorString(Err)), '.');
    Pause;
    Exit;
  end;

  Err := cudaDeviceSynchronize;
  if Err <> 0 then begin
    Writeln;
    Writeln('CUDA EXECUTION ERROR. Location: ', Where, '; Error # : ', Err, '; Message : ', StrPas(cudaGetErrorString(Err)), '.');
    Pause;
  end;
end;

// Halt when a CUDA runtime call returns an error status.
procedure CheckCudaStatus(const Status: Integer; const Where: string);
begin
  if Status <> 0 then begin
    Writeln('CUDA error in ', Where, ': ', StrPas(cudaGetErrorString(Status)), '.');
    Pause;
    Halt;
  end;
end;

// Initialize cuBLAS and allocate CUDA buffers.
procedure StartCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
begin
  InitializeCublas;
  if not CudaAllocated then
    MAllocCublas(WModelParams, WModelState, WAdamWState);
  CheckCudaError('Cuda started.');
end;

// Deallocate CUDA buffers and shut down cuBLAS.
procedure EndCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
begin
  if CudaAllocated then
    MDeallocateCublas(WModelParams, WModelState, WAdamWState);
  if CuBLAS_Shutdown then
    Writeln('CuBLAS successfully shut down.')
end;

{ Work-folder paths }
// Return True if a path is drive-qualified or rooted.
function IsAbsolutePath(const FolderName: string): Boolean;
begin
  Result := (ExtractFileDrive(FolderName) <> '') or ((Length(FolderName) > 0) and ((FolderName[1] = '\') or (FolderName[1] = '/')));
end;

// Initialize and create the standard work-folder tree.
procedure InitWorkFolders(const Root: string);
begin
  WorkRoot := IncludeTrailingPathDelimiter(ExpandFileName(Root));

  CorpusDir  := WorkRoot + 'corpus'  + DirectorySeparator;
  SymbolDir  := WorkRoot + 'symbols' + DirectorySeparator;
  MergeDir   := WorkRoot + 'merges'  + DirectorySeparator;
  TokenDir   := WorkRoot + 'tokens'  + DirectorySeparator;
  ModelDir   := WorkRoot + 'models'  + DirectorySeparator;
  LogDir     := WorkRoot + 'logs'    + DirectorySeparator;
  RunDir     := WorkRoot + 'runs'    + DirectorySeparator;
  ListDir    := WorkRoot + 'lists'   + DirectorySeparator;
  ScratchDir := WorkRoot + 'scratch' + DirectorySeparator;

  ForceDirectories(WorkRoot);
  ForceDirectories(CorpusDir);
  ForceDirectories(SymbolDir);
  ForceDirectories(MergeDir);
  ForceDirectories(TokenDir);
  ForceDirectories(ModelDir);
  ForceDirectories(LogDir);
  ForceDirectories(RunDir);
  ForceDirectories(ListDir);
  ForceDirectories(ScratchDir);
end;

// Return a filename without its path or extension.
function CleanBaseName(const FileName: string): string;
begin
  Result := ChangeFileExt(ExtractFileName(FileName), '');
end;

// Build the default symbol-table filename in SymbolDir.
function WorkSymbolFile(const BaseName: string): string;
begin
  Result := SymbolDir + ChangeFileExt(ExtractFileName(BaseName), '.sym');
end;

// Build the default token-list filename in TokenDir.
function WorkTokenFile(const BaseName: string): string;
begin
  Result := TokenDir + ChangeFileExt(ExtractFileName(BaseName), '.tok');
end;

// Build the default model filename in ModelDir.
function WorkModelFile(const BaseName: string): string;
begin
  Result := ModelDir + ChangeFileExt(ExtractFileName(BaseName), '.model');
end;

// Build the default log filename in LogDir.
function WorkLogFile(const BaseName: string): string;
begin
  Result := LogDir + ChangeFileExt(ExtractFileName(BaseName), '.log');
end;

// Build a timestamped run filename in RunDir.
function WorkRunFile(const BaseName: string): string;
begin
  Result := RunDir + BaseName + '_' + FormatDateTime('yyyy-mm-dd_hhnnss', Now) + '.run';
end;

{ Token and loss utilities }
// Pad a token vector to a multiple of Seq using PAD.
procedure PadToSeqMultiple(var TokenVectorToPad: TIVector; const Seq: Integer);
var
  OldLen, NewLen, i: Integer;
begin
  OldLen := Length(TokenVectorToPad);
  NewLen := ((OldLen + Seq - 1) div Seq) * Seq;

  SetLength(TokenVectorToPad, NewLen);

  for i := OldLen to NewLen - 1 do
    TokenVectorToPad[i] := PAD;
end;

// Compute mean cross-entropy loss from host probabilities.
function ComputeLoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;
const
  Eps = 1.0e-12;
var
  i: Integer;
  P: Double;
begin
  Result := 0.0;

  for i := 0 to SeqLen - 1 do begin
    P := Probs[i, TargetTokens[i]];

    if P < Eps then
      P := Eps;

    Result := Result - Ln(P);
  end;

  Result := Result / SeqLen;
end;

{ Model initialization }
// Xavier-Glorot initialization for a D x D weight matrix.
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

// Xavier-Glorot initialization for a D x DB weight matrix.
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);

var
  Limit, r: Single;
  i, j: Integer;
begin
  if MLPGatingKind = SplitSwiGLU then
    Limit := Sqrt(6.0 / (ModelDim + GatedDim))
  else
    Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W2 matrix.
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  if MLPGatingKind = SplitSwiGLU then
    Limit := Sqrt(6.0 / (ModelDim + GatedDim))
  else
    Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;
      W[j, i] := (2 * r - 1) * Limit;
    end;
end;

// Allocate CUDA memory for model parameters, model state, and AdamW state.
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
var
  h, k: Integer;

  procedure AllocSingle(var P: PSingle; const Bytes: Integer; const Name: string);
  begin
    CheckCudaStatus(cudaMalloc(@P, Bytes), Name);
  end;

  procedure AllocInteger(var P: PInteger; const Bytes: Integer; const Name: string);
  begin
    CheckCudaStatus(cudaMalloc(@P, Bytes), Name);
  end;

begin
  CudaAllocated := False;

  // Input and target tokens.
  AllocInteger(dInputTokens, SeqLen * SizeOf(Integer), 'dInputTokens');
  AllocInteger(dTargetTokens, SeqLen * SizeOf(Integer), 'dTargetTokens');

  // Global/shared model parameters.
  with WModelParams do begin
    AllocSingle(Embeddings.dValue, EmbeddingsSize, 'Embeddings.dValue');
    AllocSingle(Embeddings.dGrad, EmbeddingsSize, 'Embeddings.dGrad');
  end;

  // Global/shared AdamW state.
  with WAdamWState.Embeddings do begin
    AllocSingle(dM, EmbeddingsSize, 'Adam Embeddings.dM');
    AllocSingle(dV, EmbeddingsSize, 'Adam Embeddings.dV');
  end;

  // Global/shared model state.
  with WModelState do begin
    AllocSingle(dInvFreq, InvFreqSize, 'dInvFreq');
    AllocSingle(dProbs, ProbsSize, 'dProbs');
    AllocSingle(dRowLoss, SeqSize, 'dRowLoss');
    AllocSingle(dTopGradient, ProbsSize, 'dTopGradient');

    {SetLength(X7Stats, nBlock);  For dim repurposing.

    for k := 0 to nBlock - 1 do begin
      AllocSingle(X7Stats[k].dCross, WeightSize, 'X7Stats.dCross');
      AllocSingle(X7Stats[k].dCrossTemp, WeightSize, 'X7Stats.dCrossTemp');
      AllocSingle(X7Stats[k].dSum, ModelSize, 'X7Stats.dSum');
      AllocSingle(X7Stats[k].dSumTemp, ModelSize, 'X7Stats.dSumTemp');
      X7Stats[k].SampleCount := 0;
    end;
    // AllocSingle(dWeakX7Vector, ModelSize, 'dWeakX7Vector'); For dim repurposing.
    // AllocSingle(dX7Backup, XSize, 'dX7Backup');}
  end;

  // Per-block parameters, AdamW state, and transformer state.
  for k := 0 to nBlock - 1 do begin

    // Trainable model parameters.
    with WModelParams.ParamBlock[k] do begin
      AllocSingle(Wq.dValue, WeightSize, 'Wq.dValue');
      AllocSingle(Wq.dGrad, WeightSize, 'Wq.dGrad');

      AllocSingle(Wk.dValue, WeightSize, 'Wk.dValue');
      AllocSingle(Wk.dGrad, WeightSize, 'Wk.dGrad');

      AllocSingle(Wv.dValue, WeightSize, 'Wv.dValue');
      AllocSingle(Wv.dGrad, WeightSize, 'Wv.dGrad');

      AllocSingle(W0.dValue, WeightSize, 'W0.dValue');
      AllocSingle(W0.dGrad, WeightSize, 'W0.dGrad');

      AllocSingle(W1.dValue, WeightProjectedSize, 'W1.dValue');
      AllocSingle(W1.dGrad, WeightProjectedSize, 'W1.dGrad');

      AllocSingle(W2.dValue, WeightProjectedSize, 'W2.dValue');
      AllocSingle(W2.dGrad, WeightProjectedSize, 'W2.dGrad');

      AllocSingle(b1.dValue, ProjectedSize, 'b1.dValue');
      AllocSingle(b1.dGrad, ProjectedSize, 'b1.dGrad');

      AllocSingle(b2.dValue, ModelSize, 'b2.dValue');
      AllocSingle(b2.dGrad, ModelSize, 'b2.dGrad');

      AllocSingle(Gamma1.dValue, ModelSize, 'Gamma1.dValue');
      AllocSingle(Gamma1.dGrad, ModelSize, 'Gamma1.dGrad');

      AllocSingle(Beta1.dValue, ModelSize, 'Beta1.dValue');
      AllocSingle(Beta1.dGrad, ModelSize, 'Beta1.dGrad');

      AllocSingle(Gamma2.dValue, ModelSize, 'Gamma2.dValue');
      AllocSingle(Gamma2.dGrad, ModelSize, 'Gamma2.dGrad');

      AllocSingle(Beta2.dValue, ModelSize, 'Beta2.dValue');
      AllocSingle(Beta2.dGrad, ModelSize, 'Beta2.dGrad');
    end;

    // Persistent AdamW first and second moments.
    with WAdamWState.ParamBlock[k] do begin
      AllocSingle(Wq.dM, WeightSize, 'Adam Wq.dM');
      AllocSingle(Wq.dV, WeightSize, 'Adam Wq.dV');

      AllocSingle(Wk.dM, WeightSize, 'Adam Wk.dM');
      AllocSingle(Wk.dV, WeightSize, 'Adam Wk.dV');

      AllocSingle(Wv.dM, WeightSize, 'Adam Wv.dM');
      AllocSingle(Wv.dV, WeightSize, 'Adam Wv.dV');

      AllocSingle(W0.dM, WeightSize, 'Adam W0.dM');
      AllocSingle(W0.dV, WeightSize, 'Adam W0.dV');

      AllocSingle(W1.dM, WeightProjectedSize, 'Adam W1.dM');
      AllocSingle(W1.dV, WeightProjectedSize, 'Adam W1.dV');

      AllocSingle(W2.dM, WeightProjectedSize, 'Adam W2.dM');
      AllocSingle(W2.dV, WeightProjectedSize, 'Adam W2.dV');

      AllocSingle(b1.dM, ProjectedSize, 'Adam b1.dM');
      AllocSingle(b1.dV, ProjectedSize, 'Adam b1.dV');

      AllocSingle(b2.dM, ModelSize, 'Adam b2.dM');
      AllocSingle(b2.dV, ModelSize, 'Adam b2.dV');

      AllocSingle(Gamma1.dM, ModelSize, 'Adam Gamma1.dM');
      AllocSingle(Gamma1.dV, ModelSize, 'Adam Gamma1.dV');

      AllocSingle(Beta1.dM, ModelSize, 'Adam Beta1.dM');
      AllocSingle(Beta1.dV, ModelSize, 'Adam Beta1.dV');

      AllocSingle(Gamma2.dM, ModelSize, 'Adam Gamma2.dM');
      AllocSingle(Gamma2.dV, ModelSize, 'Adam Gamma2.dV');

      AllocSingle(Beta2.dM, ModelSize, 'Adam Beta2.dM');
      AllocSingle(Beta2.dV, ModelSize, 'Adam Beta2.dV');
    end;

    // Non-trainable transformer state.
    with WModelState.StateBlock[k] do begin
      AllocSingle(X.dValue, XSize, 'X.dValue');
      AllocSingle(X.dGrad, XSize, 'X.dGrad');

      AllocSingle(X1.dValue, XSize, 'X1.dValue');
      AllocSingle(X1.dGrad, XSize, 'X1.dGrad');

      AllocSingle(X2.dValue, XSize, 'X2.dValue');
      AllocSingle(X2.dGrad, XSize, 'X2.dGrad');

      AllocSingle(X3.dValue, XSize, 'X3.dValue');
      AllocSingle(X3.dGrad, XSize, 'X3.dGrad');

      AllocSingle(X4.dValue, XSize, 'X4.dValue');
      AllocSingle(X4.dGrad, XSize, 'X4.dGrad');

      AllocSingle(X5.dValue, XSize, 'X5.dValue');
      AllocSingle(X5.dGrad, XSize, 'X5.dGrad');

      AllocSingle(X6.dValue, XSize, 'X6.dValue');
      AllocSingle(X6.dGrad, XSize, 'X6.dGrad');

      AllocSingle(X7.dValue, XSize, 'X7.dValue');
      AllocSingle(X7.dGrad, XSize, 'X7.dGrad');

      AllocSingle(X1q.dValue, XSize, 'X1q.dValue');
      AllocSingle(X1q.dGrad, XSize, 'X1q.dGrad');

      AllocSingle(X1k.dValue, XSize, 'X1k.dValue');
      AllocSingle(X1k.dGrad, XSize, 'X1k.dGrad');

      AllocSingle(X1v.dValue, XSize, 'X1v.dValue');
      AllocSingle(X1v.dGrad, XSize, 'X1v.dGrad');

      AllocSingle(Q.dValue, XSize, 'Q.dValue');
      AllocSingle(Q.dGrad, XSize, 'Q.dGrad');

      AllocSingle(K.dValue, XSize, 'K.dValue');
      AllocSingle(K.dGrad, XSize, 'K.dGrad');

      AllocSingle(V.dValue, XSize, 'V.dValue');
      AllocSingle(V.dGrad, XSize, 'V.dGrad');

      for h := 0 to nHead - 1 do begin
        AllocSingle(ScoresHead1[h].dValue, ScoresSize, 'ScoresHead1.dValue');
        AllocSingle(ScoresHead1[h].dGrad, ScoresSize, 'ScoresHead1.dGrad');
        AllocSingle(ScoresHead2[h].dValue, ScoresSize, 'ScoresHead2.dValue');
        AllocSingle(ScoresHead2[h].dGrad, ScoresSize, 'ScoresHead2.dGrad');
      end;

      AllocSingle(Hidden1.dValue, HiddenSize, 'Hidden1.dValue');
      AllocSingle(Hidden1.dGrad, HiddenSize, 'Hidden1.dGrad');

      AllocSingle(Hidden2.dValue, HiddenSize, 'Hidden2.dValue');
      AllocSingle(Hidden2.dGrad, HiddenSize, 'Hidden2.dGrad');

      AllocSingle(dLNInvStd1, SeqSize, 'dLNInvStd1');
      AllocSingle(dLNXHat1, XSize, 'dLNXHat1');

      AllocSingle(dLNInvStd2, SeqSize, 'dLNInvStd2');
      AllocSingle(dLNXHat2, XSize, 'dLNXHat2');

      AllocSingle(dX4FromLN2, XSize, 'dX4FromLN2');
      AllocSingle(dXFromLN1, XSize, 'dXFromLN1');
    end;
  end;

  CheckCudaError('Allocate CUDA memory.');
  CudaAllocated := True;
end;

// De-allocate CUDA memory.
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
var
  h, k: Integer;
begin
  CudaAllocated := False;

  // Input and target tokens.
  cudaFree(dInputTokens);  dInputTokens  := nil;
  cudaFree(dTargetTokens); dTargetTokens := nil;

  // Global/shared model parameters.
  with WModelParams do begin
    cudaFree(Embeddings.dValue); Embeddings.dValue := nil;
    cudaFree(Embeddings.dGrad);  Embeddings.dGrad  := nil;
  end;

  // Global/shared AdamW state.
  with WAdamWState.Embeddings do begin
    cudaFree(dM); dM := nil;
    cudaFree(dV); dV := nil;
  end;

  // Global/shared model state.
  with WModelState do begin
    cudaFree(dInvFreq);     dInvFreq     := nil;
    cudaFree(dProbs);       dProbs       := nil;

    if dRowLoss <> nil then begin
      cudaFree(dRowLoss);
      dRowLoss := nil;
    end;

    cudaFree(dTopGradient); dTopGradient := nil;

    {// X7 correlation statistics for dim repurposing.
    for k := 0 to High(X7Stats) do begin
      if X7Stats[k].dCross <> nil then begin
        cudaFree(X7Stats[k].dCross);
        X7Stats[k].dCross := nil;
      end;

      if X7Stats[k].dCrossTemp <> nil then begin
        cudaFree(X7Stats[k].dCrossTemp);
        X7Stats[k].dCrossTemp := nil;
      end;

      if X7Stats[k].dSum <> nil then begin
        cudaFree(X7Stats[k].dSum);
        X7Stats[k].dSum := nil;
      end;

      if X7Stats[k].dSumTemp <> nil then begin
        cudaFree(X7Stats[k].dSumTemp);
        X7Stats[k].dSumTemp := nil;
      end;

      X7Stats[k].SampleCount := 0;
    end;

    SetLength(X7Stats, 0);}
    {if dWeakX7Vector <> nil then begin     For dim repurposing.
      cudaFree(dWeakX7Vector);
      dWeakX7Vector := nil;
    end;

    if dX7Backup <> nil then begin
      cudaFree(dX7Backup);
      dX7Backup := nil;
    end;}
  end;

  // Per-block parameters, AdamW state, and transformer state.
  for k := 0 to nBlock - 1 do begin

    // Trainable model parameters.
    with WModelParams.ParamBlock[k] do begin
      cudaFree(Wq.dValue);     Wq.dValue     := nil;
      cudaFree(Wq.dGrad);      Wq.dGrad      := nil;

      cudaFree(Wk.dValue);     Wk.dValue     := nil;
      cudaFree(Wk.dGrad);      Wk.dGrad      := nil;

      cudaFree(Wv.dValue);     Wv.dValue     := nil;
      cudaFree(Wv.dGrad);      Wv.dGrad      := nil;

      cudaFree(W0.dValue);     W0.dValue     := nil;
      cudaFree(W0.dGrad);      W0.dGrad      := nil;

      cudaFree(W1.dValue);     W1.dValue     := nil;
      cudaFree(W1.dGrad);      W1.dGrad      := nil;

      cudaFree(W2.dValue);     W2.dValue     := nil;
      cudaFree(W2.dGrad);      W2.dGrad      := nil;

      cudaFree(b1.dValue);     b1.dValue     := nil;
      cudaFree(b1.dGrad);      b1.dGrad      := nil;

      cudaFree(b2.dValue);     b2.dValue     := nil;
      cudaFree(b2.dGrad);      b2.dGrad      := nil;

      cudaFree(Gamma1.dValue); Gamma1.dValue := nil;
      cudaFree(Gamma1.dGrad);  Gamma1.dGrad  := nil;

      cudaFree(Beta1.dValue);  Beta1.dValue  := nil;
      cudaFree(Beta1.dGrad);   Beta1.dGrad   := nil;

      cudaFree(Gamma2.dValue); Gamma2.dValue := nil;
      cudaFree(Gamma2.dGrad);  Gamma2.dGrad  := nil;

      cudaFree(Beta2.dValue);  Beta2.dValue  := nil;
      cudaFree(Beta2.dGrad);   Beta2.dGrad   := nil;
    end;

    // Persistent AdamW first and second moments.
    with WAdamWState.ParamBlock[k] do begin
      cudaFree(Wq.dM);     Wq.dM     := nil;
      cudaFree(Wq.dV);     Wq.dV     := nil;

      cudaFree(Wk.dM);     Wk.dM     := nil;
      cudaFree(Wk.dV);     Wk.dV     := nil;

      cudaFree(Wv.dM);     Wv.dM     := nil;
      cudaFree(Wv.dV);     Wv.dV     := nil;

      cudaFree(W0.dM);     W0.dM     := nil;
      cudaFree(W0.dV);     W0.dV     := nil;

      cudaFree(W1.dM);     W1.dM     := nil;
      cudaFree(W1.dV);     W1.dV     := nil;

      cudaFree(W2.dM);     W2.dM     := nil;
      cudaFree(W2.dV);     W2.dV     := nil;

      cudaFree(b1.dM);     b1.dM     := nil;
      cudaFree(b1.dV);     b1.dV     := nil;

      cudaFree(b2.dM);     b2.dM     := nil;
      cudaFree(b2.dV);     b2.dV     := nil;

      cudaFree(Gamma1.dM); Gamma1.dM := nil;
      cudaFree(Gamma1.dV); Gamma1.dV := nil;

      cudaFree(Beta1.dM);  Beta1.dM  := nil;
      cudaFree(Beta1.dV);  Beta1.dV  := nil;

      cudaFree(Gamma2.dM); Gamma2.dM := nil;
      cudaFree(Gamma2.dV); Gamma2.dV := nil;

      cudaFree(Beta2.dM);  Beta2.dM  := nil;
      cudaFree(Beta2.dV);  Beta2.dV  := nil;
    end;

    // Non-trainable transformer state.
    with WModelState.StateBlock[k] do begin
      cudaFree(X.dValue);   X.dValue   := nil;
      cudaFree(X.dGrad);    X.dGrad    := nil;

      cudaFree(X1.dValue);  X1.dValue  := nil;
      cudaFree(X1.dGrad);   X1.dGrad   := nil;

      cudaFree(X2.dValue);  X2.dValue  := nil;
      cudaFree(X2.dGrad);   X2.dGrad   := nil;

      cudaFree(X3.dValue);  X3.dValue  := nil;
      cudaFree(X3.dGrad);   X3.dGrad   := nil;

      cudaFree(X4.dValue);  X4.dValue  := nil;
      cudaFree(X4.dGrad);   X4.dGrad   := nil;

      cudaFree(X5.dValue);  X5.dValue  := nil;
      cudaFree(X5.dGrad);   X5.dGrad   := nil;

      cudaFree(X6.dValue);  X6.dValue  := nil;
      cudaFree(X6.dGrad);   X6.dGrad   := nil;

      cudaFree(X7.dValue);  X7.dValue  := nil;
      cudaFree(X7.dGrad);   X7.dGrad   := nil;

      cudaFree(X1q.dValue); X1q.dValue := nil;
      cudaFree(X1q.dGrad);  X1q.dGrad  := nil;

      cudaFree(X1k.dValue); X1k.dValue := nil;
      cudaFree(X1k.dGrad);  X1k.dGrad  := nil;

      cudaFree(X1v.dValue); X1v.dValue := nil;
      cudaFree(X1v.dGrad);  X1v.dGrad  := nil;

      cudaFree(Q.dValue);   Q.dValue   := nil;
      cudaFree(Q.dGrad);    Q.dGrad    := nil;

      cudaFree(K.dValue);   K.dValue   := nil;
      cudaFree(K.dGrad);    K.dGrad    := nil;

      cudaFree(V.dValue);   V.dValue   := nil;
      cudaFree(V.dGrad);    V.dGrad    := nil;

      for h := 0 to nHead - 1 do begin
        cudaFree(ScoresHead1[h].dValue); ScoresHead1[h].dValue := nil;
        cudaFree(ScoresHead1[h].dGrad);  ScoresHead1[h].dGrad  := nil;

        cudaFree(ScoresHead2[h].dValue); ScoresHead2[h].dValue := nil;
        cudaFree(ScoresHead2[h].dGrad);  ScoresHead2[h].dGrad  := nil;
      end;

      cudaFree(Hidden1.dValue); Hidden1.dValue := nil;
      cudaFree(Hidden1.dGrad);  Hidden1.dGrad  := nil;

      cudaFree(Hidden2.dValue); Hidden2.dValue := nil;
      cudaFree(Hidden2.dGrad);  Hidden2.dGrad  := nil;

      cudaFree(dLNInvStd1); dLNInvStd1 := nil;
      cudaFree(dLNXHat1);   dLNXHat1   := nil;

      cudaFree(dLNInvStd2); dLNInvStd2 := nil;
      cudaFree(dLNXHat2);   dLNXHat2   := nil;

      cudaFree(dX4FromLN2); dX4FromLN2 := nil;
      cudaFree(dXFromLN1);  dXFromLN1  := nil;
    end;
  end;

  ParamsNeedCopyToDevice := True;
end;

// Copy parameters from host to device.
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
var
  k: Integer;
begin
  // Embeddings (global).
  cudaMemcpy(WModelParams.Embeddings.dValue, @WModelParams.Embeddings.Value[0,0], EmbeddingsSize, cudaMemcpyHostToDevice);

  // Params.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin
      // Attention weights.
      cudaMemcpy(Wq.dValue, @Wq.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wk.dValue, @Wk.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wv.dValue, @Wv.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W0.dValue, @W0.Value[0,0], WeightSize, cudaMemcpyHostToDevice);

      // MLP.
      cudaMemcpy(W1.dValue, @W1.Value[0,0], WeightProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W2.dValue, @W2.Value[0,0], WeightProjectedSize, cudaMemcpyHostToDevice);

      // Biases.
      cudaMemcpy(b1.dValue, @b1.Value[0], ProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b2.dValue, @b2.Value[0], ModelSize, cudaMemcpyHostToDevice);

      // LayerNorm.
      cudaMemcpy(Gamma1.dValue, @Gamma1.Value[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta1.dValue,  @Beta1.Value[0],  ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma2.dValue, @Gamma2.Value[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta2.dValue,  @Beta2.Value[0],  ModelSize, cudaMemcpyHostToDevice);
    end;
  ParamsNeedCopyToDevice := False;
  if DebugCudaChecks then
    CheckCudaError('Copy model parameters to device.');
end;

// Copy AdamW first and second moments from host to CUDA device.
procedure CopyAdamWStateToDevice(var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    cudaMemcpy(dM, @M[0, 0], EmbeddingsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, @V[0, 0], EmbeddingsSize, cudaMemcpyHostToDevice);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      cudaMemcpy(Wq.dM, @Wq.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wq.dV, @Wq.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Wk.dM, @Wk.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wk.dV, @Wk.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Wv.dM, @Wv.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wv.dV, @Wv.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      cudaMemcpy(W0.dM, @W0.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W0.dV, @W0.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      // MLP weights.
      cudaMemcpy(W1.dM, @W1.M[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W1.dV, @W1.V[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);

      cudaMemcpy(W2.dM, @W2.M[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W2.dV, @W2.V[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);

      // Biases.
      cudaMemcpy(b1.dM, @b1.M[0], ProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b1.dV, @b1.V[0], ProjectedSize, cudaMemcpyHostToDevice);

      cudaMemcpy(b2.dM, @b2.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b2.dV, @b2.V[0], ModelSize, cudaMemcpyHostToDevice);

      // LayerNorm parameters.
      cudaMemcpy(Gamma1.dM, @Gamma1.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma1.dV, @Gamma1.V[0], ModelSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Beta1.dM, @Beta1.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta1.dV, @Beta1.V[0], ModelSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Gamma2.dM, @Gamma2.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma2.dV, @Gamma2.V[0], ModelSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Beta2.dM, @Beta2.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta2.dV, @Beta2.V[0], ModelSize, cudaMemcpyHostToDevice);
    end;

  if DebugCudaChecks then
    CheckCudaError('Copy AdamW state to device.');
end;

// Copy parameters from device to host (for saving model).
procedure CopyParamsToHost(var WModelParams: TWModelParams);
var
  k: Integer;
begin
  // Embeddings (global).
  cudaMemcpy(@WModelParams.Embeddings.Value[0,0], WModelParams.Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);

  // Params.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin
      // Attention weights.
      cudaMemcpy(@Wq.Value[0,0], Wq.dValue, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wk.Value[0,0], Wk.dValue, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wv.Value[0,0], Wv.dValue, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W0.Value[0,0], W0.dValue, WeightSize, cudaMemcpyDeviceToHost);

      // MLP.
      cudaMemcpy(@W1.Value[0,0], W1.dValue, WeightProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W2.Value[0,0], W2.dValue, WeightProjectedSize, cudaMemcpyDeviceToHost);

      // Biases.
      cudaMemcpy(@b1.Value[0], b1.dValue, ProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@b2.Value[0], b2.dValue, ModelSize, cudaMemcpyDeviceToHost);

      // LayerNorm.
      cudaMemcpy(@Gamma1.Value[0], Gamma1.dValue, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta1.Value[0],  Beta1.dValue,  ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Gamma2.Value[0], Gamma2.dValue, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta2.Value[0],  Beta2.dValue,  ModelSize, cudaMemcpyDeviceToHost);
    end;
  if DebugCudaChecks then
    CheckCudaError('Copy model parameters to host.');
end;

// Copy AdamW first and second moments from CUDA device to host.
procedure CopyAdamWStateToHost(var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    cudaMemcpy(@M[0, 0], dM, EmbeddingsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(@V[0, 0], dV, EmbeddingsSize, cudaMemcpyDeviceToHost);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      cudaMemcpy(@Wq.M[0, 0], Wq.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wq.V[0, 0], Wq.dV, WeightSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Wk.M[0, 0], Wk.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wk.V[0, 0], Wk.dV, WeightSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Wv.M[0, 0], Wv.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wv.V[0, 0], Wv.dV, WeightSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@W0.M[0, 0], W0.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W0.V[0, 0], W0.dV, WeightSize, cudaMemcpyDeviceToHost);

      // MLP weights.
      cudaMemcpy(@W1.M[0, 0], W1.dM, WeightProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W1.V[0, 0], W1.dV, WeightProjectedSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@W2.M[0, 0], W2.dM, WeightProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W2.V[0, 0], W2.dV, WeightProjectedSize, cudaMemcpyDeviceToHost);

      // Biases.
      cudaMemcpy(@b1.M[0], b1.dM, ProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@b1.V[0], b1.dV, ProjectedSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@b2.M[0], b2.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@b2.V[0], b2.dV, ModelSize, cudaMemcpyDeviceToHost);

      // LayerNorm parameters.
      cudaMemcpy(@Gamma1.M[0], Gamma1.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Gamma1.V[0], Gamma1.dV, ModelSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Beta1.M[0], Beta1.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta1.V[0], Beta1.dV, ModelSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Gamma2.M[0], Gamma2.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Gamma2.V[0], Gamma2.dV, ModelSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Beta2.M[0], Beta2.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta2.V[0], Beta2.dV, ModelSize, cudaMemcpyDeviceToHost);
    end;

  if DebugCudaChecks then
    CheckCudaError('Copy AdamW state to host.');
end;

// Copy InvFreq from host to device.
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
begin
  cudaMemcpy(WModelState.dInvFreq, @WModelState.InvFreq[0], InvFreqSize, cudaMemcpyHostToDevice);
  if DebugCudaChecks then
    CheckCudaError('Copy inverse frequencies to device.');
end;

// Initialize non-trainable transformer state needed before CUDA execution.
procedure InitializeTransformerState(var WModelState: TWModelState);
var
  j: Integer;
begin
  // State gradients are zeroed separately before each training window.
  // Do not zero the state values -- that is not necessary.

  // Compute InvFreq.
  SetLength(WModelState.InvFreq, HeadDim div 2);

  for j := 0 to (HeadDim div 2) - 1 do
    WModelState.InvFreq[j] := Exp(-(2.0 * j) / HeadDim * Ln(10000.0));
end;

// Initialize trainable transformer parameters.
procedure InitializeTransformerParams(var WModelParams: TWModelParams);
var
  i, j, k: Integer;
begin
  // Do not zero the param grads -- that is done in zero gradient.
  // As to the param values -- they are zeroed below.

  // Initialize embeddings. Clear inactive capacity, then randomize active vocabulary rows.
  FillChar(WModelParams.Embeddings.Value, EmbeddingsSize, 0);
  for i := 0 to nVocab - 1 do               // Random normal distribution.
    for j := 0 to ModelDim - 1 do           // Mean = 0, SD = 0.02.
      WModelParams.Embeddings.Value[i, j] := RandG(0.0, 0.02); // Only time I use this randomizer.

  // Initialize per-block params.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin

      // Initialize weight matrix W0.
      XGUniformW(W0.Value, ModelDim, ModelDim);

      // Initialize the weights with Xavier-Glorot function.
      XGUniformW(Wq.Value, ModelDim, ModelDim);
      XGUniformW(Wk.Value, ModelDim, ModelDim);
      XGUniformW(Wv.Value, ModelDim, ModelDim);

      // Initialize W1 and W2 weight matrices.
      XGUniformW1(W1.Value, ModelDim, ModelDimProj);
      XGUniformW2(W2.Value, ModelDimProj, ModelDim);

      // Initialize b1 and b2.
      FillChar(b1.Value, ProjectedSize, 0);
      FillChar(b2.Value, ModelSize, 0);

      // Initialize Beta and Gamma, LN 1 and 2, with SD and mean.
      FillChar(Beta1.Value, ModelSize, 0);
      FillChar(Beta2.Value, ModelSize, 0);
      for j := 0 to ModelDim - 1 do begin
        Gamma1.Value[j] := 1.0;
        Gamma2.Value[j] := 1.0;
      end;
    end;
end;

// Initialize AdamW first and second moments to zero for a new optimizer state.
procedure InitializeWAdamWState(var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    cudaMemset(dM, 0, EmbeddingsSize);
    cudaMemset(dV, 0, EmbeddingsSize);
  end;

  // Per-block AdamW moments.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      cudaMemset(Wq.dM, 0, WeightSize);
      cudaMemset(Wq.dV, 0, WeightSize);

      cudaMemset(Wk.dM, 0, WeightSize);
      cudaMemset(Wk.dV, 0, WeightSize);

      cudaMemset(Wv.dM, 0, WeightSize);
      cudaMemset(Wv.dV, 0, WeightSize);

      cudaMemset(W0.dM, 0, WeightSize);
      cudaMemset(W0.dV, 0, WeightSize);

      // MLP weights.
      cudaMemset(W1.dM, 0, WeightProjectedSize);
      cudaMemset(W1.dV, 0, WeightProjectedSize);

      cudaMemset(W2.dM, 0, WeightProjectedSize);
      cudaMemset(W2.dV, 0, WeightProjectedSize);

      // Biases.
      cudaMemset(b1.dM, 0, ProjectedSize);
      cudaMemset(b1.dV, 0, ProjectedSize);

      cudaMemset(b2.dM, 0, ModelSize);
      cudaMemset(b2.dV, 0, ModelSize);

      // LayerNorm parameters.
      cudaMemset(Gamma1.dM, 0, ModelSize);
      cudaMemset(Gamma1.dV, 0, ModelSize);

      cudaMemset(Beta1.dM, 0, ModelSize);
      cudaMemset(Beta1.dV, 0, ModelSize);

      cudaMemset(Gamma2.dM, 0, ModelSize);
      cudaMemset(Gamma2.dV, 0, ModelSize);

      cudaMemset(Beta2.dM, 0, ModelSize);
      cudaMemset(Beta2.dV, 0, ModelSize);
    end;

  if DebugCudaChecks then
    CheckCudaError('Initialize AdamW state.');
end;

// Reinitialize LayerNorm Beta parameters and their AdamW state.
procedure ReinitializeBeta(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  for k := 0 to nBlock - 1 do begin
    // Beta parameter values and gradients on host.
    FillChar(WModelParams.ParamBlock[k].Beta1.Value, ModelSize, 0);
    FillChar(WModelParams.ParamBlock[k].Beta1.Grad, ModelSize, 0);
    FillChar(WModelParams.ParamBlock[k].Beta2.Value, ModelSize, 0);
    FillChar(WModelParams.ParamBlock[k].Beta2.Grad, ModelSize, 0);

    // Beta AdamW moments on host.
    FillChar(WAdamWState.ParamBlock[k].Beta1.M, ModelSize, 0);
    FillChar(WAdamWState.ParamBlock[k].Beta1.V, ModelSize, 0);
    FillChar(WAdamWState.ParamBlock[k].Beta2.M, ModelSize, 0);
    FillChar(WAdamWState.ParamBlock[k].Beta2.V, ModelSize, 0);

    if CudaAllocated then begin
      // Beta parameter values and gradients on device.
      cudaMemset(WModelParams.ParamBlock[k].Beta1.dValue, 0, ModelSize);
      cudaMemset(WModelParams.ParamBlock[k].Beta1.dGrad, 0, ModelSize);
      cudaMemset(WModelParams.ParamBlock[k].Beta2.dValue, 0, ModelSize);
      cudaMemset(WModelParams.ParamBlock[k].Beta2.dGrad, 0, ModelSize);

      // Beta AdamW moments on device.
      cudaMemset(WAdamWState.ParamBlock[k].Beta1.dM, 0, ModelSize);
      cudaMemset(WAdamWState.ParamBlock[k].Beta1.dV, 0, ModelSize);
      cudaMemset(WAdamWState.ParamBlock[k].Beta2.dM, 0, ModelSize);
      cudaMemset(WAdamWState.ParamBlock[k].Beta2.dV, 0, ModelSize);
    end;
  end;

  if DebugCudaChecks and CudaAllocated then
    CheckCudaError('Reinitialize Beta.');
end;

// Reset optimizer and training history before enhancement on a new corpus.
procedure ResetTrainingStateForEnhancement(var WAdamWState: TWAdamWState);
begin
  ResetAdamWStateForEnhancement(WAdamWState);

  GlobalStep := 0;
  AdamWStep := 0;

  CompletedEpochs := 1;

  MinLoss := MaxDouble;
  MinLossEpoch := -1;

  BestSavedLoss := MaxDouble;
  LastBestSaveEpoch := -1;

  FillChar(AdaptiveLRState, SizeOf(AdaptiveLRState), 0);

  GlobalSeed := 123456789;
end;

// Zero device gradients for one transformer block and shared embeddings when Blk = 0.
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h: Integer;
begin
  with WModelState.StateBlock[Blk] do begin
    // Optional host-side diagnostic buffers.
    FillChar(Hidden1.Grad, HiddenSize, 0);
    FillChar(Hidden2.Grad, HiddenSize, 0);

    // SeqLen x ModelDim state gradients.
    cudaMemset(X.dGrad,   0, XSize);
    cudaMemset(X1.dGrad,  0, XSize);
    cudaMemset(X2.dGrad,  0, XSize);
    cudaMemset(X3.dGrad,  0, XSize);
    cudaMemset(X4.dGrad,  0, XSize);
    cudaMemset(X5.dGrad,  0, XSize);
    cudaMemset(X6.dGrad,  0, XSize);
    cudaMemset(X7.dGrad,  0, XSize);

    cudaMemset(Q.dGrad,   0, XSize);
    cudaMemset(K.dGrad,   0, XSize);
    cudaMemset(V.dGrad,   0, XSize);

    cudaMemset(X1q.dGrad, 0, XSize);
    cudaMemset(X1k.dGrad, 0, XSize);
    cudaMemset(X1v.dGrad, 0, XSize);

    // SeqLen x ModelDimProj state gradients.
    cudaMemset(Hidden1.dGrad, 0, HiddenSize);
    cudaMemset(Hidden2.dGrad, 0, HiddenSize);

    // SeqLen x SeqLen attention gradients for each head.
    for h := 0 to nHead - 1 do begin
      cudaMemset(ScoresHead1[h].dGrad, 0, ScoresSize);
      cudaMemset(ScoresHead2[h].dGrad, 0, ScoresSize);
    end;
    // Temporary parameters.
    cudaMemset(dX4FromLN2, 0, XSize);
    cudaMemset(dXFromLN1, 0, XSize);
  end;

  // Embeddings are shared by all blocks, so clear them once.
  if Blk = 0 then
    cudaMemset(WModelParams.Embeddings.dGrad, 0, EmbeddingsSize);

  with WModelParams.ParamBlock[Blk] do begin
    // Attention parameters.
    cudaMemset(Wq.dGrad, 0, WeightSize);
    cudaMemset(Wk.dGrad, 0, WeightSize);
    cudaMemset(Wv.dGrad, 0, WeightSize);
    cudaMemset(W0.dGrad, 0, WeightSize);

    // FFN parameters.
    cudaMemset(W1.dGrad, 0, WeightProjectedSize);
    cudaMemset(W2.dGrad, 0, WeightProjectedSize);
    cudaMemset(b1.dGrad, 0, ProjectedSize);
    cudaMemset(b2.dGrad, 0, ModelSize);

    // Layer-normalization parameters.
    cudaMemset(Gamma1.dGrad, 0, ModelSize);
    cudaMemset(Beta1.dGrad,  0, ModelSize);
    cudaMemset(Gamma2.dGrad, 0, ModelSize);
    cudaMemset(Beta2.dGrad,  0, ModelSize);
  end;
end;

{ AdamW optimization }

// Update one transformer block using AdamW.
procedure AdamWOptimizeBlock(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Blk: Integer;
  const Beta1Power, Beta2Power: Single);
begin
  with WAdamWState.ParamBlock[Blk] do with WModelParams.ParamBlock[Blk] do begin

    // Attention weights. Apply weight decay.
    LaunchAdamWUpdate(Wq.dValue, Wq.dGrad, WAdamWState.ParamBlock[Blk].Wq.dM, WAdamWState.ParamBlock[Blk].Wq.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(Wk.dValue, Wk.dGrad, WAdamWState.ParamBlock[Blk].Wk.dM, WAdamWState.ParamBlock[Blk].Wk.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(Wv.dValue, Wv.dGrad, WAdamWState.ParamBlock[Blk].Wv.dM, WAdamWState.ParamBlock[Blk].Wv.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(W0.dValue, W0.dGrad, WAdamWState.ParamBlock[Blk].W0.dM, WAdamWState.ParamBlock[Blk].W0.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    // MLP weights. Apply weight decay.
    LaunchAdamWUpdate(W1.dValue, W1.dGrad, WAdamWState.ParamBlock[Blk].W1.dM, WAdamWState.ParamBlock[Blk].W1.dV,
      ModelDim * ModelDimProj, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(W2.dValue, W2.dGrad, WAdamWState.ParamBlock[Blk].W2.dM, WAdamWState.ParamBlock[Blk].W2.dV,
      ModelDimProj * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    // Biases. No weight decay.
    LaunchAdamWUpdate(b1.dValue, b1.dGrad, WAdamWState.ParamBlock[Blk].b1.dM, WAdamWState.ParamBlock[Blk].b1.dV,
      ModelDimProj, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    LaunchAdamWUpdate(b2.dValue, b2.dGrad, WAdamWState.ParamBlock[Blk].b2.dM, WAdamWState.ParamBlock[Blk].b2.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    // Normalization 1. Gamma is used by LayerNorm and RMSNorm.
    LaunchAdamWUpdate(Gamma1.dValue, Gamma1.dGrad, WAdamWState.ParamBlock[Blk].Gamma1.dM, WAdamWState.ParamBlock[Blk].Gamma1.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    // Beta is used only by LayerNorm.
    if NormKind = LayerNorm then
      LaunchAdamWUpdate(Beta1.dValue, Beta1.dGrad, WAdamWState.ParamBlock[Blk].Beta1.dM, WAdamWState.ParamBlock[Blk].Beta1.dV,
        ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    // Normalization 2. Gamma is used by LayerNorm and RMSNorm.
    LaunchAdamWUpdate(Gamma2.dValue, Gamma2.dGrad, WAdamWState.ParamBlock[Blk].Gamma2.dM, WAdamWState.ParamBlock[Blk].Gamma2.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    // Beta is used only by LayerNorm.
    if NormKind = LayerNorm then
      LaunchAdamWUpdate(Beta2.dValue, Beta2.dGrad, WAdamWState.ParamBlock[Blk].Beta2.dM, WAdamWState.ParamBlock[Blk].Beta2.dV,
        ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);
  end;

  if DebugCudaChecks then
    CheckCudaError('AdamW optimizer block ' + IntToStr(Blk));
end;

// Reset AdamW params for learning from new corpus.
procedure ResetAdamWStateForEnhancement(var WAdamWState: TWAdamWState);
begin
  AdamWStep := 0;
  InitializeWAdamWState(WAdamWState);
end;

// AdamW optimize embeddings.
procedure AdamWOptimizeEmbeddings(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Beta1Power, Beta2Power: Single);
begin
  with WModelParams.Embeddings do
    LaunchAdamWUpdate(dValue, dGrad, WAdamWState.Embeddings.dM, WAdamWState.Embeddings.dV,
      nVocab * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

  if DebugCudaChecks then
    CheckCudaError('AdamW optimizer embeddings.');
end;

// Accumulate AdamW statistics for one parameter tensor. Param, M, and V are HOST pointers.
procedure AccumulateAdamWStatistics(Param, M, V: PSingle; const Count: Integer; const Decay: Single;
  const Beta1Power, Beta2Power: Double; var SumParam2, SumUpdate2, SumM2, SumV: Double; var TotalCount: Int64);
var
  i: Integer;
  PNew, POld, MV, VV, MHat, VHat, AdamTerm, Update, Denom: Double;
begin
  Denom := 1.0 - LearningRate * Decay;

  for i := 0 to Count - 1 do begin
    PNew := Param^;
    MV := M^;
    VV := V^;

    // Bias-corrected AdamW moments for the most recent update.
    MHat := MV / (1.0 - Beta1Power);
    VHat := VV / (1.0 - Beta2Power);

    if VHat < 0.0 then
      VHat := 0.0;

    AdamTerm := MHat / (Sqrt(VHat) + AdamEpsilon);

    // AdamW kernel did: PNew = POld - LR * (AdamTerm + Decay * POld). Therefore reconstruct POld.
    POld := (PNew + LearningRate * AdamTerm) / Denom;

    Update := PNew - POld;

    SumParam2 := SumParam2 + POld * POld;
    SumUpdate2 := SumUpdate2 + Update * Update;
    SumM2 := SumM2 + MV * MV;
    SumV := SumV + VV;

    Inc(Param);
    Inc(M);
    Inc(V);
  end;

  Inc(TotalCount, Count);
end;

// Compute whole-model AdamW statistics for the most recently completed update.
procedure GetAdamWStatistics(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState;
  out ParamRMS, UpdateRMS, UpdateRatio, MRMS, SqrtVRMS: Double);
var
  k: Integer;
  TotalCount: Int64;
  SumParam2, SumUpdate2, SumM2, SumV: Double;
  Beta1Power, Beta2Power: Double;
begin
  ParamRMS := 0.0;
  UpdateRMS := 0.0;
  UpdateRatio := 0.0;
  MRMS := 0.0;
  SqrtVRMS := 0.0;

  if AdamWStep <= 0 then Exit;

  // Copy current parameters and AdamW moments to host.
  CopyParamsToHost(WModelParams);
  CopyAdamWStateToHost(WAdamWState);

  // AdamWStep is the number of updates already completed.
  Beta1Power := Power(AdamBeta1, AdamWStep);
  Beta2Power := Power(AdamBeta2, AdamWStep);

  SumParam2 := 0.0;
  SumUpdate2 := 0.0;
  SumM2 := 0.0;
  SumV := 0.0;
  TotalCount := 0;

  // Tied embeddings. Weight decay applies.
  AccumulateAdamWStatistics(@WModelParams.Embeddings.Value[0, 0], @WAdamWState.Embeddings.M[0, 0], @WAdamWState.Embeddings.V[0, 0],
    nVocab * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

  // Transformer blocks.
  for k := 0 to nBlock - 1 do begin

    // Attention weights. Weight decay applies.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Wq.Value[0, 0], @WAdamWState.ParamBlock[k].Wq.M[0, 0], @WAdamWState.ParamBlock[k].Wq.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Wk.Value[0, 0], @WAdamWState.ParamBlock[k].Wk.M[0, 0], @WAdamWState.ParamBlock[k].Wk.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Wv.Value[0, 0], @WAdamWState.ParamBlock[k].Wv.M[0, 0], @WAdamWState.ParamBlock[k].Wv.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].W0.Value[0, 0], @WAdamWState.ParamBlock[k].W0.M[0, 0], @WAdamWState.ParamBlock[k].W0.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // MLP weights. Weight decay applies.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].W1.Value[0, 0], @WAdamWState.ParamBlock[k].W1.M[0, 0], @WAdamWState.ParamBlock[k].W1.V[0, 0],
      ModelDim * ModelDimProj, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].W2.Value[0, 0], @WAdamWState.ParamBlock[k].W2.M[0, 0], @WAdamWState.ParamBlock[k].W2.V[0, 0],
      ModelDimProj * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // Biases. No weight decay.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].b1.Value[0], @WAdamWState.ParamBlock[k].b1.M[0], @WAdamWState.ParamBlock[k].b1.V[0],
      ModelDimProj, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].b2.Value[0], @WAdamWState.ParamBlock[k].b2.M[0], @WAdamWState.ParamBlock[k].b2.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // LayerNorm 1. No weight decay.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Gamma1.Value[0], @WAdamWState.ParamBlock[k].Gamma1.M[0], @WAdamWState.ParamBlock[k].Gamma1.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Beta1.Value[0], @WAdamWState.ParamBlock[k].Beta1.M[0], @WAdamWState.ParamBlock[k].Beta1.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // LayerNorm 2. No weight decay.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Gamma2.Value[0], @WAdamWState.ParamBlock[k].Gamma2.M[0], @WAdamWState.ParamBlock[k].Gamma2.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Beta2.Value[0], @WAdamWState.ParamBlock[k].Beta2.M[0], @WAdamWState.ParamBlock[k].Beta2.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power,
      SumParam2, SumUpdate2, SumM2, SumV, TotalCount);
  end;

  if TotalCount > 0 then begin
    ParamRMS := Sqrt(SumParam2 / TotalCount);
    UpdateRMS := Sqrt(SumUpdate2 / TotalCount);
    MRMS := Sqrt(SumM2 / TotalCount);

    // RMS(sqrt(V)) = sqrt(mean(V)).
    SqrtVRMS := Sqrt(SumV / TotalCount);

    if ParamRMS <> 0.0 then
      UpdateRatio := UpdateRMS / ParamRMS;
  end;
end;

procedure UpdateEmbeddingGradient(var WModelParams: TWModelParams; var WModelState: TWModelState);
begin
  with WModelParams do with WModelState do begin

    // Backpropagate through X = Embedding * Scale.
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dGrad);

    // Add input-side embedding gradient to the existing output-side
    // tied-embedding gradient.
    LaunchAddInputEmbeddingGrad(StateBlock[0].X.dGrad, Embeddings.dGrad, dInputTokens, SeqLen, ModelDim, nVocab);

    // Clip the complete tied-embedding gradient.
    LaunchClipVector(Embeddings.dGrad, nVocab * ModelDim, ClipLimit);
  end;
end;

// Report AdamW statistics for one CUDA parameter tensor.
procedure ReportOneAdamWTensor(const Name: string; dParam, dGrad, dM, dV: PSingle; const Count: Integer);
var
  ParamBuf, GradBuf, MBuf, VBuf: TStatBuffer;
  i: Integer;
  P, G, MVal, VVal: Double;
  SumP2, SumG2, SumM2, SumV: Double;
  ParamRMS, GradRMS, MRMS, SqrtVRMS: Double;
  MaxParam, MaxGrad: Double;
begin
  if Count <= 0 then Exit;

  SetLength(ParamBuf, Count);
  SetLength(GradBuf, Count);
  SetLength(MBuf, Count);
  SetLength(VBuf, Count);

  cudaMemcpy(@ParamBuf[0], dParam, Count * SizeOf(Single), cudaMemcpyDeviceToHost);
  cudaMemcpy(@GradBuf[0], dGrad, Count * SizeOf(Single), cudaMemcpyDeviceToHost);
  cudaMemcpy(@MBuf[0], dM, Count * SizeOf(Single), cudaMemcpyDeviceToHost);
  cudaMemcpy(@VBuf[0], dV, Count * SizeOf(Single), cudaMemcpyDeviceToHost);

  if DebugCudaChecks then
    CheckCudaError('Report statistics for ' + Name);

  SumP2 := 0.0;
  SumG2 := 0.0;
  SumM2 := 0.0;
  SumV := 0.0;
  MaxParam := 0.0;
  MaxGrad := 0.0;

  for i := 0 to Count - 1 do begin
    P := ParamBuf[i];
    G := GradBuf[i];
    MVal := MBuf[i];
    VVal := VBuf[i];

    SumP2 := SumP2 + P * P;
    SumG2 := SumG2 + G * G;
    SumM2 := SumM2 + MVal * MVal;

    if VVal > 0.0 then
      SumV := SumV + VVal;

    if Abs(P) > MaxParam then MaxParam := Abs(P);
    if Abs(G) > MaxGrad then MaxGrad := Abs(G);
  end;

  ParamRMS := Sqrt(SumP2 / Count);
  GradRMS := Sqrt(SumG2 / Count);
  MRMS := Sqrt(SumM2 / Count);
  SqrtVRMS := Sqrt(SumV / Count);

  Writeln(Name:18, '  Param RMS=', ParamRMS:11:7, '  Grad RMS=', GradRMS:11:7, '  M RMS=', MRMS:11:7,
    '  sqrt(V) RMS=', SqrtVRMS:11:7, '  Max|P|=', MaxParam:11:7, '  Max|G|=', MaxGrad:11:7);
end;

// Return RMS and maximum absolute value for a CUDA tensor.
procedure GetTensorStats(dPtr: PSingle; const Count: Integer; out RMS, MaxAbs: Double);
var
  Buf: TStatBuffer;
  i: Integer;
  V, Sum2: Double;
begin
  RMS := 0.0;
  MaxAbs := 0.0;

  if Count <= 0 then Exit;

  SetLength(Buf, Count);
  cudaMemcpy(@Buf[0], dPtr, Count * SizeOf(Single), cudaMemcpyDeviceToHost);

  Sum2 := 0.0;

  for i := 0 to Count - 1 do begin
    V := Buf[i];
    Sum2 := Sum2 + V * V;

    if Abs(V) > MaxAbs then
      MaxAbs := Abs(V);
  end;

  RMS := Sqrt(Sum2 / Count);
end;

// Compact per-tensor diagnostic report.
procedure ReportCompactTensorStatistics(var WModelParams: TWModelParams; out Stats: TCompactTensorStats);
var
  k: Integer;
  EmbPRMS, EmbGRMS, EmbMaxP, EmbMaxG: Double;
  EmbClippedPercent: Double;
  EmbClippedCount: Integer;
  WqG, WqMax: Double;
  W1G, W1Max: Double;
  W2G, W2Max: Double;
  G1P, G1Max: Double;
  G2P, G2Max: Double;
begin
  FillChar(Stats, SizeOf(Stats), 0);
  Writeln('++Compact tensor statistics.');

  // Embeddings.
  GetTensorStats(WModelParams.Embeddings.dValue, nVocab * ModelDim, EmbPRMS, EmbMaxP);
  GetTensorStats(WModelParams.Embeddings.dGrad, nVocab * ModelDim, EmbGRMS, EmbMaxG);

  GetClippedGradientPercent(WModelParams.Embeddings.dGrad, nVocab * ModelDim,
    ClipLimit, EmbClippedCount, EmbClippedPercent);

  Stats.EmbParamRMS := EmbPRMS;
  Stats.EmbGradRMS := EmbGRMS;
  Stats.EmbMaxParam := EmbMaxP;
  Stats.EmbMaxGrad := EmbMaxG;
  Stats.EmbClippedCount := EmbClippedCount;
  Stats.EmbClippedPercent := EmbClippedPercent;

  Writeln('++Embeddings: ParamRMS=', EmbPRMS: 9: 7, ', GradRMS=', EmbGRMS: 9: 7, ', MaxP=', EmbMaxP: 9: 7, ', MaxG=', EmbMaxG: 9: 7,
    ', Clipped=', EmbClippedCount, ' (', EmbClippedPercent: 0: 4, '%).');

  // Transformer blocks.
  for k := 0 to nBlock - 1 do begin
    with WModelParams.ParamBlock[k] do begin
      GetTensorStats(Wq.dGrad, ModelDim * ModelDim, WqG, WqMax);
      GetTensorStats(W1.dGrad, ModelDim * ModelDimProj, W1G, W1Max);
      GetTensorStats(W2.dGrad, ModelDimProj * ModelDim, W2G, W2Max);

      GetTensorStats(Gamma1.dValue, ModelDim, G1P, G1Max);
      GetTensorStats(Gamma2.dValue, ModelDim, G2P, G2Max);

      // Largest weight-gradient RMS in the entire transformer.
      if WqG > Stats.MaxWeightGradRMS then Stats.MaxWeightGradRMS := WqG;
      if W1G > Stats.MaxWeightGradRMS then Stats.MaxWeightGradRMS := W1G;
      if W2G > Stats.MaxWeightGradRMS then Stats.MaxWeightGradRMS := W2G;

      // Largest Gamma RMS in the entire transformer.
      if G1P > Stats.MaxGammaRMS then Stats.MaxGammaRMS := G1P;
      if G2P > Stats.MaxGammaRMS then Stats.MaxGammaRMS := G2P;

      Writeln('++Block ', k, ': WqG=', WqG: 9: 7, ', W1G=', W1G: 9: 7, ', W2G=', W2G: 9: 7, ', G1P=', G1P: 9: 7, ', G2P=', G2P: 9: 7, '.');
    end;
  end;

  Writeln('++Max weight GradRMS=', Stats.MaxWeightGradRMS: 9: 7, '; Max Gamma RMS=', Stats.MaxGammaRMS: 9: 7, '.');
  Writeln;
end;

// Return percentage of gradient elements at the clipping limit.
procedure GetClippedGradientPercent(dGrad: PSingle; const Count: Integer;
  const ClipLimit: Single; out ClippedCount: Integer; out ClippedPercent: Double);
const
  Tolerance = 0.000001;
var
  Buf: TStatBuffer;
  i: Integer;
begin
  ClippedCount := 0;
  ClippedPercent := 0.0;

  if Count <= 0 then Exit;

  SetLength(Buf, Count);
  cudaMemcpy(@Buf[0], dGrad, Count * SizeOf(Single), cudaMemcpyDeviceToHost);

  for i := 0 to Count - 1 do
    if Abs(Abs(Buf[i]) - ClipLimit) <= Tolerance then
      Inc(ClippedCount);

  ClippedPercent := 100.0 * ClippedCount / Count;
end;

// Report AdamW statistics tensor by tensor.
procedure ReportAdamWTensorStatistics(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  Writeln;
  Writeln('--- AdamW per-tensor statistics ---');
  Writeln('AdamW step = ', AdamWStep, '; Learning rate = ', LearningRate:0:7);

  // Tied embeddings.
  ReportOneAdamWTensor('Embeddings', WModelParams.Embeddings.dValue,
    WModelParams.Embeddings.dGrad, WAdamWState.Embeddings.dM,
    WAdamWState.Embeddings.dV, nVocab * ModelDim);

  for k := 0 to nBlock - 1 do begin
    Writeln;
    Writeln('Block ', k, ':');

    with WModelParams.ParamBlock[k] do begin

      // Attention weights.
      ReportOneAdamWTensor('Wq', Wq.dValue, Wq.dGrad,
        WAdamWState.ParamBlock[k].Wq.dM, WAdamWState.ParamBlock[k].Wq.dV,
        ModelDim * ModelDim);

      ReportOneAdamWTensor('Wk', Wk.dValue, Wk.dGrad,
        WAdamWState.ParamBlock[k].Wk.dM, WAdamWState.ParamBlock[k].Wk.dV,
        ModelDim * ModelDim);

      ReportOneAdamWTensor('Wv', Wv.dValue, Wv.dGrad,
        WAdamWState.ParamBlock[k].Wv.dM, WAdamWState.ParamBlock[k].Wv.dV,
        ModelDim * ModelDim);

      ReportOneAdamWTensor('W0', W0.dValue, W0.dGrad,
        WAdamWState.ParamBlock[k].W0.dM, WAdamWState.ParamBlock[k].W0.dV,
        ModelDim * ModelDim);

      // MLP weights.
      ReportOneAdamWTensor('W1', W1.dValue, W1.dGrad,
        WAdamWState.ParamBlock[k].W1.dM, WAdamWState.ParamBlock[k].W1.dV,
        ModelDim * ModelDimProj);

      ReportOneAdamWTensor('W2', W2.dValue, W2.dGrad,
        WAdamWState.ParamBlock[k].W2.dM, WAdamWState.ParamBlock[k].W2.dV,
        ModelDimProj * ModelDim);

      // Biases.
      ReportOneAdamWTensor('b1', b1.dValue, b1.dGrad,
        WAdamWState.ParamBlock[k].b1.dM, WAdamWState.ParamBlock[k].b1.dV,
        ModelDimProj);

      ReportOneAdamWTensor('b2', b2.dValue, b2.dGrad,
        WAdamWState.ParamBlock[k].b2.dM, WAdamWState.ParamBlock[k].b2.dV,
        ModelDim);

      // LayerNorm 1.
      ReportOneAdamWTensor('Gamma1', Gamma1.dValue, Gamma1.dGrad,
        WAdamWState.ParamBlock[k].Gamma1.dM,
        WAdamWState.ParamBlock[k].Gamma1.dV, ModelDim);

      ReportOneAdamWTensor('Beta1', Beta1.dValue, Beta1.dGrad,
        WAdamWState.ParamBlock[k].Beta1.dM,
        WAdamWState.ParamBlock[k].Beta1.dV, ModelDim);

      // LayerNorm 2.
      ReportOneAdamWTensor('Gamma2', Gamma2.dValue, Gamma2.dGrad,
        WAdamWState.ParamBlock[k].Gamma2.dM,
        WAdamWState.ParamBlock[k].Gamma2.dV, ModelDim);

      ReportOneAdamWTensor('Beta2', Beta2.dValue, Beta2.dGrad,
        WAdamWState.ParamBlock[k].Beta2.dM,
        WAdamWState.ParamBlock[k].Beta2.dV, ModelDim);
    end;
  end;

  Writeln;
  Writeln('--- End AdamW per-tensor statistics ---');
  Writeln;
end;

procedure InitializeAdaptiveLRState(var LRState: TAdaptiveLRState);
begin
  FillChar(LRState, SizeOf(LRState), 0);
  LRState.Initialized := False;
  LRState.LastLRChangeEpoch := -1000;
end;

// Apply adaptive LearningRate.
procedure ApplyAdaptiveLR(var LRState: TAdaptiveLRState; var LearningRate: Double;  const FloorLR, CurrentLoss, BestLoss, RollingImprovement,
  ParamRMS, UpdateRatio, MRMS, SqrtVRMS, MaxGammaRMS: Double; const Epoch: Integer; out Reason: string);
const
  FlatImprovement = 0.00025;
  ClearWorsening = -0.00100;
  SharpWorseningFraction = 0.02;

  UpdateRatioWarning = 1.75;
  MomentWarning = 1.75;
  ParamGrowthWarning = 1.10;
  GammaGrowthWarning = 1.10;

  CooldownEpochs = 5;
var
  NewLR: Double;
  LossChange, LossWorseFraction, UpdateRatioChange, MRMSChange, VRMSChange, ParamChange, GammaChange: Double;
  SafetyWarnings: Integer;
  CanChangeLR: Boolean;
begin
  Reason := 'Learning rate unchanged.';

  if not LRState.Initialized then begin
    LRState.PrevLoss := CurrentLoss;
    LRState.PrevParamRMS := ParamRMS;
    LRState.PrevUpdateRatio := UpdateRatio;
    LRState.PrevMRMS := MRMS;
    LRState.PrevSqrtVRMS := SqrtVRMS;
    LRState.PrevMaxGammaRMS := MaxGammaRMS;
    LRState.ConsecutiveWorse := 0;
    LRState.ConsecutiveFlat := 0;
    LRState.Initialized := True;

    Reason := 'Initial adaptive-LR observation; learning rate unchanged.';
    Exit;
  end;

  LossChange := CurrentLoss - LRState.PrevLoss;

  if BestLoss > 0.0 then
    LossWorseFraction := (CurrentLoss - BestLoss) / BestLoss
  else
    LossWorseFraction := 0.0;

  if LRState.PrevUpdateRatio > 0.0 then
    UpdateRatioChange := UpdateRatio / LRState.PrevUpdateRatio
  else
    UpdateRatioChange := 1.0;

  if LRState.PrevMRMS > 0.0 then
    MRMSChange := MRMS / LRState.PrevMRMS
  else
    MRMSChange := 1.0;

  if LRState.PrevSqrtVRMS > 0.0 then
    VRMSChange := SqrtVRMS / LRState.PrevSqrtVRMS
  else
    VRMSChange := 1.0;

  if LRState.PrevParamRMS > 0.0 then
    ParamChange := ParamRMS / LRState.PrevParamRMS
  else
    ParamChange := 1.0;

  if LRState.PrevMaxGammaRMS > 0.0 then
    GammaChange := MaxGammaRMS / LRState.PrevMaxGammaRMS
  else
    GammaChange := 1.0;

  if LossChange > 0.0 then
    Inc(LRState.ConsecutiveWorse)
  else
    LRState.ConsecutiveWorse := 0;

  if Abs(RollingImprovement) < FlatImprovement then
    Inc(LRState.ConsecutiveFlat)
  else
    LRState.ConsecutiveFlat := 0;

  SafetyWarnings := 0;

  if UpdateRatioChange > UpdateRatioWarning then Inc(SafetyWarnings);
  if MRMSChange > MomentWarning then Inc(SafetyWarnings);
  if VRMSChange > MomentWarning then Inc(SafetyWarnings);
  if ParamChange > ParamGrowthWarning then Inc(SafetyWarnings);
  if GammaChange > GammaGrowthWarning then Inc(SafetyWarnings);

  CanChangeLR := (Epoch - LRState.LastLRChangeEpoch) >= CooldownEpochs;
  NewLR := LearningRate;

  if CanChangeLR then begin

    if (LossWorseFraction > SharpWorseningFraction) and
       (SafetyWarnings >= 2) then begin
      NewLR := LearningRate * 0.25;
      Reason := 'LR reduced 75%: loss is above best and model statistics are unstable.';
    end

    else if (LRState.ConsecutiveWorse >= 3) and
            (SafetyWarnings >= 1) then begin
      NewLR := LearningRate * 0.50;
      Reason := 'LR reduced 50%: loss worsened for 3 epochs with statistical warning.';
    end

    else if (LRState.ConsecutiveWorse >= 5) or
            (RollingImprovement < ClearWorsening) then begin
      NewLR := LearningRate * 0.50;
      Reason := 'LR reduced 50%: persistent loss deterioration.';
    end

    else if LRState.ConsecutiveFlat >= 5 then begin
      NewLR := LearningRate * 0.75;
      Reason := 'LR reduced 25%: training has plateaued.';
    end

    else if SafetyWarnings >= 3 then begin
      NewLR := LearningRate * 0.75;
      Reason := 'LR reduced 25%: several model statistics are rising.';
    end

    else if RollingImprovement > 0.0 then
      Reason := 'Loss is improving; learning rate unchanged.'

    else
      Reason := 'No sustained reason to reduce learning rate.';
  end
  else
    Reason := 'Adaptive-LR cooldown active; learning rate unchanged.';

  if NewLR < FloorLR then
    NewLR := FloorLR;

  if NewLR < LearningRate then begin
    LearningRate := NewLR;
    LRState.LastLRChangeEpoch := Epoch;

    // Start fresh after an actual LR change.
    LRState.ConsecutiveWorse := 0;
    LRState.ConsecutiveFlat := 0;
  end;

  LRState.PrevLoss := CurrentLoss;
  LRState.PrevParamRMS := ParamRMS;
  LRState.PrevUpdateRatio := UpdateRatio;
  LRState.PrevMRMS := MRMS;
  LRState.PrevSqrtVRMS := SqrtVRMS;
  LRState.PrevMaxGammaRMS := MaxGammaRMS;
end;

{ Kernel wrappers }
// Call the strided softmax kernel for a tightly packed matrix.
procedure LaunchSoftmaxForward(dIn: PSingle; dOut: PSingle; Rows: Integer; N: Integer; Temperature: Single);
begin
  LaunchSoftmaxForwardStrided(dIn, dOut, Rows, N, N, Temperature);
end;

end.
