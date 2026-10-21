unit Display;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.3, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
{ WesChat units }
Global;

{ Pausing and keyboard control }
procedure HardPauseNNL;
procedure HardPause;
procedure Pause;
procedure PauseNNL;
function CheckForControlKey: Char;

{ Symbol, activation, and normalization display }
function CleanUpSymbol(const x: RawByteString): RawByteString;
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
function ConsoleText(const S: UnicodeString): UnicodeString;
function TokenizerKindName(const Kind: TTokenizerKind): string;
function ActivationKindName(const Kind: TActivationKind): string;
function EffectiveMLPActivationName: string;
function NormKindName(const Kind: TNormKind): string;
function MLPGatingKindName(const Kind: TMLPGatingKind): string;

{ Vector and matrix display }
procedure DisplayVector(const V: TIVector);
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TSeqMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: THiddenMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TScoresMatrix; const Part: TPart = B); overload;

{ Token display and decoding }
procedure TC100(const TC: TIVector);
procedure TCFull(const TC: TIVector);
function Decode(const x: Integer): UnicodeString;
function DecodeToken(const x: Integer; const Kind: TTokenizerKind): UnicodeString;
procedure WesDetokenizeTextToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);

{ Program reports }
procedure ReportStartupKeyVariables;
procedure ReportKeyVariables;
procedure ReportProgramInfo;
procedure ReportTrainableParameters;
procedure FullReportTrainableParameters;
function NumberTrainableParameters: Int64;

implementation

uses
  Crt,
  DateUtils,
  Math,
  SysUtils;

{ Pausing and keyboard control }
// Pause, unconditional, no new line.
procedure HardPauseNNL;
begin
  Write('Hit <CR> to continue.... ');
  Readln;
end;

// Pause, unconditional, new line.
procedure HardPause;
begin
  HardPauseNNL;
  Writeln;
end;

// Pause, subject to DoNotPause.
procedure Pause;
var
  tt: TDateTime;
begin
  if not DoNotPause then begin
    tt := Now;
    HardPause;
    Stoptime := StopTime + Now - tt;
  end;
end;

// Pause, subject to DoNotPause, no new line.
procedure PauseNNL;
var
  tt: TDateTime;
begin
  if not DoNotPause then begin
    tt := Now;
    HardPauseNNL;
    Stoptime := StopTime + Now - tt;
  end;
end;

// Return the pending control key, or #0 if no key is waiting.
function CheckForControlKey: Char;
begin
  if KeyPressed then
    Result := ReadKey
  else
    Result := #0;  // No key.
end;

{ Program reports }
// Parameter counts are calculated when needed, not kept as mutable globals.
type
  TParameterCounts = record
    Embeddings, Attention, FFN, Norm, PerBlock, AllBlocks, Total: Int64;
  end;

// Compute the number of logically trainable parameters (tied embeddings counted once).
// All three MLP gating schemes reuse the same physical W1 and W2 matrices.
function CalculateParameterCounts: TParameterCounts;
begin
  FillChar(Result, SizeOf(Result), 0);

  Result.Embeddings := Int64(nVocab) * ModelDim;
  Result.Attention := Int64(4) * ModelDim * ModelDim;  // Wq, Wk, Wv, W0.
  Result.FFN := Int64(2) * ModelDim * ModelDimProj + ModelDimProj + ModelDim;

  case NormKind of
    LayerNorm: Result.Norm := Int64(4) * ModelDim;
    RMSNorm:   Result.Norm := Int64(2) * ModelDim;
  end;

  Result.PerBlock := Result.Attention + Result.FFN + Result.Norm;
  Result.AllBlocks := Int64(nBlock) * Result.PerBlock;
  Result.Total := Result.Embeddings + Result.AllBlocks;
end;

// Return the number of trainable parameters without producing console output.
function NumberTrainableParameters: Int64;
var
  P: TParameterCounts;
begin
  P := CalculateParameterCounts;
  Result := P.Total;
end;

// Compact trainable-parameter breakdown.
procedure ReportTrainableParameters;
var
  P: TParameterCounts;
begin
  P := CalculateParameterCounts;

  Writeln('--- Trainable Parameters ---');
  Writeln('Embeddings = ', P.Embeddings, '; Attention/block = ', P.Attention, '; FFN/block = ', P.FFN);
  Writeln('Norm/block = ', P.Norm, '; Total/block = ', P.PerBlock, '; All blocks = ', P.AllBlocks);
  Writeln('Total trainable parameters = ', P.Total);
end;

// Full parameter calculation, including the underlying formulas.
procedure FullReportTrainableParameters;
var
  P: TParameterCounts;
begin
  P := CalculateParameterCounts;

  Writeln('--- Detailed Trainable Parameter Calculation ---');
  Writeln('Architecture: D = ', ModelDim, '; Hidden = ', ModelDimProj, '; Blocks = ', nBlock,
    '; nVocab = ', nVocab, '; Gating = ', MLPGatingKindName(MLPGatingKind));

  Writeln('Embeddings (tied input/output):');
  Writeln('  ', nVocab, ' * ', ModelDim, ' = ', P.Embeddings);

  Writeln('Attention per block (Wq, Wk, Wv, W0):');
  Writeln('  4 * ', ModelDim, ' * ', ModelDim, ' = ', P.Attention);

  Writeln('MLP per block (W1, W2, b1, b2):');
  Writeln('  2 * ', ModelDim, ' * ', ModelDimProj, ' + ', ModelDimProj, ' + ', ModelDim,
    ' = ', P.FFN);

  case NormKind of
    LayerNorm: begin
      Writeln('LayerNorm per block (Gamma1, Beta1, Gamma2, Beta2):');
      Writeln('  4 * ', ModelDim, ' = ', P.Norm);
    end;
    RMSNorm: begin
      Writeln('RMSNorm per block (Gamma1, Gamma2):');
      Writeln('  2 * ', ModelDim, ' = ', P.Norm);
    end;
  end;

  Writeln('Per block: ', P.Attention, ' + ', P.FFN, ' + ', P.Norm, ' = ', P.PerBlock);
  Writeln('All blocks: ', nBlock, ' * ', P.PerBlock, ' = ', P.AllBlocks);
  Writeln('Total: ', P.Embeddings, ' + ', P.AllBlocks, ' = ', P.Total);
end;

// Common report components: use the same labels and ordering in every report.
procedure ReportModelSummary;
begin
  Writeln('Architecture: D=', ModelDim, '; Hidden=', ModelDimProj, '; Proj=', Proj, '; Heads=', nHead, '; Blocks=', nBlock, '; SeqLen=', SeqLen);
  Writeln('Vocabulary: nVocab=', nVocab, '; DimVocab=', DimVocab, '; MaxSymbols=', MaxSymbols);
end;

procedure ReportModelOptions;
begin
  Writeln('Model options: Tokenizer=', TokenizerKindName(TokenizerKind), '; Gating=', MLPGatingKindName(MLPGatingKind),
    '; Activation=', EffectiveMLPActivationName, '; Norm=', NormKindName(NormKind), '; Alpha=', ActivationAlpha:0:4);
end;

procedure ReportRunSettings;
begin
  Writeln('Run: Training=', Training, '; Shuffle=', ShuffleWindows, '; Stride=', Stride, '; AdaptiveLR=', AdaptiveLR,
    'Dropout rates (Attention/MLP/Residual) = ', ADropOut:0:3, '/', MLPDropOut:0:3, '/', RDropOut:0:3);
end;

procedure ReportTrainingProgress;
begin
  Writeln('Progress: Epoch=', CompletedEpochs, '; Step=', GlobalStep, '; AdamWStep=', AdamWStep,
    '; LR=', LearningRate:0:8, '; MaxEpochs=', MaxEpochs, '; StartStride=', StartStride);
end;

procedure ReportOptimizerSettings;
begin
  Write('Learning rate: Base=', BaseLearningRate:0:8, '; Floor=', FloorLearningRate:0:8, '; RollOff=', RollOff:0:8);

  if OverrideLearningRate <> -1.0 then
    Writeln('; LR override = ', OverrideLearningRate:0:8)
  else
    Writeln('; LR override = none');

  Writeln('AdamW: Beta1=', AdamBeta1:0:6, '; Beta2=', AdamBeta2:0:6, '; Epsilon=', AdamEpsilon:0:10, '; WeightDecay=', WeightDecay:0:7);
  Writeln('Other: ClipLimit=', ClipLimit:0:4, '; GlobalSeed=', GlobalSeed, '; Training temperature=', TTemperature:0:4, '; Inference temperature=', ITemperature:0:4);
end;

procedure ReportBestLoss;
begin
  Writeln('--- Best Loss ---');

  if (MinLoss < MaxDouble) and (MinLoss <> 1000000) then
    Writeln('Minimum = ', MinLoss:0:7, '; Epoch = ', MinLossEpoch)
  else
    Writeln('Minimum = none');

  if BestSavedLoss < MaxDouble then
    Writeln('Saved best = ', BestSavedLoss:0:7, '; Epoch = ', LastBestSaveEpoch)
  else
    Writeln('Saved best = none');
end;

procedure ReportPath(const PathLabel, PathValue: string);
begin
  if Trim(PathValue) = '' then
    Writeln(PathLabel, ': (none)')
  else
    Writeln(PathLabel, ': ', PathValue);
end;

// Startup: configuration only, before a training run.
procedure ReportStartupKeyVariables;
begin
  Writeln('--- Startup Configuration ---');
  ReportModelSummary;
  ReportModelOptions;
  ReportRunSettings;
end;

// Short status report while training or diagnosing a running model.
procedure ReportKeyVariables;
begin
  Writeln('--- Current Status ---');
  ReportModelSummary;
  ReportModelOptions;
  ReportTrainingProgress;
  ReportRunSettings;
end;

// Full inventory of the program, corpus, model and training state.
procedure ReportProgramInfo;
begin
  Writeln('--- Program Information ---');
  Writeln('WesChat ', Version, '; begun January 10, 2026; Author: Wesley R. Parsons');

  Writeln('--- Paths ---');
  ReportPath('Work root', WorkRoot);
  ReportPath('Working directory', WorkingDir);
  ReportPath('Best model', BestModelFileName);

  Writeln('--- Corpus / Tokenizer ---');
  Writeln('Tokenizer = ', TokenizerKindName(TokenizerKind), '; Corpus ID = ', CorpusID);
  Writeln('Corpus bytes = ', nCorpus, '; Raw tokens = ', RawTokenCount,
    '; Stored tokens = ', nTokenizedCorpus, '; Symbols = ', nSymbols);

  Writeln('--- Model ---');
  ReportModelSummary;
  ReportModelOptions;

  Writeln('--- Training ---');
  ReportTrainingProgress;
  ReportRunSettings;
  ReportOptimizerSettings;

  ReportBestLoss;
  ReportTrainableParameters;
end;

// Replace unprintable symbols with space.
function CleanUpSymbol(const x: RawByteString): RawByteString;
var
  j, L: Integer;
  ch: Char;
begin
  L := Length(x);
  SetLength(Result, L);   // Allocate output string.

  for j := 1 to L do begin
    ch := x[j];

    if Ord(ch) in [0..31, 127..255] then
      Result[j] := ' '
    else
      Result[j] := ch;
  end;
end;

// Display the symbol table.
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
var
  i: Integer;
begin
  Writeln('--- Symbol Table ---');
  for i := 0 to High(SymbolTable) do begin  // Loop thru each symbol in table.
    if (i in [0..31]) or (i in [127..255]) then
      Write(i: 8, IntToHex(i, 2): 15)       // Hex for non-ASCII characters.
    else
      Write(i: 8, '"' + SymbolTable[i] + '"': 15);
    if (i mod 5) = 4 then Writeln;
    if (i > 0) and (i mod 100 = 99) then Pause;
  end;
  Writeln('Symbol table length = ', Length(SymbolTable));
end;

// Display for inference CR/LF.
function ConsoleText(const S: UnicodeString): UnicodeString;
begin
  Result := UnicodeString(StringReplace(String(S), #13#10, #10, [rfReplaceAll]));
  Result := UnicodeString(StringReplace(String(Result), #13, #10, [rfReplaceAll]));
  Result := UnicodeString(StringReplace(String(Result), #10, #13#10, [rfReplaceAll]));
end;

// Display the tokenizer kind.
function TokenizerKindName(const Kind: TTokenizerKind): string;
begin
  case Kind of
    WesTokenizer:  Result := 'WesTokenizer';
    UDTokenizer:   Result := 'UDTokenizer';
    GPT2Tokenizer: Result := 'GPT2Tokenizer';
  else
    Result := 'Unknown';
  end;
end;

// Display the gating kind.
function MLPGatingKindName(const Kind: TMLPGatingKind): string;
begin
  case Kind of
    NoGating:  Result := 'None';
    SelfGating:   Result := 'Self';
    SplitSwiGLU: Result := 'SplitSwiGLU';
  else
    Result := 'Unknown';
  end;
end;

// Display the activation kind.
function ActivationKindName(const Kind: TActivationKind): string;
begin
  case Kind of
    ReLU: Result := 'ReLU';
    GELU: Result := 'GELU';
    SiLU: Result := 'SiLU';
    LeakyReLU: Result := 'LeakyReLU';
    ELU: Result := 'ELU';
    SoftPlus: Result := 'SoftPlus';
    Mish: Result := 'Mish';
  else
    Result := 'Unknown activation function';
  end;
end;

// Display SilU as activation when using SplitSwiGLU or Self-Gating.
function EffectiveMLPActivationName: string;
begin
  case MLPGatingKind of
    NoGating:    Result := ActivationKindName(ActivationKind);
    SelfGating:  Result := 'SiLU (self-gated)';
    SplitSwiGLU: Result := 'SiLU (gate only)';
  else
    Result := 'Unknown';
  end;
end;

// Display the norm kind.
function NormKindName(const Kind: TNormKind): string;
begin
  case Kind of
    LayerNorm: Result := 'LayerNorm';
    RMSNorm: Result := 'RMSNorm';
  else
    Result := 'Unknown norm function';
  end;
end;

{ Vector and matrix display }
// Display an integer vector and pause.
procedure DisplayVector(const V: TIVector);
var
  i: Integer;
begin
  for i := 0 to High(V) do   // V is an array of integers.
    Write(V[i], ' ');
  Writeln;
  Pause;
end;

// Display scope of display below.
procedure PartScope(const Part: TPart);
begin
  Case Part of
    B: Writeln(' Beginning.');
    E: Writeln(' End.');
    F: Writeln(' Full.');
    G: Writeln(' Sample.');
  end;
end;

// Display an X matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride, hStride: Integer;
begin
  vStride := 1;
  hStride := 1;
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TSeqMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Hidden matrix, B, E, F, or G.
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride, hStride: Integer;
begin
  vStride := 1;
  hStride := 1;
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: THiddenMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Embeddings matrix, B, E, F, or G.
procedure DisplayX(const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE, ActiveVocab: Integer;
  vStride, hStride: Integer;
begin
  vStride := 1;
  hStride := 1;
  ActiveVocab := Min(nVocab, Length(X));
  if ActiveVocab <= 0 then begin
    Writeln('(no active vocabulary rows)');
    Exit;
  end;

  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := Max(0, ActiveVocab - 10);
      iE := ActiveVocab - 1;
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := ActiveVocab - 1;
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(ActiveVocab / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(ActiveVocab - 1: 4);
    for j := jB to jE do
      Write(X[ActiveVocab - 1, j * hStride]: 11: 5, ' ');
    Write(X[ActiveVocab - 1, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Vocab TSeq matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE, ActiveVocab: Integer;
  vStride, hStride: Integer;
begin
  vStride := 1;
  hStride := 1;
  ActiveVocab := Min(nVocab, Length(X[0]));
  if ActiveVocab <= 0 then begin
    Writeln('(no active vocabulary columns)');
    Exit;
  end;

  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := Max(0, ActiveVocab - 10);
      jE := ActiveVocab - 1;
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := ActiveVocab - 1;
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(ActiveVocab / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(ActiveVocab - 1: 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      Write(X[i * vStride, ActiveVocab - 1]: 11: 7, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 7, ' ');
    Write(X[High(X), ActiveVocab - 1]: 11: 7, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TSeqVocabMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a ScoresHead matrix, B, E, F, or G.
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride, hStride: Integer;
begin
  vStride := 1;
  hStride := 1;
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 7, ' ');
    Write(X[High(X), High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TScoresMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

{ Token display and decoding }
// Display up to the first 100 token IDs and their decoded text.
procedure TC100(const TC: TIVector);
var
  i: Integer;
begin
  Write('Tokenized Corpus (length up to 100): ');
  for i := 0 to Min(99, High(TC)) do
    Write(TC[i], ' ');
  Writeln;

  Write('Detokenized Corpus (length up to 100): ');
  for i := 0 to Min(99, High(TC)) do
    Write(Decode(TC[i]));
  Writeln;
end;

// Display all token IDs and their decoded text.
procedure TCFull(const TC: TIVector);
var
  i: Integer;
begin
  Write('Tokenized Corpus (full length ', Length(TC), '): ');
  for i := 0 to High(TC) do
    Write(TC[i], ' ');
  Writeln;

  Write('Detokenized Corpus (full length ', Length(TC), '): ');
  for i := 0 to High(TC) do
    Write(Decode(TC[i]));
  Writeln;
  Pause;
end;

// Decode one token using the currently selected tokenizer.
function Decode(const x: Integer): UnicodeString;
begin
  Result := DecodeToken(x, TokenizerKind);
end;

// Decode one token using an explicitly supplied tokenizer kind.
function DecodeToken(const x: Integer; const Kind: TTokenizerKind): UnicodeString;
begin
  case Kind of
    WesTokenizer, UDTokenizer: begin
      if (x >= 0) and (x < Length(SymbolTable)) then
        Result := UTF8Decode(SymbolTable[x])
      else
        Result := 'BAD TOKEN';
    end;

    GPT2Tokenizer: begin
      if x = GPT2BOS then
        Result := '<BOS>'
      else if x = GPT2EOS then
        Result := '<EOS>'
      else if x = GPT2PAD then
        Result := '<PAD>'
      else if x = GPT2UNK then
        Result := '<UNK>'
      else if Assigned(Vocab) and (x >= 0) and (x < Vocab.Count) then
        Result := UTF8Decode(Vocab[x])
      else
        Result := 'BAD TOKEN';
    end;
  else
    Result := 'BAD TOKEN';
  end;
end;

// Detokenize a Wes/UD token list to text for diagnostics.
procedure WesDetokenizeTextToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
var
  i, j, iB, iE, Cutoff: Integer;
  S: string;
begin
  if Length(TokenizedCorpus) = 0 then begin
    Writeln('Detokenized Corpus is empty.');
    Exit;
  end;

  if Length(TokenizedCorpus) < 500 then
    Cutoff := Length(TokenizedCorpus) - 1
  else
    Cutoff := 499;

  Case Part of
    B: Write('Detokenized Corpus, First 500 tokens: ');
    E: Write('Detokenized Corpus, Last 500 tokens: ');
    F: Write('Detokenized Corpus, all tokens: ');
  end;

  Case Part of
    B: begin
      iB := 0;
      iE := Cutoff;
    end;
    E: begin
      iB := High(TokenizedCorpus) - Cutoff;
      iE := High(TokenizedCorpus);
    end;
    F: begin
      iB := 0;
      iE := High(TokenizedCorpus);
    end;
  end;

  for i := iB to iE do begin
    S := SymbolTable[TokenizedCorpus[i]];
    if TokenizedCorpus[i] = EOS then begin
      Write(#13, #10);
      Continue;
    end;

    for j := 1 to Length(S) do begin
      if S[j] = #10 then begin
        if (j = 1) or (S[j - 1] <> #13) then
          Write(#13);
        Write(#10);
      end
      else
        Write(S[j]);
    end;
  end;

  Writeln;
end;

end.
