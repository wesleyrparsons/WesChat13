unit IOHandler;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  Math, SysUtils,
  { WesChat units }
  Display, Global, Util, WesTokenize;

{ File input and output }
function ComputeCorpusID(const OneCorpus: TBVector): QWord;
procedure ReadFileBytes(const FileName: string; var OneCorpus: TBVector);
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
procedure SaveSymbolTableIfMissing(const BaseName: string);
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: string);
procedure RestoreTrainingCheckpoint(const C: TTrainingCheckpoint);
function SaveModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
function LoadModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;

{ TinyStories conversion }
procedure ConvertTSEndOfText(const FileName: string);
procedure ConvertTSSeparators(const FileName: string);

implementation

{ WES3 file formats }
const
  SymbolMagic3: array[0..7] of Char = ('W','E','S','3','S','Y','M','T');
  TokenMagic3: array[0..7] of Char = ('W','E','S','3','T','O','K','L');
  ModelMagic3: array[0..7] of Char = ('W','E','S','3','M','O','D','L');
  WES3ModelHeaderSize = 1024;
  WES3FileInfoSize = 1024;
  WES3FileInfoVersion = 1;
  WES3ModelHeaderVersion = 1;
  WES3CheckpointVersion = 1;
  WES3OptimizerVersion = 1;
  WES3FlagHasMLPGating = 1; // Header.Flags bit 0: Reserved[0] stores TMLPGatingKind.

type
  TProgramVersionChars = array[0..15] of Char;
  TWES3FileInfo = packed record
    HeaderVersion: UInt32;
    HeaderSize: UInt32;
    TokenizerKind: UInt32;
    Flags: UInt32;
    CorpusBytes: Int64;
    ProcessedBytes: Int64;
    RawTokenCount: Int64;
    TokenCount: Int64;
    PaddingCount: Int64;
    SymbolCount: UInt32;
    MergeCount: UInt32;
    CorpusID: UInt64;
    ProgramVersion: TProgramVersionChars;
    Reserved: array[0..935] of Byte;
  end;
  {$if SizeOf(TWES3FileInfo) <> WES3FileInfoSize}
    {$fatal TWES3FileInfo must be exactly 1024 bytes}
  {$endif}
  TWES3ModelHeader = packed record
    HeaderVersion: UInt32;
    HeaderSize: UInt32;
    CheckpointVersion: UInt32;
    OptimizerVersion: UInt32;
    TokenizerKind: UInt32;
    Flags: UInt32;
    CorpusID: QWord;
    CorpusBytes: Int64;
    RawTokenCount: Int64;
    TokenCount: Int64;
    PaddingCount: Int64;
    SymbolCount: UInt32;
    MergeCount: UInt32;
    ModelDim: UInt32;
    ModelDimProj: UInt32;
    Proj: UInt32;
    NVocab: UInt32;
    DimVocab: UInt32;
    NBlock: UInt32;
    NHead: UInt32;
    SeqLen: UInt32;
    ProgramVersion: array[0..15] of Char;
    ModelOptionsVersion: UInt32;      // Model options.
    NormKind: UInt32;
    ActivationKind: UInt32;
    ActivationAlpha: Single;
    Reserved: array[0..887] of Byte; // Reserved[0] = TMLPGatingKind when Flags bit 0 is set.
  end;
  {$if SizeOf(TWES3ModelHeader) <> WES3ModelHeaderSize}
    {$fatal TWES3ModelHeader must be exactly 1024 bytes}
  {$endif}

{ File metadata helpers }
// Copy the running program version into a zero-padded fixed-length header field.
procedure StoreProgramVersion(var Dest: TProgramVersionChars);
var
  N: Integer;
begin
  FillChar(Dest, SizeOf(Dest), 0);
  N := Length(Version);
  if N > SizeOf(Dest) then N := SizeOf(Dest);
  if N > 0 then
    Move(Version[1], Dest[0], N);
end;

{ Tokenizer metadata }
// Set special-token IDs from a tokenizer kind when the file format does not store the IDs separately.
procedure SetSpecialTokenIDsForKind(const Kind: TTokenizerKind);
begin
  case Kind of
    WesTokenizer, UDTokenizer: begin
      BOS := TokBOS;
      EOS := TokEOS;
      PAD := TokPAD;
      UNK := TokNUL;
    end;

    GPT2Tokenizer: begin
      BOS := GPT2BOS;
      EOS := GPT2EOS;
      PAD := GPT2PAD;
      UNK := GPT2UNK;
    end;
  end;
end;

{ Corpus files }
// Compute a stable FNV-1a identifier for a complete corpus byte vector.
function ComputeCorpusID(const OneCorpus: TBVector): QWord;
const
  FNVOffsetBasis: QWord = 14695981039346656037;
  FNVPrime: QWord = 1099511628211;
var
  i: Integer;
begin
  Result := FNVOffsetBasis;

  for i := 0 to High(OneCorpus) do begin
    Result := Result xor QWord(OneCorpus[i]);
    Result := Result * FNVPrime;
  end;
end;

// Read a corpus file as raw bytes.
procedure ReadFileBytes(const FileName: string; var OneCorpus: TBVector);
var
  F: File;
  Size, i: Integer;
  B: Byte;
begin
  AssignFile(F, FileName);
  Reset(F, 1);     // Open in binary mode.
  Size := FileSize(F);
  SetLength(OneCorpus, Size);

  // Write the Corpus as it is read.
  if VeryVerboseTokenize then
    Writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VeryVerboseTokenize then
      if DisplayEachByteRead then
        if B < 32 then
          Write('<', B, '>')
        else
          Write(Chr(B));
  end;
  CloseFile(F);

  if VeryVerboseTokenize then begin
    Writeln('ReadByteFile: ');
    for i := 0 to Min(150, High(OneCorpus)) do
      Write(OneCorpus[i], ' ');
    Readln;
  end;
  if VeryVerboseTokenize then
    Writeln;

  // Report corpus length.
  Writeln('Read ', Size, ' bytes from ', FileName, '.');
end;

{ Symbol tables }
// Load a WES3 symbol table. Earlier file formats are no longer supported.
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  FileMagic: array[0..7] of Char;
  FileInfo: TWES3FileInfo;
  S: string;
  i, Len: Integer;
begin
  BOS := 256;
  EOS := 257;
  PAD := 258;
  UNK := 259;

  AssignFile(F, FileName);
  Reset(F, 1);

  if FileSize(F) < SizeOf(FileMagic) + SizeOf(FileInfo) + 4 * SizeOf(Integer) then begin
    CloseFile(F);
    Writeln('Invalid or incomplete WES3 symbol table: ', FileName);
    Pause;
    Exit;
  end;

  BlockRead(F, FileMagic, SizeOf(FileMagic));
  if not CompareMem(@FileMagic[0], @SymbolMagic3[0], SizeOf(FileMagic)) then begin
    CloseFile(F);
    Writeln('WES3SYMT header not found: ', FileName);
    Pause;
    Exit;
  end;

  BlockRead(F, FileInfo, SizeOf(FileInfo));
  if FileInfo.HeaderSize <> WES3FileInfoSize then begin
    CloseFile(F);
    Writeln('Invalid WES3 symbol-table header size: ', FileInfo.HeaderSize, '.');
    Pause;
    Exit;
  end;
  if FileInfo.HeaderVersion <> WES3FileInfoVersion then begin
    CloseFile(F);
    Writeln('Unsupported WES3 symbol-table version: ', FileInfo.HeaderVersion, '.');
    Pause;
    Exit;
  end;
  if FileInfo.TokenizerKind > Ord(High(TTokenizerKind)) then begin
    CloseFile(F);
    Writeln('Invalid tokenizer kind in WES3 symbol table: ', FileInfo.TokenizerKind, '.');
    Pause;
    Exit;
  end;

  // The symbol file must not replace counts loaded from the token list.
  TokenizerKind := TTokenizerKind(FileInfo.TokenizerKind);
  SetSpecialTokenIDsForKind(TokenizerKind);
  nCorpus := FileInfo.CorpusBytes;
  nSymbols := FileInfo.SymbolCount;
  nMerges := FileInfo.MergeCount;
  MergeCount := nMerges;
  if FileInfo.CorpusID <> 0 then
    CorpusID := FileInfo.CorpusID;

  SetLength(SymbolTable, nSymbols);

  // Special token IDs are stored immediately after the fixed WES3 header.
  BlockRead(F, BOS, SizeOf(BOS));
  BlockRead(F, EOS, SizeOf(EOS));
  BlockRead(F, PAD, SizeOf(PAD));
  BlockRead(F, UNK, SizeOf(UNK));

  for i := 0 to nSymbols - 1 do begin
    BlockRead(F, Len, SizeOf(Len));
    if Len < 0 then begin
      CloseFile(F);
      Writeln('Invalid WES3 symbol length at index ', i, '.');
      Pause;
      Exit;
    end;
    SetLength(S, Len);
    if Len > 0 then
      BlockRead(F, S[1], Len);
    SymbolTable[i] := S;
  end;

  CloseFile(F);
  nSymbols := Length(SymbolTable);
  Writeln('Loaded WES3 symbol table with ', nSymbols, ' symbols from ', FileName, '.');
end;


// Save a symbol table in WES3 format.
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
var
  F: file;
  FileInfo: TWES3FileInfo;
  NumSymbols, NumMerges: Integer;
  i, Len: Integer;
begin
  FillChar(FileInfo, SizeOf(FileInfo), 0);

  NumSymbols := Length(SymbolTable);

  case TokenizerKind of
    WesTokenizer: NumMerges := Max(0, NumSymbols - (TokNUL + 1));
    UDTokenizer: NumMerges := Max(0, NumSymbols - UDTagBoundary);
    GPT2Tokenizer: NumMerges := 0;
  else
    NumMerges := 0;
  end;

  FileInfo.HeaderVersion := WES3FileInfoVersion;
  FileInfo.HeaderSize := WES3FileInfoSize;

  FileInfo.TokenizerKind := Ord(TokenizerKind);
  FileInfo.Flags := 0;

  FileInfo.CorpusBytes := nCorpus;
  FileInfo.ProcessedBytes := nCorpus;

  FileInfo.RawTokenCount := RawTokenCount;
  FileInfo.TokenCount := nTokenizedCorpus;
  if nTokenizedCorpus >= RawTokenCount then
    FileInfo.PaddingCount := nTokenizedCorpus - RawTokenCount
  else
    FileInfo.PaddingCount := 0;

  FileInfo.SymbolCount := NumSymbols;
  FileInfo.MergeCount := NumMerges;
  FileInfo.CorpusID := CorpusID;

  StoreProgramVersion(FileInfo.ProgramVersion);

  Assign(F, SymbolFileName);
  Rewrite(F, 1);

  // WES3SYMT magic.
  BlockWrite(F, SymbolMagic3, SizeOf(SymbolMagic3));

  // Fixed 1024-byte WES3 metadata header.
  BlockWrite(F, FileInfo, SizeOf(FileInfo));

  // Special token IDs.
  BlockWrite(F, BOS, SizeOf(BOS));
  BlockWrite(F, EOS, SizeOf(EOS));
  BlockWrite(F, PAD, SizeOf(PAD));
  BlockWrite(F, UNK, SizeOf(UNK));

  // Write each symbol.
  for i := 0 to NumSymbols - 1 do begin
    Len := Length(SymbolTable[i]);
    BlockWrite(F, Len, SizeOf(Len));
    if Len > 0 then
      BlockWrite(F, SymbolTable[i][1], Len);
  end;

  Close(F);
  Writeln('File ', SymbolFileName, ' successfully saved.');
end;


// Save the current symbol table in SymbolDir if the target file does not already exist.
procedure SaveSymbolTableIfMissing(const BaseName: string);
var
  FileName: string;
begin
  FileName := SymbolDir + ChangeFileExt(CleanBaseName(BaseName), '.sym');

  if FileExists(FileName) then begin
    Writeln('Symbol table already exists: ', FileName);
    Exit;
  end;

  SaveSymbolTable(FileName, SymbolTable);
  Writeln('Symbol table saved: ', FileName);
end;

{ Token lists }
// Load a WES3 token list. Earlier file formats are no longer supported.
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
var
  F: file;
  FileMagic: array[0..7] of Char;
  FileInfo: TWES3FileInfo;
  v, i: Integer;
  Count: Int64;
begin
  AssignFile(F, TokenFileName);
  Reset(F, 1);

  if FileSize(F) < SizeOf(FileMagic) + SizeOf(FileInfo) then begin
    CloseFile(F);
    Writeln('Invalid or incomplete WES3 token list: ', TokenFileName);
    Pause;
    Exit;
  end;

  BlockRead(F, FileMagic, SizeOf(FileMagic));
  if not CompareMem(@FileMagic[0], @TokenMagic3[0], SizeOf(FileMagic)) then begin
    CloseFile(F);
    Writeln('WES3TOKL header not found: ', TokenFileName);
    Pause;
    Exit;
  end;

  BlockRead(F, FileInfo, SizeOf(FileInfo));
  if FileInfo.HeaderSize <> WES3FileInfoSize then begin
    CloseFile(F);
    Writeln('Invalid WES3 token-file header size: ', FileInfo.HeaderSize, '.');
    Pause;
    Exit;
  end;
  if FileInfo.HeaderVersion <> WES3FileInfoVersion then begin
    CloseFile(F);
    Writeln('Unsupported WES3 token-file version: ', FileInfo.HeaderVersion, '.');
    Pause;
    Exit;
  end;
  if FileInfo.TokenizerKind > Ord(High(TTokenizerKind)) then begin
    CloseFile(F);
    Writeln('Invalid tokenizer kind in WES3 token file: ', FileInfo.TokenizerKind, '.');
    Pause;
    Exit;
  end;

  Count := FileInfo.TokenCount;
  if (Count < 0) or (Count > High(Integer)) or
     (Count > (FileSize(F) - FilePos(F)) div SizeOf(Integer)) then begin
    CloseFile(F);
    Writeln('Invalid token count or incomplete WES3 token list: ', TokenFileName);
    Pause;
    Exit;
  end;

  TokenizerKind := TTokenizerKind(FileInfo.TokenizerKind);
  SetSpecialTokenIDsForKind(TokenizerKind);
  nCorpus := FileInfo.CorpusBytes;
  RawTokenCount := FileInfo.RawTokenCount;
  nTokenizedCorpus := FileInfo.TokenCount;
  nSymbols := FileInfo.SymbolCount;
  nMerges := FileInfo.MergeCount;
  MergeCount := nMerges;
  CorpusID := FileInfo.CorpusID;

  SetLength(TokenizedCorpus, Count);
  for i := 0 to Count - 1 do begin
    BlockRead(F, v, SizeOf(v));
    TokenizedCorpus[i] := v;
  end;

  CloseFile(F);
  nTokenizedCorpus := Length(TokenizedCorpus);
  PaddedTokenCount := nTokenizedCorpus;
  Writeln('Loaded WES3 token list with ', Count, ' tokens from ', TokenFileName, '.');
end;


// Save a token list in WES3 format.
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: string);
var
  F: file;
  FileInfo: TWES3FileInfo;
  TokenCount, PaddingCount: Int64;
  i, v: Integer;
begin
  FillChar(FileInfo, SizeOf(FileInfo), 0);

  TokenCount := Length(TokenizedCorpus);
  if TokenCount >= RawTokenCount then
    PaddingCount := TokenCount - RawTokenCount
  else
    PaddingCount := 0;

  // WES3 metadata.
  FileInfo.HeaderVersion := WES3FileInfoVersion;
  FileInfo.HeaderSize := WES3FileInfoSize;

  FileInfo.TokenizerKind := Ord(TokenizerKind);
  FileInfo.Flags := 0;

  FileInfo.CorpusBytes := nCorpus;
  FileInfo.ProcessedBytes := nCorpus;

  FileInfo.RawTokenCount := RawTokenCount;
  FileInfo.TokenCount := TokenCount;
  FileInfo.PaddingCount := PaddingCount;

  FileInfo.SymbolCount := nSymbols;
  FileInfo.MergeCount := nMerges;

  FileInfo.CorpusID := CorpusID;

  StoreProgramVersion(FileInfo.ProgramVersion);

  Assign(F, TokenFileName);
  Rewrite(F, 1);

  // WES3TOKL magic.
  BlockWrite(F, TokenMagic3, SizeOf(TokenMagic3));

  // Fixed 1024-byte WES3 metadata header.
  BlockWrite(F, FileInfo, SizeOf(FileInfo));

  // Token data.
  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    BlockWrite(F, v, SizeOf(v));
  end;

  Close(F);

  Writeln('Saved ', TokenCount, ' tokens to ', TokenFileName, '.');
end;


{ Model checkpoints }
// Human-readable name for the saved MLP mode.
function WES3GatingName(const Kind: TMLPGatingKind): string;
begin
  case Kind of
    NoGating: Result := 'None';
    SelfGating: Result := 'Self';
    SplitSwiGLU: Result := 'SplitSwiGLU';
  else
    Result := 'Unknown';
  end;
end;

// Clear device pointers deserialized as part of the model record.
procedure ClearDevicePointers(var Model: TWModelParams);
var
  b: Integer;
begin
  Model.Embeddings.dValue := nil;
  Model.Embeddings.dGrad := nil;

  for b := 0 to nBlock - 1 do begin
    with Model.ParamBlock[b] do begin
      Wq.dValue := nil; Wq.dGrad := nil;
      Wk.dValue := nil; Wk.dGrad := nil;
      Wv.dValue := nil; Wv.dGrad := nil;
      W0.dValue := nil; W0.dGrad := nil;
      W1.dValue := nil; W1.dGrad := nil;
      W2.dValue := nil; W2.dGrad := nil;
      b1.dValue := nil; b1.dGrad := nil;
      b2.dValue := nil; b2.dGrad := nil;
      Gamma1.dValue := nil; Gamma1.dGrad := nil;
      Beta1.dValue := nil; Beta1.dGrad := nil;
      Gamma2.dValue := nil; Gamma2.dGrad := nil;
      Beta2.dValue := nil; Beta2.dGrad := nil;
    end;
  end;
end;

// Capture the global training state stored in a model checkpoint.
procedure CaptureTrainingCheckpoint(out C: TTrainingCheckpoint);
begin
  C.GlobalStep := GlobalStep;
  C.CompletedEpochs := CompletedEpochs;

  C.LearningStyle := LearningStyle;
  C.LearningRate := LearningRate;
  C.OverrideLearningRate := OverrideLearningRate;
  C.BaseLearningRate := BaseLearningRate;
  C.FloorLearningRate := FloorLearningRate;
  C.RollOff := RollOff;

  C.WeightDecay := WeightDecay;
  C.ClipLimit := ClipLimit;
  C.TTemperature := TTemperature;
  C.ITemperature := ITemperature;

  C.ADropOut := ADropOut;
  C.RDropOut := RDropOut;
  C.MLPDropOut := MLPDropOut;

  C.ShuffleWindows := ShuffleWindows;

  C.Stride := Stride;
  C.StartStride := StartStride;
  C.GlobalSeed := GlobalSeed;

  C.AdamWStep := AdamWStep;
  C.AdamBeta1 := AdamBeta1;
  C.AdamBeta2 := AdamBeta2;
  C.AdamEpsilon := AdamEpsilon;

  // Adaptive learning-rate state.
  C.AdaptiveLR := AdaptiveLR;
  C.AdaptiveLRState := AdaptiveLRState;

  // Historical loss/best-model state.
  C.MinLoss := MinLoss;
  C.MinLossEpoch := MinLossEpoch;
  C.BestSavedLoss := BestSavedLoss;
  C.LastBestSaveEpoch := LastBestSaveEpoch;
end;

// Restore global training state from a model checkpoint.
procedure RestoreTrainingCheckpoint(const C: TTrainingCheckpoint);
begin
  GlobalStep := C.GlobalStep;
  CompletedEpochs := C.CompletedEpochs;

  LearningStyle := C.LearningStyle;
  LearningRate := C.LearningRate;
  OverrideLearningRate := C.OverrideLearningRate;
  BaseLearningRate := C.BaseLearningRate;
  FloorLearningRate := C.FloorLearningRate;
  RollOff := C.RollOff;

  WeightDecay := C.WeightDecay;
  ClipLimit := C.ClipLimit;
  TTemperature := C.TTemperature;
  ITemperature := C.ITemperature;

  ADropOut := C.ADropOut;
  RDropOut := C.RDropOut;
  MLPDropOut := C.MLPDropOut;

  ShuffleWindows := C.ShuffleWindows;

  Stride := C.Stride;
  StartStride := C.StartStride;
  GlobalSeed := C.GlobalSeed;

  AdamWStep := C.AdamWStep;
  AdamBeta1 := C.AdamBeta1;
  AdamBeta2 := C.AdamBeta2;
  AdamEpsilon := C.AdamEpsilon;

  // Adaptive learning-rate state.
  AdaptiveLR := C.AdaptiveLR;
  AdaptiveLRState := C.AdaptiveLRState;

  // Historical loss/best-model state.
  MinLoss := C.MinLoss;
  MinLossEpoch := C.MinLossEpoch;
  BestSavedLoss := C.BestSavedLoss;
  LastBestSaveEpoch := C.LastBestSaveEpoch;

  // Derived value.
  DecayScale := 1.0 - LearningRate * WeightDecay;
end;


// Save AdamW first and second moments. Only host M and V arrays are written.
procedure SaveAdamWState(var F: file; const WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    BlockWrite(F, M, EmbeddingsSize);
    BlockWrite(F, V, EmbeddingsSize);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      BlockWrite(F, Wq.M, WeightSize);
      BlockWrite(F, Wq.V, WeightSize);

      BlockWrite(F, Wk.M, WeightSize);
      BlockWrite(F, Wk.V, WeightSize);

      BlockWrite(F, Wv.M, WeightSize);
      BlockWrite(F, Wv.V, WeightSize);

      BlockWrite(F, W0.M, WeightSize);
      BlockWrite(F, W0.V, WeightSize);

      // MLP weights.
      BlockWrite(F, W1.M, WeightProjectedSize);
      BlockWrite(F, W1.V, WeightProjectedSize);

      BlockWrite(F, W2.M, WeightProjectedSize);
      BlockWrite(F, W2.V, WeightProjectedSize);

      // Biases.
      BlockWrite(F, b1.M, ProjectedSize);
      BlockWrite(F, b1.V, ProjectedSize);

      BlockWrite(F, b2.M, ModelSize);
      BlockWrite(F, b2.V, ModelSize);

      // LayerNorm 1.
      BlockWrite(F, Gamma1.M, ModelSize);
      BlockWrite(F, Gamma1.V, ModelSize);

      BlockWrite(F, Beta1.M, ModelSize);
      BlockWrite(F, Beta1.V, ModelSize);

      // LayerNorm 2.
      BlockWrite(F, Gamma2.M, ModelSize);
      BlockWrite(F, Gamma2.V, ModelSize);

      BlockWrite(F, Beta2.M, ModelSize);
      BlockWrite(F, Beta2.V, ModelSize);
    end;
end;

// Save a WES3 model.
function SaveModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
var
  F: file;
  Header: TWES3ModelHeader;
  Checkpoint: TTrainingCheckpoint;
  TokenCount, PaddingCount: Int64;
begin
  Result := False;

  if CudaAllocated then begin
    CopyParamsToHost(Model);
    CopyAdamWStateToHost(AdamWState);
  end;

  FillChar(Header, SizeOf(Header), 0);

  TokenCount := nTokenizedCorpus;

  if TokenCount >= RawTokenCount then
    PaddingCount := TokenCount - RawTokenCount
  else
    PaddingCount := 0;

  // WES3 file information.
  Header.HeaderVersion := WES3ModelHeaderVersion;
  Header.HeaderSize := WES3ModelHeaderSize;
  Header.CheckpointVersion := WES3CheckpointVersion;
  Header.OptimizerVersion := WES3OptimizerVersion;
  Header.Flags := WES3FlagHasMLPGating;

  // Tokenizer/corpus information.
  Header.TokenizerKind := Ord(TokenizerKind);
  Header.CorpusID := CorpusID;
  Header.CorpusBytes := nCorpus;
  Header.RawTokenCount := RawTokenCount;
  Header.TokenCount := TokenCount;
  Header.PaddingCount := PaddingCount;
  Header.SymbolCount := nSymbols;

  // Use this if nMerges exists. Otherwise leave it zero.
  Header.MergeCount := nMerges;

  // Model architecture.
  Header.ModelDim := ModelDim;
  Header.ModelDimProj := ModelDimProj;
  Header.Proj := Proj;
  Header.NVocab := nVocab;
  Header.DimVocab := DimVocab;
  Header.NBlock := nBlock;
  Header.NHead := nHead;
  Header.SeqLen := SeqLen;

  // Model options.
  Header.ModelOptionsVersion := WES3ModelOptionsVersion;
  Header.NormKind := Ord(NormKind);
  Header.ActivationKind := Ord(ActivationKind);
  Header.ActivationAlpha := ActivationAlpha;
  Header.Reserved[0] := Byte(Ord(MLPGatingKind));

  // Program version.
  StoreProgramVersion(Header.ProgramVersion);

  // Capture everything needed to resume training.
  CaptureTrainingCheckpoint(Checkpoint);

  AssignFile(F, FileName);

  try
    Rewrite(F, 1);

    // WES3 model identification.
    BlockWrite(F, ModelMagic3, SizeOf(ModelMagic3));

    // Fixed 1024-byte WES3 header.
    BlockWrite(F, Header, SizeOf(Header));

    // Model parameters.
    BlockWrite(F, Model, SizeOf(Model));

    // Training state.
    BlockWrite(F, Checkpoint, SizeOf(Checkpoint));

    // AdamW M/V state.
    SaveAdamWState(F, AdamWState);

    CloseFile(F);
    Result := True;
  except
    try
      CloseFile(F);
    except
    end;

    Result := False;
  end;
end;

// Load AdamW first and second moments into host arrays. CUDA pointers are allocated separately.
procedure LoadAdamWState(var F: file; var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    BlockRead(F, M, EmbeddingsSize);
    BlockRead(F, V, EmbeddingsSize);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      BlockRead(F, Wq.M, WeightSize);
      BlockRead(F, Wq.V, WeightSize);

      BlockRead(F, Wk.M, WeightSize);
      BlockRead(F, Wk.V, WeightSize);

      BlockRead(F, Wv.M, WeightSize);
      BlockRead(F, Wv.V, WeightSize);

      BlockRead(F, W0.M, WeightSize);
      BlockRead(F, W0.V, WeightSize);

      // MLP weights.
      BlockRead(F, W1.M, WeightProjectedSize);
      BlockRead(F, W1.V, WeightProjectedSize);

      BlockRead(F, W2.M, WeightProjectedSize);
      BlockRead(F, W2.V, WeightProjectedSize);

      // Biases.
      BlockRead(F, b1.M, ProjectedSize);
      BlockRead(F, b1.V, ProjectedSize);

      BlockRead(F, b2.M, ModelSize);
      BlockRead(F, b2.V, ModelSize);

      // LayerNorm 1.
      BlockRead(F, Gamma1.M, ModelSize);
      BlockRead(F, Gamma1.V, ModelSize);

      BlockRead(F, Beta1.M, ModelSize);
      BlockRead(F, Beta1.V, ModelSize);

      // LayerNorm 2.
      BlockRead(F, Gamma2.M, ModelSize);
      BlockRead(F, Gamma2.V, ModelSize);

      BlockRead(F, Beta2.M, ModelSize);
      BlockRead(F, Beta2.V, ModelSize);
    end;
end;

// Load a WES3 model.
function LoadModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
var
  F: file;
  FileMagic: array[0..7] of Char;
  Header: TWES3ModelHeader;
  Checkpoint: TTrainingCheckpoint;
begin
  Result := False;

  AssignFile(F, FileName);

  try
    Reset(F, 1);

    if FileSize(F) < SizeOf(FileMagic) + SizeOf(Header) then begin
      CloseFile(F);
      Writeln('Invalid or incomplete WES3 model file.');
      Exit;
    end;

    // Read and validate WES3MODL.
    BlockRead(F, FileMagic, SizeOf(FileMagic));

    if not (
      (FileMagic[0] = 'W') and
      (FileMagic[1] = 'E') and
      (FileMagic[2] = 'S') and
      (FileMagic[3] = '3') and
      (FileMagic[4] = 'M') and
      (FileMagic[5] = 'O') and
      (FileMagic[6] = 'D') and
      (FileMagic[7] = 'L')) then begin
      CloseFile(F);
      Writeln('Invalid model file. WES3MODL header not found.');
      Exit;
    end;

    // Read fixed WES3 model header.
    BlockRead(F, Header, SizeOf(Header));

    if Header.HeaderSize <> WES3ModelHeaderSize then begin
      CloseFile(F);
      Writeln('Invalid WES3 model header size. File = ', Header.HeaderSize,
        '; expected = ', WES3ModelHeaderSize, '.');
      Exit;
    end;

    if Header.HeaderVersion <> WES3ModelHeaderVersion then begin
      CloseFile(F);
      Writeln('Unsupported WES3 model header version. File = ',
        Header.HeaderVersion, '; expected = ', WES3ModelHeaderVersion, '.');
      Exit;
    end;

    if Header.CheckpointVersion <> WES3CheckpointVersion then begin
      CloseFile(F);
      Writeln('Unsupported checkpoint version. File = ',
        Header.CheckpointVersion, '; expected = ', WES3CheckpointVersion, '.');
      Exit;
    end;

    if Header.OptimizerVersion <> WES3OptimizerVersion then begin
      CloseFile(F);
      Writeln('Unsupported optimizer version. File = ', Header.OptimizerVersion, '; expected = ', WES3OptimizerVersion, '.');
      Exit;
    end;

    if Header.TokenizerKind > Ord(High(TTokenizerKind)) then begin
      CloseFile(F);
      Writeln('Invalid tokenizer kind in WES3 model: ', Header.TokenizerKind, '.');
      Exit;
    end;

    // Validate architecture against current program settings.
    if Header.ModelDim <> ModelDim then begin
      CloseFile(F);
      Writeln('ModelDim mismatch. File = ', Header.ModelDim, '; Program = ', ModelDim, '.');
      Exit;
    end;

    if Header.ModelDimProj <> ModelDimProj then begin
      CloseFile(F);
      Writeln('ModelDimProj mismatch. File = ', Header.ModelDimProj, '; Program = ', ModelDimProj, '.');
      Exit;
    end;

    {if Header.Proj <> Proj then begin   // Mismatch possible if old model Proj = 14
      CloseFile(F);                     // And new model Proj = 12 or 8.
      Writeln('Proj mismatch. File = ', Header.Proj, '; Program = ', Proj, '.');
      Exit;
    end;}

    if Header.NVocab > DimVocab then begin
      CloseFile(F);
      Writeln('nVocab in model exceeds DimVocab. File = ', Header.NVocab,
        '; Program DimVocab = ', DimVocab, '.');
      Exit;
    end;

    if Header.DimVocab <> DimVocab then begin
      CloseFile(F);
      Writeln('DimVocab mismatch. File = ', Header.DimVocab,
        '; Program = ', DimVocab, '.');
      Exit;
    end;

    if Header.NBlock <> nBlock then begin
      CloseFile(F);
      Writeln('nBlock mismatch. File = ', Header.NBlock,
        '; Program = ', nBlock, '.');
      Exit;
    end;

    if Header.NHead <> nHead then begin
      CloseFile(F);
      Writeln('nHead mismatch. File = ', Header.NHead,
        '; Program = ', nHead, '.');
      Exit;
    end;

    if Header.SeqLen <> SeqLen then begin
      CloseFile(F);
      Writeln('SeqLen mismatch. File = ', Header.SeqLen,
        '; Program = ', SeqLen, '.');
      Exit;
    end;

    // Restore normalization and activation options.
    if Header.ModelOptionsVersion = 0 then begin

      // Older WES3 models used LayerNorm and ReLU.
      NormKind := LayerNorm;
      ActivationKind := ReLU;
      ActivationAlpha := 0.01;

    end
    else if Header.ModelOptionsVersion = WES3ModelOptionsVersion then begin

      if Header.NormKind > Ord(High(TNormKind)) then begin
        CloseFile(F);
        Writeln('Invalid normalization kind in WES3 model: ',
          Header.NormKind, '.');
        Exit;
      end;

      if Header.ActivationKind > Ord(High(TActivationKind)) then begin
        CloseFile(F);
        Writeln('Invalid activation kind in WES3 model: ',
          Header.ActivationKind, '.');
        Exit;
      end;

      NormKind := TNormKind(Header.NormKind);
      ActivationKind := TActivationKind(Header.ActivationKind);
      ActivationAlpha := Header.ActivationAlpha;

    end
    else begin
      CloseFile(F);
      Writeln('Unsupported model options version. File = ', Header.ModelOptionsVersion, '; Program = ', WES3ModelOptionsVersion, '.');
      Exit;
    end;

    // New WES3 models explicitly identify their MLP gating mode.
    // Earlier WES3 models had Flags=0 and did not record this information.
    if (Header.Flags and WES3FlagHasMLPGating) <> 0 then begin
      if Header.Reserved[0] > Ord(High(TMLPGatingKind)) then begin
        CloseFile(F);
        Writeln('Invalid MLP gating kind in WES3 model: ', Header.Reserved[0], '.');
        Exit;
      end;
      MLPGatingKind := TMLPGatingKind(Header.Reserved[0]);
    end
    else
      Writeln('This WES3 model does not store its MLP gating method. Using the current setting: ', WES3GatingName(MLPGatingKind),
        '. Verify it matches the model before training or inference.');

    // Restore model metadata.
    TokenizerKind := TTokenizerKind(Header.TokenizerKind);
    SetSpecialTokenIDsForKind(TokenizerKind);
    CorpusID := Header.CorpusID;
    nCorpus := Header.CorpusBytes;
    RawTokenCount := Header.RawTokenCount;
    nTokenizedCorpus := Header.TokenCount;
    PaddedTokenCount := nTokenizedCorpus;
    nSymbols := Header.SymbolCount;
    nMerges := Header.MergeCount;
    MergeCount := nMerges;

    // nVocab may be smaller than DimVocab and should come from the model.
    nVocab := Header.NVocab;

    // Read model parameters.
    BlockRead(F, Model, SizeOf(Model));

    // Read saved training state.
    BlockRead(F, Checkpoint, SizeOf(Checkpoint));

    // Read AdamW M/V state.
    LoadAdamWState(F, AdamWState);

    CloseFile(F);

    RestoreTrainingCheckpoint(Checkpoint);
    AdamWStateLoaded := True;

    // Host record contains pointer values written into the file. These must never be treated as valid device pointers.
    ClearDevicePointers(Model);

    NewModel := False;
    Result := True;

    // Display results of load.
    Writeln('Loaded WES3 model: ', FileName, '. Norm = ', NormKindName(NormKind), '; Activation = ', ActivationKindName(ActivationKind),
      '; Gating = ', WES3GatingName(MLPGatingKind), '; alpha = ', ActivationAlpha:0:4, '.');

  except
    try
      CloseFile(F);
    except
    end;

    Result := False;
  end;
end;

{ TinyStories conversion }
// Convert <|endoftext|> story separators to byte 254.
procedure ConvertTSEndOfText(const FileName: string);
const
  Marker: AnsiString = '<|endoftext|>';
var
  F: file;
  Corpus: TBVector;
  Source, Dest, j, BytesRead, Count: Integer;
  Match: Boolean;
begin
  if not FileExists(FileName) then begin
    Writeln('File not found: ', FileName);
    Exit;
  end;

  // Read entire file.
  Assign(F, FileName);
  Reset(F, 1);

  SetLength(Corpus, FileSize(F));

  if Length(Corpus) > 0 then
    BlockRead(F, Corpus[0], Length(Corpus), BytesRead)
  else
    BytesRead := 0;

  Close(F);

  if BytesRead <> Length(Corpus) then begin
    Writeln('Error reading file: ', FileName);
    Exit;
  end;

  // Replace <|endoftext|> with byte 254.
  Source := 0;
  Dest := 0;
  Count := 0;

  while Source < Length(Corpus) do begin
    Match := Source + Length(Marker) <= Length(Corpus);

    if Match then
      for j := 1 to Length(Marker) do
        if Corpus[Source + j - 1] <> Ord(Marker[j]) then begin
          Match := False;
          Break;
        end;

    if Match then begin
      Corpus[Dest] := 254;
      Inc(Dest);
      Inc(Source, Length(Marker));
      Inc(Count);
    end
    else begin
      Corpus[Dest] := Corpus[Source];
      Inc(Dest);
      Inc(Source);
    end;
  end;

  SetLength(Corpus, Dest);

  // Rewrite converted file.
  Assign(F, FileName);
  Rewrite(F, 1);

  if Length(Corpus) > 0 then
    BlockWrite(F, Corpus[0], Length(Corpus));

  Close(F);

  Writeln('Converted ', Count, ' <|endoftext|> separators in ', FileName, '.');
  Writeln('New file size = ', Length(Corpus), ' bytes.');
end;

// Convert UTF-8 black-square story separators to byte 254.
procedure ConvertTSSeparators(const FileName: string);
var
  F: file;
  Corpus: TBVector;
  Source, Dest, BytesRead, Count: Integer;
begin
  if not FileExists(FileName) then begin
    Writeln('File not found: ', FileName);
    Exit;
  end;

  // Read entire file.
  Assign(F, FileName);
  Reset(F, 1);

  SetLength(Corpus, FileSize(F));

  if Length(Corpus) > 0 then
    BlockRead(F, Corpus[0], Length(Corpus), BytesRead)
  else
    BytesRead := 0;

  Close(F);

  if BytesRead <> Length(Corpus) then begin
    Writeln('Error reading file: ', FileName);
    Exit;
  end;

  // Replace UTF-8 black square E2 96 A0 with byte 254.
  Source := 0;
  Dest := 0;
  Count := 0;

  while Source < Length(Corpus) do begin
    if (Source + 2 < Length(Corpus)) and
       (Corpus[Source] = 226) and
       (Corpus[Source + 1] = 150) and
       (Corpus[Source + 2] = 160) then begin

      Corpus[Dest] := 254;
      Inc(Dest);
      Inc(Source, 3);
      Inc(Count);
    end
    else begin
      Corpus[Dest] := Corpus[Source];
      Inc(Dest);
      Inc(Source);
    end;
  end;

  SetLength(Corpus, Dest);

  // Rewrite file with converted corpus.
  Assign(F, FileName);
  Rewrite(F, 1);

  if Length(Corpus) > 0 then
    BlockWrite(F, Corpus[0], Length(Corpus));

  Close(F);

  Writeln('Converted ', Count, ' TinyStories separators in ', FileName, '.');
  Writeln('New file size = ', Length(Corpus), ' bytes.');
end;

end.
