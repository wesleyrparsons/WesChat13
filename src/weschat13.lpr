program weschat13;

{$mode ObjFPC}{$H+}{$R-}{$I proprietary.txt}  // Use long strings. Disable range checking.

{ WesChat Version 1.3 begun January 10, 2026 by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }
{ Last edits: October 21, 2026, 2 pm }
{ Main program for work folders, tokenization, training, inference, and predefined model workflows }
{ TokenizedCorpusPresent, SymbolTablePresent, and ModelPresent are maintained only by predefined-model workflows }
{ ExistingWorkRoot is the root for work folders, WorkingDir is the selected work directory, and WorkRoot is its normalized form }

uses
  { RTL and platform units }
  Classes, Crt, FileUtil, ShellAPI, StrUtils, SysUtils, Windows,
  { WesChat units }
  CombineTables, Display, Global, GPT2Tokenize, Infer, IOHandler, Matrix, OutputHead, Symbolize, Train, UDTag, Util, WesTokenize;

var
  { Corpus data }
  Corpus: TBVector;
  TokenizedCorpus: TIVector;

  { Model data }
  WModelParams: TWModelParams;
  WModelState: TWModelState;
  WAdamWState: TWAdamWState;

  { Minimum accepted sizes }
  MinSymbols: Integer = 50;
  MinTokens: Integer = 50;
  MinCorpus: Integer = 50;

  { Miscellaneous state }
  Ch: string;
  CombinedSymbolTable: TSymbolTable;

{ General helpers }
// Return a filename-safe timestamp.
function TimeStamp: string;
begin
  Result := FormatDateTime('yyyy-mm-dd_hhnnss', Now);
end;

// Prompt for a yes-or-no response and apply the requested default.
function AskYesNo(const Prompt: string; DefaultYes: Boolean = True): Boolean;
var
  S: string;
begin
  Write(Prompt);

  if DefaultYes then
    Write(' (Y/n) ')
  else
    Write(' (y/N) ');

  if YesToAll then begin
    Result := True;
    Writeln('Yes, based on yes to all.');
  end
  else begin
    Readln(S);
    S := UpperCase(Trim(S));

    if S = '' then
      Result := DefaultYes
    else
      Result := S[1] = 'Y';
  end;
end;

// Prompt for a menu choice and return an uppercase trimmed response.
function AskChoice(const Prompt, Choices: string): string;
begin
  Write(Prompt, ' [', Choices, ']: ');
  Readln(Result);
  Result := UpperCase(Trim(Result));
end;

// Append one integer token vector to another.
procedure AppendTokens(var Dest: TIVector; const Src: TIVector);
var
  OldLen, i: Integer;
begin
  OldLen := Length(Dest);
  SetLength(Dest, OldLen + Length(Src));

  for i := 0 to High(Src) do
    Dest[OldLen + i] := Src[i];
end;


{ Work folder and file helpers }

 // Establish WesChat directories relative to the executable.
procedure InitializeWesChatFolders;
var
  ExeDir: string;
begin
  // Assume the executable is in WesChatRoot\bin.
  ExeDir := ExtractFileDir(ExpandFileName(ParamStr(0)));

  // The parent of bin is the WesChat installation root.
  WesChatRoot := IncludeTrailingPathDelimiter(ExtractFileDir(ExeDir));

  ExistingWorkRoot := WesChatRoot + 'run' + PathDelim;
  SharedCorpusDir := WesChatRoot + 'cor' + PathDelim;
  DataDir := WesChatRoot + 'data' + PathDelim;
  ToolsDir := WesChatRoot + 'tools' + PathDelim;

  UDPipeFileName := ToolsDir + 'udpipe.exe';
  UDModelFileName := DataDir + 'english-ewt-ud-2.4-190531.udpipe';

  if not ForceDirectories(ExistingWorkRoot) then
    raise Exception.Create('Cannot create run directory: ' + ExistingWorkRoot);

  if not ForceDirectories(SharedCorpusDir) then
    raise Exception.Create('Cannot create corpus directory: ' + SharedCorpusDir);

  Writeln('WesChat root = ', WesChatRoot, '; Run root     = ', ExistingWorkRoot, '; Corpus root  = ', SharedCorpusDir, '.');
end;

// Normalize the root used to locate predefined work folders.
procedure NormalizeExistingWorkRoot;
begin
  ExistingWorkRoot := Trim(ExistingWorkRoot);

  if ExistingWorkRoot = '' then
    ExistingWorkRoot := WesChatRoot + 'run' + PathDelim;

  ExistingWorkRoot := IncludeTrailingPathDelimiter(ExpandFileName(ExistingWorkRoot));
end;

// Resolve a work-folder name against ExistingWorkRoot.
function ResolveWorkFolder(const FolderName: string): string;
var
  S: string;
begin
  NormalizeExistingWorkRoot;

  S := Trim(FolderName);

  if S = '' then
    S := 'WesChatWork';

  if IsAbsolutePath(S) then
    Result := ExpandFileName(S)
  else
    Result := ExpandFileName(ExistingWorkRoot + S);
end;

// Open the current work folder in Windows Explorer.
procedure OpenWorkFolderInExplorer;
begin
  if Trim(WorkRoot) = '' then begin
    Writeln('WorkRoot is blank.');
    Exit;
  end;

  ShellExecute(0, 'open', PChar(WorkRoot), nil, nil, SW_SHOWNORMAL);
end;

// Set the active work identity and the default output filenames.
procedure SetWorkIdentity(const RootDir, BaseName: string);
begin
  WorkingDir := IncludeTrailingPathDelimiter(ExpandFileName(RootDir));
  InitWorkFolders(WorkingDir);

  CurrentBaseName := BaseName;
  WorkingName := BaseName;

  BestModelFileName := ModelDir + WorkingName + '_best.model';
  TokenFileName := TokenDir + WorkingName + '.tok';
  SymbolFileName := SymbolDir + WorkingName + '.sym';
  RunFileName := RunDir + WorkingName + '.run';
  LogFileName := LogDir + WorkingName + '.log';

  Writeln('Using work folder: ', WorkRoot, '. Using tokenizer: ', TokenizerKindName(TokenizerKind), '.');
end;

// Select an existing work folder and database name.
procedure SelectExistingWork(const FolderName, BaseName: string);
begin
  SetWorkIdentity(ResolveWorkFolder(FolderName), BaseName);
end;

// Return True when a filename includes a directory path.
function PathHasDirectory(const S: string): Boolean;
begin
  Result := ExtractFilePath(S) <> '';
end;

// Resolve an input filename against the current and preferred folders.
function ResolveInputFile(const UserName, PreferredDir: string): string;
var
  S: string;
begin
  S := Trim(UserName);
  Result := S;
  if S = '' then Exit;

  // Use an existing filename or path exactly as entered.
  if FileExists(S) then begin
    Result := ExpandFileName(S);
    Exit;
  end;

  // Search for a bare filename.
  if not PathHasDirectory(S) then begin
    if FileExists(WorkingDir + S) then begin
      Result := WorkingDir + S;
      Exit;
    end;

    if FileExists(PreferredDir + S) then begin
      Result := PreferredDir + S;
      Exit;
    end;

    // Search the shared corpus directory for corpus inputs.
    if SameText(ExcludeTrailingPathDelimiter(PreferredDir), ExcludeTrailingPathDelimiter(CorpusDir)) then begin
      if FileExists(SharedCorpusDir + S) then begin
        Result := SharedCorpusDir + S;
        Exit;
      end;
    end;
  end;
end;

// Build an output filename from the user entry or a default name.
function MakeOutputFileName(const UserName, DefaultDir, BaseName, Ext: string): string;
var
  S: string;
begin
  S := Trim(UserName);

  if S = '' then
    S := ChangeFileExt(BaseName, Ext);

  if ExtractFileExt(S) = '' then
    S := S + Ext;

  if PathHasDirectory(S) then
    Result := ExpandFileName(S)
  else
    Result := DefaultDir + S;
end;

// Resolve a filename and require that the file exists.
function RequireExistingFile(var FileName: string; const PreferredDir: string): Boolean;
begin
  FileName := ResolveInputFile(FileName, PreferredDir);
  Result := FileExists(FileName);

  if not Result then
    Writeln('File not found: ', FileName, '.');
end;

// Require that a file meets the requested minimum size.
function RequireMinFileSize(const FileName: string; MinSize: Integer): Boolean;
begin
  Result := FileSize(FileName) >= MinSize;

  if not Result then
    Writeln('File too small: ', FileName, '. Size = ', FileSize(FileName), ' minimum = ', MinSize, '.');
end;

// Return the default symbol-statistics filename.
function DefaultSymbolStatsFile(const BaseName: string): string;
begin
  Result := LogDir + CleanBaseName(BaseName) + '.sym.tok';
end;

// Return the default symbol-table filename.
function DefaultSymbolFile(const BaseName: string): string;
begin
  Result := SymbolDir + ChangeFileExt(CleanBaseName(BaseName), '.sym');
end;

// Return the default merge-table filename.
function DefaultMergeFile(const BaseName: string): string;
begin
  Result := MergeDir + ChangeFileExt(CleanBaseName(BaseName), '.mer');
end;

// Return the default token-list filename.
function DefaultTokenFile(const BaseName: string): string;
begin
  Result := TokenDir + ChangeFileExt(CleanBaseName(BaseName), '.tok');
end;

// Return the default timestamped model filename.
function DefaultModelFile(const BaseName: string): string;
begin
  Result := ModelDir + CleanBaseName(BaseName) + '_' + TimeStamp + '.model';
end;

// Return the default tokenization-log filename.
function DefaultTokenLogFile(const BaseName: string): string;
begin
  Result := LogDir + ChangeFileExt(CleanBaseName(BaseName), '.tok.log');
end;

// Return the default timestamped program-log filename.
function DefaultLogFile(const BaseName: string): string;
begin
  Result := LogDir + CleanBaseName(BaseName) + '_' + TimeStamp + '.log';
end;

{ Tokenizer, corpus, and token-list helpers }
// Determine the tokenizer kind from a token or symbol filename suffix.
function TokenizerKindFromFileName(const FileName: string; out Kind: TTokenizerKind): Boolean;
var
  S: string;
begin
  S := LowerCase(ExtractFileName(FileName));

  if EndsText('_ud.tok', S) or EndsText('_ud.sym', S) or EndsText('.ud.tok', S) or EndsText('.ud.sym', S) then
    Kind := UDTokenizer
  else if EndsText('_g.tok', S) or EndsText('_g.sym', S) or EndsText('_gpt2.tok', S) or EndsText('_gpt2.sym', S) or
          EndsText('.g.tok', S) or EndsText('.g.sym', S) or EndsText('.gpt2.tok', S) or EndsText('.gpt2.sym', S) then
    Kind := GPT2Tokenizer
  else if EndsText('_w.tok', S) or EndsText('_w.sym', S) or EndsText('_wes.tok', S) or EndsText('_wes.sym', S) or
          EndsText('_west.tok', S) or EndsText('_west.sym', S) or EndsText('.w.tok', S) or EndsText('.w.sym', S) or
          EndsText('.wes.tok', S) or EndsText('.wes.sym', S) or EndsText('.west.tok', S) or EndsText('.west.sym', S) then
    Kind := WesTokenizer
  else begin
    Result := False;
    Exit;
  end;

  Result := True;
end;

// Set the active tokenizer and its special-token values.
procedure SetTokenizerMode(const NewTokenizerKind: TTokenizerKind);
begin
  TokenizerKind := NewTokenizerKind;

  case TokenizerKind of
    WesTokenizer, UDTokenizer: begin
      BOS := 256;
      EOS := 257;
      PAD := 258;
      UNK := 259;
      nVocab := nSymbols;
    end;

    GPT2Tokenizer: begin
      BOS := GPT2BOS;
      EOS := GPT2EOS;
      PAD := GPT2PAD;
      UNK := GPT2UNK;
      nVocab := GPT2ModelVocabSize;
      VocabFileName := DataDir + 'vocab1.json';
      MergeFileName := DataDir + 'merges.txt';
    end;
  end;
end;

// Set raw and padded token counts for an already loaded token list.
procedure SetLoadedTokenCounts(var Tokens: TIVector);
begin
  // Loaded token lists may already contain padding.
  RawTokenCount := Length(Tokens);

  while (RawTokenCount > 0) and (Tokens[RawTokenCount - 1] = PAD) do
    Dec(RawTokenCount);

  PadToSeqMultiple(Tokens, SeqLen);

  PaddedTokenCount := Length(Tokens);
  nTokenizedCorpus := PaddedTokenCount;
end;

// Prompt for one corpus file, read it, and record its metadata.
function ReadCorpusFilePrompt(var OutCorpusFileName: string; var OutCorpus: TBVector): Boolean;
begin
  Result := False;

  Write('Enter corpus file name: ');
  Readln(OutCorpusFileName);

  if not RequireExistingFile(OutCorpusFileName, CorpusDir) then Exit;

  if not RequireMinFileSize(OutCorpusFileName, MinCorpus) then Exit;

  ReadFileBytes(OutCorpusFileName, OutCorpus);
  nCorpus := Length(Corpus);
  CorpusID := ComputeCorpusID(Corpus);

  CurrentBaseName := CleanBaseName(OutCorpusFileName);

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] := OutCorpusFileName + '   ' + IntToStr(FileSize(OutCorpusFileName)) + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(OutCorpusFileName)));

  Result := True;
end;

// Read a list of corpus files and concatenate their bytes.
procedure ProcessFileList(var ListFileName: string; var OutCorpus: TBVector);
var
  F: TextFile;
  Line, FullName, ListBaseDir: string;
  OneCorpus: TBVector;
  Count: Integer;
begin
  MultipleFileName := EmptyStr;

  Write('Enter name of file list: ');
  Readln(ListFileName);

  if not RequireExistingFile(ListFileName, ListDir) then Exit;

  CurrentBaseName := CleanBaseName(ListFileName);
  ListBaseDir := ExtractFilePath(ListFileName);

  AssignFile(F, ListFileName);
  Reset(F);

  Count := 0;
  FromSymbolTable := False;
  SetLength(OutCorpus, 0);
  SetLength(CorpusFileNames, 0);

  while not EOF(F) do begin
    ReadLn(F, Line);
    Line := Trim(Line);

    if Line = '' then Continue;

    FullName := Line;

    if not FileExists(FullName) then
      FullName := ListBaseDir + Line;

    if not FileExists(FullName) then
      FullName := CorpusDir + Line;

    if not FileExists(FullName) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;

    if FileSize(FullName) < MinCorpus then begin
      Writeln('  Corpus too small, skipping: ', FullName);
      Continue;
    end;

    ReadFileBytes(FullName, OneCorpus);

    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] := FullName + '   ' + IntToStr(FileSize(FullName)) + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(FullName)));

    OutCorpus := Concat(OutCorpus, OneCorpus);
    nCorpus := Length(OutCorpus);

    Writeln('  File processed: ', FullName, '; corpus bytes read: ', Length(OneCorpus), '.');
    Writeln('  Total bytes read: ', Length(OutCorpus), '.');

    Inc(Count);
  end;

  CloseFile(F);

  Writeln('Combined corpus length = ', Length(OutCorpus), '.');

  nCorpus := Length(OutCorpus);
  CorpusID := ComputeCorpusID(OutCorpus);
end;

// Run WesTokenize while suppressing its automatic file saves.
procedure RunWesTokenizeNoAutoSave(const InCorpus: TBVector; var OutTokens: TIVector);
var
  OldSaveFiles: Boolean;
  OldSaveTokenizationFiles: Boolean;
begin
  OldSaveFiles := SaveFiles;
  OldSaveTokenizationFiles := SaveTokenizationFiles;

  SaveFiles := False;
  SaveTokenizationFiles := False;

  try
    RunWesTokenize(InCorpus, OutTokens);
  finally
    SaveTokenizationFiles := OldSaveTokenizationFiles;
    SaveFiles := OldSaveFiles;
  end;
end;

// Tokenize a file list.
procedure TokenizeWesFileList(var ListFileName: string; var OutTokens: TIVector);
var
  F: TextFile;
  Line, FullName, ListBaseDir: string;
  OneCorpus: TBVector;
  OneTokens: TIVector;
  Count, OldLen: Integer;
begin
  Write('Enter name of file list: ');
  Readln(ListFileName);

  if not RequireExistingFile(ListFileName, ListDir) then Exit;

  CurrentBaseName := CleanBaseName(ListFileName);
  ListBaseDir := ExtractFilePath(ListFileName);

  AssignFile(F, ListFileName);
  Reset(F);

  Count := 0;
  SetLength(OutTokens, 0);
  SetLength(CorpusFileNames, 0);

  FromSymbolTable := True;
  ResetWesTrie;

  try
    while not EOF(F) do begin
      Readln(F, Line);
      Line := Trim(Line);

      if Line = '' then Continue;

      FullName := Line;

      if not FileExists(FullName) then
        FullName := ListBaseDir + Line;

      if not FileExists(FullName) then
        FullName := CorpusDir + Line;

      if not FileExists(FullName) then begin
        Writeln('File not found: ', Line);
        Continue;
      end;

      if FileSize(FullName) < MinCorpus then begin
        Writeln('Corpus too small: ', FullName);
        Continue;
      end;

      ReadFileBytes(FullName, OneCorpus);

      SetLength(OneTokens, 0);
      RunWesTokenizeNoAutoSave(OneCorpus, OneTokens);

      // Remove any trailing padding or EOS already added by the tokenizer.
      while (Length(OneTokens) > 0) and ((OneTokens[High(OneTokens)] = PAD) or (OneTokens[High(OneTokens)] = EOS)) do
        SetLength(OneTokens, Length(OneTokens) - 1);

      if Length(OneTokens) = 0 then Continue;

      // Insert EOS between documents.
      if Count > 0 then begin
        OldLen := Length(OutTokens);
        SetLength(OutTokens, OldLen + 1);
        OutTokens[OldLen] := EOS;
      end;

      AppendTokens(OutTokens, OneTokens);

      SetLength(CorpusFileNames, Count + 1);
      CorpusFileNames[Count] := FullName;

      Inc(Count);

      Writeln('Tokenized: ', FullName);
      Writeln('Document tokens = ', Length(OneTokens), '; Total tokens = ', Length(OutTokens));
    end;
  finally
    CloseFile(F);
  end;

  RawTokenCount := Length(OutTokens);

  // Pad only once, after combining all documents.
  PadToSeqMultiple(OutTokens, SeqLen);

  PaddedTokenCount := Length(OutTokens);
  nTokenizedCorpus := PaddedTokenCount;

  Writeln;
  Writeln('Documents processed = ', Count);
  Writeln('Raw tokens = ', RawTokenCount);
  Writeln('Padded tokens = ', PaddedTokenCount);
end;

{ File loading and saving }
// Prompt for and load a symbol table.
function LoadSymbolTablePrompt: Boolean;
var
  DetectedKind: TTokenizerKind;
begin
  Result := False;

  Write('Input symbol table file name: ');
  Readln(SymbolFileName);

  if not RequireExistingFile(SymbolFileName, SymbolDir) then Exit;

  if TokenizerKindFromFileName(SymbolFileName, DetectedKind) then begin
    SetTokenizerMode(DetectedKind);
    Writeln('Tokenizer selected from symbol file name: ', TokenizerKindName(TokenizerKind));
  end;

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);
  ResetWesTrie;

  if Length(SymbolTable) < MinSymbols then begin
    Writeln('Too few symbols found. Length(SymbolTable) = ', Length(SymbolTable));
    Exit;
  end;

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  Writeln('Using symbol table: ', SymbolFileName);

  Result := True;
end;

// Prompt for and load a token list.
function LoadTokenListPrompt: Boolean;
var
  DetectedKind: TTokenizerKind;
begin
  Result := False;

  Write('Enter token list file name: ');
  Readln(TokenFileName);

  if not RequireExistingFile(TokenFileName, TokenDir) then Exit;

  if TokenizerKindFromFileName(TokenFileName, DetectedKind) then begin
    SetTokenizerMode(DetectedKind);
    Writeln('Tokenizer selected from token file name: ', TokenizerKindName(TokenizerKind));
  end;

  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);
  SetLoadedTokenCounts(TokenizedCorpus);

  if Length(TokenizedCorpus) < MinTokens then begin
    Writeln('Token list too small. Length = ', Length(TokenizedCorpus));
    Exit;
  end;

  CurrentBaseName := CleanBaseName(TokenFileName);

  Writeln('Using token list: ', TokenFileName);

  Result := True;
end;

// Prompt for and load a saved model.
function LoadModelPrompt: Boolean;
begin
  Result := False;

  Write('Enter model file name: ');
  Readln(ModelFileName);

  if not RequireExistingFile(ModelFileName, ModelDir) then Exit;

  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState, WAdamWState);

  if LoadModel(ModelFileName, WModelParams, WAdamWState) then begin
    Write('Loading model: ', ModelFileName);
    Write('; Model size: ', FileSize(ModelFileName), ' bytes');
    Writeln('; Model date: ', DateTimeToStr(FileDateToDateTime(FileAge(ModelFileName))), '.');
    NewModel := False;
    ParamsNeedCopyToDevice := True;
    Result := True;
  end
  else Writeln('File not loaded.');
end;

// Save the current symbol table, merge table, and symbol statistics.
procedure SaveSymbolizationFilesDefault(const BaseName: string);
begin
  if Length(SymbolTable) = 0 then begin
    Writeln('No symbol table to save.');
    Exit;
  end;

  Writeln('--- Saving Symbolization Files ---');

  SymbolFileName := DefaultSymbolFile(BaseName);
  SaveSymbolTable(SymbolFileName, SymbolTable);

  if Length(Merges) > 0 then
    SaveMergeTable(Merges, DefaultMergeFile(BaseName))
  else
    Writeln('No merges to save.');

  SaveMetaData(DefaultSymbolStatsFile(BaseName));
end;

// Save the current token list and tokenization log.
procedure SaveTokenizationFilesDefault(const BaseName: string);
begin
  if Length(TokenizedCorpus) = 0 then begin
    Writeln('No token list to save.');
    Exit;
  end;

  Writeln('--- Saving Tokenization Files ---');

  TokenFileName := DefaultTokenFile(BaseName);
  SaveTokenList(TokenizedCorpus, TokenFileName);

  SaveTokenizationLog(TokenizedCorpus, DefaultTokenLogFile(BaseName));
end;

// Save the current token list under its default filename.
procedure SaveCurrentTokenListDefault;
begin
  if Length(TokenizedCorpus) = 0 then Exit;

  TokenFileName := DefaultTokenFile(CurrentBaseName);
  SaveTokenList(TokenizedCorpus, TokenFileName);
end;

// Prompt for and save the current token list.
procedure MaybeSaveTokenList;
var
  S: string;
begin
  if Length(TokenizedCorpus) = 0 then Exit;

  if AskYesNo('Save token list?', True) then begin
    Write('Enter token list file name, blank for ', ExtractFileName(DefaultTokenFile(CurrentBaseName)), ': ');
    Readln(S);

    TokenFileName := MakeOutputFileName(S, TokenDir, CurrentBaseName, '.tok');
    SaveTokenList(TokenizedCorpus, TokenFileName);
  end;
end;

// Prompt for and save the current model.
procedure MaybeSaveModel;
var
  S: string;
begin
  if AskYesNo('Save model?', True) then begin
    Write('Output model file name, blank for ', ExtractFileName(DefaultModelFile(CurrentBaseName)), ': ');
    Readln(S);

    ModelFileName := MakeOutputFileName(S, ModelDir, CurrentBaseName + '_' + TimeStamp, '.model');

    if SaveModel(ModelFileName, WModelParams, WAdamWState) then
      Writeln('File ', ModelFileName, ' successfully saved.')
    else
      Writeln('File not saved.');
  end;
end;

// Write the current program information to a timestamped log.
procedure WriteInfoLog(const BaseName: string);
var
  SaveOut: Text;
  LogName: string;
begin
  if not SaveFiles then Exit;

  LogName := DefaultLogFile(BaseName);

  SaveOut := Output;

  Assign(Output, LogName);
  Rewrite(Output);

  ReportProgramInfo;

  Close(Output);
  Output := SaveOut;

  Writeln('Log written: ', LogName, '.');
end;


{ Tokenization workflows }
// Run GPT2Tokenize while suppressing its automatic file saves.
procedure RunGPT2TokenizeNoAutoSave(const InFileName: string; var OutTokens: TIVector);
var
  OldSaveFiles: Boolean;
begin
  OldSaveFiles := SaveFiles;
  SaveFiles := False;

  try
    RunGPT2TokenizeFile(InFileName, OutTokens);
  finally
    SaveFiles := OldSaveFiles;
  end;
  RawTokenCount := Length(TokenizedCorpus);
end;

// Symbolize and tokenize Corpus with the currently selected Wes-family tokenizer.
procedure TokenizePreparedWesCorpus;
var
  SymbolChoice: string;
begin
  SetLength(TokenizedCorpus, 0);

  if not YesToAll then
    SymbolChoice := AskChoice('Symbol table: C = create from corpus, S = load existing symbol table', 'C/S')
  else
    SymbolChoice := 'C';

  case SymbolChoice of
    'S': begin
      if not LoadSymbolTablePrompt then Exit;
    end;

    'C': begin
      FromSymbolTable := False;
      nSymbols := 0;
      nVocab := 0;

      ResetWesTrie;
      SetLength(SymbolTable, 0);

      Symbolize.RunSymbolize(Corpus);

      nSymbols := Length(SymbolTable);
      nVocab := nSymbols;

      Writeln('After RunSymbolize: nSymbols = ', nSymbols, '; nVocab = ', nVocab);

      if nSymbols < MinSymbols then begin
        Writeln('Too few symbols found after symbolization. nSymbols = ', nSymbols);
        Exit;
      end;

      ResetWesTrie;

      if AskYesNo('Save symbolization files?', True) then
        SaveSymbolizationFilesDefault(CurrentBaseName);
    end;

    else begin
      Writeln('Invalid symbol table selection.');
      Exit;
    end;
  end;

  Writeln('Tokenizing with ', TokenizerKindName(TokenizerKind), '...');

  if VerboseTokenize then
    Writeln('Before WesTokenize: Length(SymbolTable) = ', Length(SymbolTable), '; nSymbols = ', nSymbols, '; nVocab = ', nVocab, '.');

  RunWesTokenizeNoAutoSave(Corpus, TokenizedCorpus);

  RawTokenCount := Length(TokenizedCorpus);

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  PaddedTokenCount := Length(TokenizedCorpus);
  nTokenizedCorpus := PaddedTokenCount;

  Writeln(TokenizerKindName(TokenizerKind), ' tokenization complete. Raw tokens = ', RawTokenCount, '; Padded tokens = ', PaddedTokenCount, '; Padding added = ', PaddedTokenCount - RawTokenCount, '; Symbols = ', nSymbols, '.');

  if AskYesNo('Save tokenization files?', True) then
    SaveTokenizationFilesDefault(CurrentBaseName);
end;

// Read a corpus and tokenize it with WesTokenizer.
procedure TokenizeWithWes;
var
  SourceChoice: string;
begin
  SetTokenizerMode(WesTokenizer);

  SourceChoice := AskChoice('Corpus source: F = one file, L = list of corpus file names', 'F/L');

  if SourceChoice = 'L' then begin
    ProcessFileList(ListFile, Corpus);

    if Length(Corpus) < MinCorpus then begin
      Writeln('Combined corpus too small. Aborting tokenization.');
      Exit;
    end;
  end
  else
    if not ReadCorpusFilePrompt(CorpusFileName, Corpus) then Exit;

  // Keep the explicit tokenizer selection authoritative.
  SetTokenizerMode(WesTokenizer);

  TokenizePreparedWesCorpus;
end;

// Create a UD-tagged corpus file with UDPipe.
function CreateUDTaggedCorpus(out TaggedFileName: string): Boolean;
var
  InputFileName, S: string;
begin
  Result := False;

  Writeln;
  Writeln('--- Universal Dependencies Tokenization ---');

  Write('Enter input corpus file name: ');
  ReadLn(InputFileName);
  InputFileName := Trim(InputFileName);

  if InputFileName = '' then Exit;

  if not RequireExistingFile(InputFileName, CorpusDir) then Exit;

  TaggedFileName := ChangeFileExt(InputFileName, '') + '_ud.txt';

  Write('Enter tagged corpus file name, blank for ', TaggedFileName, ': ');
  ReadLn(S);

  if Trim(S) <> '' then
    TaggedFileName := Trim(S);

  try
    Writeln('Tagging corpus with Universal Dependencies...');

    UDTagFile(UDPipeFileName, UDModelFileName, InputFileName, TaggedFileName);

    SetTokenizerMode(UDTokenizer);

    Writeln('UD tagging complete. Tagged corpus: ', TaggedFileName, '. Tokenizer is ', TokenizerKindName(TokenizerKind), '.');

    Result := True;

  except
    on E: Exception do
      Writeln('UD tagging error: ', E.Message);
  end;
end;

// Create UD tags, then symbolize and tokenize the tagged corpus.
procedure TokenizeWithUD;
begin
  SetTokenizerMode(UDTokenizer);

  if not CreateUDTaggedCorpus(CorpusFileName) then Exit;

  if not RequireMinFileSize(CorpusFileName, MinCorpus) then Exit;

  ReadFileBytes(CorpusFileName, Corpus);
  nCorpus := Length(Corpus);
  CorpusID := ComputeCorpusID(Corpus);

  CurrentBaseName := CleanBaseName(CorpusFileName);

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName)) + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

  // Keep UDTokenizer selected even though the underlying BPE engine is WesTokenize.
  SetTokenizerMode(UDTokenizer);

  Writeln('Using UD-tagged corpus: ', CorpusFileName, '. Tokenizer is ', TokenizerKindName(TokenizerKind), '.');

  TokenizePreparedWesCorpus;
end;

// Tokenize one corpus file with GPT-2.
procedure TokenizeGPTSingleFile;
var
  PaddedTokenCount, RawTokenCount: Integer;
begin
  SetLength(TokenizedCorpus, 0);

  if not ReadCorpusFilePrompt(CorpusFileName, Corpus) then Exit;

  SetTokenizerMode(GPT2Tokenizer);
  RunGPT2TokenizeNoAutoSave(CorpusFileName, TokenizedCorpus);

  RawTokenCount := Length(TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  PaddedTokenCount := Length(TokenizedCorpus);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('GPT tokenization complete. Raw tokens = ', RawTokenCount, '; Padded tokens = ', PaddedTokenCount, '; Padding added = ', PaddedTokenCount - RawTokenCount, '; Vocabulary = ', nVocab, '.');

  MaybeSaveTokenList;
end;

// Tokenize and concatenate a list of corpus files with GPT-2.
procedure TokenizeGPTFileList;
var
  F: TextFile;
  Line, FullName, ListBaseDir: string;
  OneTokens: TIVector;
  Count: Integer;
begin
  SetLength(TokenizedCorpus, 0);

  Write('Enter name of file list: ');
  Readln(ListFile);

  if not RequireExistingFile(ListFile, ListDir) then Exit;

  CurrentBaseName := CleanBaseName(ListFile);
  ListBaseDir := ExtractFilePath(ListFile);

  AssignFile(F, ListFile);
  Reset(F);

  Count := 0;
  SetLength(CorpusFileNames, 0);

  while not EOF(F) do begin
    Readln(F, Line);
    Line := Trim(Line);

    if Line = '' then Continue;

    FullName := Line;

    if not FileExists(FullName) then
      FullName := ListBaseDir + Line;

    if not FileExists(FullName) then
      FullName := CorpusDir + Line;

    if not FileExists(FullName) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;

    if FileSize(FullName) < MinCorpus then begin
      Writeln('  Corpus too small, skipping: ', FullName);
      Continue;
    end;

    SetLength(OneTokens, 0);
    SetTokenizerMode(GPT2Tokenizer);
    RunGPT2TokenizeNoAutoSave(FullName, OneTokens);
    AppendTokens(TokenizedCorpus, OneTokens);

    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] := FullName + '   ' + IntToStr(FileSize(FullName)) + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(FullName)));

    Inc(Count);

    Writeln('  GPT-tokenized: ', FullName, '; tokens added= ', Length(OneTokens), '; total tokens = ', Length(TokenizedCorpus), '.');
  end;

  CloseFile(F);

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('GPT list tokenization complete. Tokens = ', Length(TokenizedCorpus), '.');

  MaybeSaveTokenList;
end;

// Run the user-selected Wes, UD, or GPT-2 tokenization workflow.
procedure DoTokenize;
var
  TokChoice, SourceChoice: string;
begin
  Writeln;
  Writeln('--- Tokenize ---');
  Writeln('W = Wes tokenizer');
  Writeln('U = Universal Dependencies tokenizer');
  Writeln('G = GPT-2 tokenizer');
  Writeln;

  TokChoice := AskChoice('Tokenizer: W = Wes, U = Universal Dependencies, G = GPT-2', 'W/U/G');

  case TokChoice of
    'W': begin
      SetTokenizerMode(WesTokenizer);
      TokenizeWithWes;
    end;

    'U': begin
      SetTokenizerMode(UDTokenizer);
      TokenizeWithUD;
    end;

    'G': begin
      SetTokenizerMode(GPT2Tokenizer);
      EnsureGPT2VocabLoaded;

      SourceChoice := AskChoice('Corpus source: F = one file, L = list of corpus file names', 'F/L');

      if SourceChoice = 'L' then
        TokenizeGPTFileList
      else
        TokenizeGPTSingleFile;
    end;

    else begin
      Writeln('Invalid tokenizer selection.');
      Exit;
    end;
  end;

  if Length(TokenizedCorpus) > 0 then begin
    // Writeln('Tokenizer kind before training: ', TokenizerKindName(TokenizerKind));

    if AskYesNo('Proceed to training now?', True) then begin
      NewModel := True;
      ParamsNeedCopyToDevice := True;

      SetWorkIdentity(WorkingDir, CurrentBaseName);

      WriteInfoLog(CurrentBaseName);
      RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

      if TrainSuccess then
        MaybeSaveModel;

      if TrainSuccess and AskYesNo('Proceed to inference?', True) then
        RunInfer(WModelParams, WModelState, WAdamWState);
    end;
  end;
end;

{ Training, inference, and model workflows }
// Train a new model or resume training from user-selected files.
procedure DoTrain;
var
  ModelChoice: string;
begin
  Writeln;
  Writeln('--- Train ---');
  Writeln('Required: token list and matching symbol table. You may start a new model or resume from a saved model.');

  if Length(TokenizedCorpus) = 0 then begin
    if not LoadTokenListPrompt then
      Exit;
  end
  else if AskYesNo('Use token list already in memory?', True) then begin
    Writeln('Using in-memory token list. Tokens = ', Length(TokenizedCorpus));
  end
  else
    if not LoadTokenListPrompt then Exit;

  if Length(SymbolTable) < MinSymbols then
    if not LoadSymbolTablePrompt then Exit
  else begin
    nSymbols := Length(SymbolTable);
    nVocab := nSymbols;
    Writeln('Using symbol table already in memory. Symbols = ', nSymbols);
  end;

  ModelChoice := AskChoice('Model: N = new model, R = resume/load saved model', 'N/R');

  if ModelChoice = 'R' then begin
    if not LoadModelPrompt then Exit;

    if nVocab <> nSymbols then begin
      Writeln('Warning: loaded model nVocab = ', nVocab, ' but current symbol table nSymbols = ', nSymbols, '.');
      Writeln('For resumed training these should match.');

      if not AskYesNo('Continue anyway?', False) then Exit;
    end;
  end
  else begin
    NewModel := True;
    ParamsNeedCopyToDevice := True;
    nVocab := nSymbols;
  end;

  SetWorkIdentity(WorkingDir, CurrentBaseName);

  WriteInfoLog(CurrentBaseName);
  RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

  if TrainSuccess then begin
    MaybeSaveModel;

    if AskYesNo('Proceed to inference?', False) then
      RunInfer(WModelParams, WModelState, WAdamWState);
  end;
end;

// Load the files needed for inference and run inference.
procedure DoInfer;
begin
  Writeln;
  Writeln('--- Infer ---');
  Writeln('Required: model and matching symbol table.');
  Writeln;

  // Establish tokenizer and symbol-table information first.
  if not LoadSymbolTablePrompt then Exit;

  // LoadModel restores the saved model nVocab.
  if not LoadModelPrompt then Exit;

  if nVocab <> nSymbols then begin
    Writeln('Vocabulary mismatch. Inference aborted.');
    Writeln('Model nVocab       = ', nVocab);
    Writeln('Symbol table count = ', nSymbols);
    Exit;
  end;

  ParamsNeedCopyToDevice := True;
  RunInfer(WModelParams, WModelState, WAdamWState);
end;


 // Continue training an existing model using a corpus or corpus file list.
procedure DoEnhanceModel;
var
  TokChoice, SourceChoice: string;
  SelectedTokenizer: TTokenizerKind;
begin
  Writeln;
  Writeln('--- Enhance Existing Model ---');
  Writeln('Continue training an existing model using new corpus data.');
  Writeln;

  TokChoice := AskChoice('Tokenizer: W = Wes, U = UD, G = GPT-2', 'W/U/G');

  case TokChoice of
    'W': SelectedTokenizer := WesTokenizer;
    'U': SelectedTokenizer := UDTokenizer;
    'G': SelectedTokenizer := GPT2Tokenizer;
    else Exit;
  end;

  SetTokenizerMode(SelectedTokenizer);

  // Load the existing vocabulary for Wes or UD.
  if SelectedTokenizer in [WesTokenizer, UDTokenizer] then begin
    if not LoadSymbolTablePrompt then Exit;

    // The selected tokenizer is authoritative.
    SetTokenizerMode(SelectedTokenizer);
    FromSymbolTable := True;
    ResetWesTrie;
  end
  else begin
    EnsureGPT2VocabLoaded;
    SetLength(SymbolTable, 0);
    nSymbols := 0;
    FromSymbolTable := False;
  end;

  // Load the existing model and optimizer state.
  if not LoadModelPrompt then Exit;

  // Validate vocabulary compatibility.
  if SelectedTokenizer = GPT2Tokenizer then begin
    if nVocab <> GPT2ModelVocabSize then begin
      Writeln('GPT-2 vocabulary mismatch. Enhancement aborted.');
      Exit;
    end;
  end
  else begin
    if nVocab <> Length(SymbolTable) then begin
      Writeln('Vocabulary mismatch. Enhancement aborted.');
      Writeln('Model nVocab = ', nVocab);
      Writeln('Symbol count = ', Length(SymbolTable));
      Exit;
    end;
  end;

  if nVocab > DimVocab then begin
    Writeln('Model vocabulary exceeds DimVocab.');
    Exit;
  end;

  SourceChoice := AskChoice('Corpus source: F = one file, L = list of files', 'F/L');

  if not ((SourceChoice = 'F') or (SourceChoice = 'L')) then Exit;

  SetLength(TokenizedCorpus, 0);
  SetLength(Corpus, 0);

  case SelectedTokenizer of

    WesTokenizer, UDTokenizer: begin

      FromSymbolTable := True;
      ResetWesTrie;

      if SourceChoice = 'F' then begin
        if not ReadCorpusFilePrompt(CorpusFileName, Corpus) then Exit;

        RunWesTokenizeNoAutoSave(Corpus, TokenizedCorpus);

        RawTokenCount := Length(TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);
        PaddedTokenCount := Length(TokenizedCorpus);
        nTokenizedCorpus := PaddedTokenCount;
      end
      else
        TokenizeWesFileList(ListFile, TokenizedCorpus);

    end;

    GPT2Tokenizer: begin

      SetTokenizerMode(GPT2Tokenizer);

      if SourceChoice = 'F' then
        TokenizeGPTSingleFile
      else
        TokenizeGPTFileList;

      // Both existing GPT-2 routines pad their output.
      // Count unpadded tokens using the active PAD value.
      RawTokenCount := Length(TokenizedCorpus);

      while (RawTokenCount > 0) and
            (TokenizedCorpus[RawTokenCount - 1] = PAD) do
        Dec(RawTokenCount);

      PaddedTokenCount := Length(TokenizedCorpus);
      nTokenizedCorpus := PaddedTokenCount;
    end;

  end;

  if RawTokenCount < MinTokens then begin
    Writeln('Too few tokens. Enhancement aborted.');
    Exit;
  end;

  Writeln;
  Writeln('New training data prepared.');
  Writeln('Tokenizer = ', TokenizerKindName(SelectedTokenizer));
  Writeln('Raw tokens = ', RawTokenCount);
  Writeln('Padded tokens = ', PaddedTokenCount);
  Writeln('Vocabulary = ', nVocab);

  if AskYesNo('Save new token list?', False) then
    MaybeSaveTokenList;

  // Preserve the existing model weights.
  NewModel := False;
  ParamsNeedCopyToDevice := True;

  // Establish filenames for the new training data.
  SetWorkIdentity(WorkingDir, CurrentBaseName);
  WriteInfoLog(CurrentBaseName);

  // Enhancement starts a new optimizer/training run.
  // Keep this call if that is the intended behavior.
  ResetTrainingStateForEnhancement(WAdamWState);

  if not AskYesNo('Proceed to training?', True) then Exit;

  RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

  if TrainSuccess then begin
    MaybeSaveModel;

    if AskYesNo('Proceed to inference?', False) then
      RunInfer(WModelParams, WModelState, WAdamWState);
  end;
end;

// Combine symbol tables and save the resulting table.
procedure DoJoinSymbolTables;
var
  S, DefaultBase: string;
  CombinedKind: TTokenizerKind;
begin
  MergeSymbolTables(CombinedSymbolTable);

  if Length(CombinedSymbolTable) = 0 then begin
    Writeln('No combined symbol table was created.');
    Exit;
  end;

  // Determine the tokenizer kind from the fixed symbol-table layout.
  if (Length(CombinedSymbolTable) >= UDTagBoundary) and (CombinedSymbolTable[TokNoun] = '|noun') and
     (CombinedSymbolTable[TokVerb] = '|verb') and (CombinedSymbolTable[TokVerbPart] = '|participle') then begin
    CombinedKind := UDTokenizer;
    DefaultBase := 'joined_ud';
  end
  else begin
    CombinedKind := WesTokenizer;
    DefaultBase := 'joined_w';
  end;

  SetTokenizerMode(CombinedKind);

  Write('Output combined symbol table name, blank for ', DefaultBase, '.sym: ');
  Readln(S);
  S := Trim(S);

  if S = '' then
    S := DefaultBase
  else begin
    // Remove a supplied .sym extension before adding the tokenizer suffix.
    S := ChangeFileExt(S, '');

    // Make sure the filename identifies the tokenizer kind.
    case CombinedKind of
      WesTokenizer:
        if not EndsText('_w', S) and not EndsText('_wes', S) then
          S := S + '_w';

      UDTokenizer:
        if not EndsText('_ud', S) then
          S := S + '_ud';
    end;
  end;

  SymbolFileName := MakeOutputFileName(S, SymbolDir, DefaultBase, '.sym');

  SaveSymbolTable(SymbolFileName, CombinedSymbolTable);

  SymbolTable := Copy(CombinedSymbolTable);
  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;
  SymbolTablePresent := True;

  Writeln('Combined symbol table saved: ', SymbolFileName);
  Writeln('Symbols = ', nSymbols);
end;

{ File and folder utilities }
// Display the active work folders.
procedure ShowWorkFolders;
begin
  Writeln;
  Writeln('--- Current Work Folders ---');
  Writeln('ExistingWorkRoot = ', ExistingWorkRoot);
  Writeln('WorkingDir       = ', WorkingDir);
  Writeln('WorkRoot         = ', WorkRoot);
  Writeln('CorpusDir        = ', CorpusDir);
  Writeln('SymbolDir        = ', SymbolDir);
  Writeln('MergeDir         = ', MergeDir);
  Writeln('TokenDir         = ', TokenDir);
  Writeln('ModelDir         = ', ModelDir);
  Writeln('LogDir           = ', LogDir);
  Writeln('ListDir          = ', ListDir);
  Writeln('ScratchDir       = ', ScratchDir);
end;

// Count ordinary files in a directory.
function CountFilesInDir(const DirName: string): Integer;
var
  SR: TSearchRec;
  Path: string;
begin
  Result := 0;

  if Trim(DirName) = '' then Exit;

  Path := IncludeTrailingPathDelimiter(DirName);

  if SysUtils.FindFirst(Path + '*.*', faAnyFile, SR) = 0 then begin
    try
      repeat
        if (SR.Name <> '.') and (SR.Name <> '..') then
          if (SR.Attr and faDirectory) = 0 then
            Inc(Result);
      until SysUtils.FindNext(SR) <> 0;
    finally
      SysUtils.FindClose(SR);
    end;
  end;
end;

// Display file counts for the active work folders.
procedure ShowWorkFolderFileCounts;
begin
  Writeln;
  Writeln('--- Work Folder File Counts ---');
  Writeln('CorpusDir  : ', CountFilesInDir(CorpusDir));
  Writeln('SymbolDir  : ', CountFilesInDir(SymbolDir));
  Writeln('MergeDir   : ', CountFilesInDir(MergeDir));
  Writeln('TokenDir   : ', CountFilesInDir(TokenDir));
  Writeln('ModelDir   : ', CountFilesInDir(ModelDir));
  Writeln('LogDir     : ', CountFilesInDir(LogDir));
  Writeln('ListDir    : ', CountFilesInDir(ListDir));
  Writeln('ScratchDir : ', CountFilesInDir(ScratchDir));
end;

// Display the paths of the active work subfolders.
procedure ListWorkFolderSubfolders;
begin
  Writeln;
  Writeln('--- Work Folder Subfolders ---');

  Writeln('corpus  : ', CorpusDir);
  Writeln('symbols : ', SymbolDir);
  Writeln('merges  : ', MergeDir);
  Writeln('tokens  : ', TokenDir);
  Writeln('models  : ', ModelDir);
  Writeln('logs    : ', LogDir);
  Writeln('lists   : ', ListDir);
  Writeln('scratch : ', ScratchDir);
end;

// Change the root used to locate predefined work folders.
procedure ChangeExistingWorkRoot;
var
  NewDir, OldDir: string;
begin
  Writeln;
  Writeln('Current predefined-work root: ', ExistingWorkRoot);
  Write('Enter new predefined-work root, blank to cancel: ');
  Readln(NewDir);

  NewDir := Trim(NewDir);

  if NewDir = '' then begin
    Writeln('Predefined-work root unchanged.');
    Exit;
  end;

  OldDir := ExistingWorkRoot;
  ExistingWorkRoot := NewDir;
  NormalizeExistingWorkRoot;

  if not DirectoryExists(ExistingWorkRoot) then begin
    if AskYesNo('Folder does not exist. Create it?', True) then begin
      if not ForceDirectories(ExistingWorkRoot) then begin
        Writeln('Unable to create folder: ', ExistingWorkRoot);
        ExistingWorkRoot := OldDir;
        Exit;
      end;
    end
    else begin
      ExistingWorkRoot := OldDir;
      Writeln('Predefined-work root unchanged.');
      Exit;
    end;
  end;

  Writeln('Predefined-work root changed to: ', ExistingWorkRoot);
  Writeln('The current active work folder remains: ', WorkRoot);
end;

// Change the current work folder.
procedure ChangeWorkFolder;
var
  NewDir: string;
begin
  Writeln;
  Write('Enter new work folder, blank to cancel: ');
  Readln(NewDir);

  NewDir := Trim(NewDir);

  if NewDir = '' then begin
    Writeln('Work folder unchanged.');
    Exit;
  end;

  WorkingDir := ResolveWorkFolder(NewDir);
  InitWorkFolders(WorkingDir);

  Writeln('Work folder changed.');
  ShowWorkFolders;
end;

// Display and process the file-and-folder utilities menu.
procedure DoFolderUtilities;
var
  Choice: string;
begin
  repeat
    Writeln;
    Writeln('--- File/Folder Utilities ---');
    Writeln('C: Change current work folder');
    Writeln('E: Change existing/predefined work root');
    Writeln('L: List work folder subfolders');
    Writeln('S: Show current folder settings');
    Writeln('D: Show file counts in work folders');
    Writeln('O: Open work folder in Explorer');
    Writeln('X: Return to main menu');
    Writeln;

    Choice := AskChoice('Folder command', 'C/E/L/S/D/O/X');

    case Choice of
      'C': ChangeWorkFolder;
      'E': ChangeExistingWorkRoot;
      'L': ListWorkFolderSubfolders;
      'S': ShowWorkFolders;
      'D': ShowWorkFolderFileCounts;
      'O': OpenWorkFolderInExplorer;
    end;

    if Choice <> 'X' then Pause;

  until Choice = 'X';
end;


{ Predefined model workflows }
// Prepare and train the predefined Bela Wes model.
procedure DoBelaModel;
begin
  Writeln;
  Writeln('--- Bela Corpus ---');

  nCorpus := 1036;
  SelectExistingWork('belaw', 'belaw');
  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  CorpusFileName := ResolveInputFile('bela.txt', CorpusDir);
  SymbolFileName := ResolveInputFile('bela.sym', SymbolDir);

  if not FileExists(CorpusFileName) then begin
    Writeln('File not found: ', CorpusFileName);
    Exit;
  end;

  if not FileExists(SymbolFileName) then begin
    Writeln('File not found: ', SymbolFileName);
    Exit;
  end;

  ReadFileBytes(CorpusFileName, Corpus);
  nCorpus := Length(Corpus);
  CorpusID := ComputeCorpusID(Corpus);

  FromSymbolTable := True;

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] := CorpusFileName;

  LoadSymbolTable(SymbolFileName, SymbolTable);
  ResetWesTrie;

  nSymbols := Length(SymbolTable);
  SetTokenizerMode(WesTokenizer);

  NewModel := True;
  ParamsNeedCopyToDevice := True;

  RunWesTokenizeNoAutoSave(Corpus, TokenizedCorpus);

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);
  TokenizedCorpusPresent := True;
  SymbolTablePresent := True;
  ModelPresent := False;

  Writeln('Bela tokenization complete. Tokens = ', Length(TokenizedCorpus), '; Symbols = ', nSymbols, '.');

  if AskYesNo('Save refreshed Bela token list?', False) then
    SaveCurrentTokenListDefault;

  if AskYesNo('Proceed to training?', True) then begin
    WorkingName := CurrentBaseName;
    WriteInfoLog(CurrentBaseName);
    RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

    if TrainSuccess then begin
      MaybeSaveModel;

      if AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState, WAdamWState);
    end;
  end;
end;

// Prepare and train the predefined Damned Thing Wes model.
procedure DoDamnedThingModel;
begin
  Writeln;
  Writeln('--- Damned Thing Corpus ---');

  nCorpus := 18457;
  SelectExistingWork('dt904', 'dt904');
  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  TokenFileName := ResolveInputFile('damnedthing.tok', TokenDir);
  SymbolFileName := ResolveInputFile('damnedthing.sym', SymbolDir);

  if not FileExists(TokenFileName) then begin
    Writeln('File not found: ', TokenFileName);
    Exit;
  end;

  if not FileExists(SymbolFileName) then begin
    Writeln('File not found: ', SymbolFileName);
    Exit;
  end;

  SetTokenizerMode(WesTokenizer);

  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);
  SetLoadedTokenCounts(TokenizedCorpus);

  TokenizedCorpusPresent := True;
  SymbolTablePresent := True;
  ModelPresent := False;

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);
  ResetWesTrie;

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  NewModel := True;
  ParamsNeedCopyToDevice := True;

  Writeln('Damned Thing data loaded. Tokens = ', Length(TokenizedCorpus), '; Symbols = ', nSymbols, '; Corpus size = ', nCorpus, '.');

  if AskYesNo('Proceed to training?', True) then begin
    WorkingName := CurrentBaseName;
    WriteInfoLog(CurrentBaseName);
    RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

    if TrainSuccess then begin
      MaybeSaveModel;

      if AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState, WAdamWState);
    end;
  end;
end;

// Resume a predefined Wes or UD best model.
procedure DoResumeBestWesTModel(const ModelTitle, FolderName, DataBaseName: string; const ATokenizerKind: TTokenizerKind);
begin
  Writeln;
  Writeln('--- ', ModelTitle, ' ---');

  if not (ATokenizerKind in [WesTokenizer, UDTokenizer]) then begin
    Writeln('Invalid tokenizer kind for Wes-style model.');
    Exit;
  end;

  SelectExistingWork(FolderName, DataBaseName);
  SetTokenizerMode(ATokenizerKind);

  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  TokenFileName := TokenDir + DataBaseName + '.tok';
  SymbolFileName := SymbolDir + DataBaseName + '.sym';
  ModelFileName := ModelDir + DataBaseName + '_best.model';

  // Verify all required files before replacing in-memory data.
  if not FileExists(TokenFileName) then begin
    Writeln('Token file not found: ', TokenFileName);
    Exit;
  end;

  if not FileExists(SymbolFileName) then begin
    Writeln('Symbol table not found: ', SymbolFileName);
    Exit;
  end;

  if not FileExists(ModelFileName) then begin
    Writeln('Model file not found: ', ModelFileName);
    Exit;
  end;

  // Load tokenized corpus.
  SetLength(TokenizedCorpus, 0);
  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);
  SetLoadedTokenCounts(TokenizedCorpus);

  if Length(TokenizedCorpus) < MinTokens then begin
    Writeln('Token list is too small. Length = ', Length(TokenizedCorpus), '.');
    Exit;
  end;

  // Load matching Wes symbol table.
  ResetWesTrie;
  SetLength(SymbolTable, 0);

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);

  if Length(SymbolTable) < MinSymbols then begin
    Writeln('Symbol table is too small. Length = ', Length(SymbolTable), '.');
    Exit;
  end;

  nSymbols := Length(SymbolTable);

  // Discard CUDA storage belonging to the preceding model.
  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState, WAdamWState);

  // Load saved parameters and model information.
  if not LoadModel(ModelFileName, WModelParams, WAdamWState) then begin
    Writeln('Model not loaded: ', ModelFileName);
    Exit;
  end;

  NewModel := False;
  ParamsNeedCopyToDevice := True;

  // LoadModel should have restored nVocab.
  if nVocab <> nSymbols then begin
    Writeln('Vocabulary mismatch. Training aborted.');
    Writeln('Model nVocab       = ', nVocab);
    Writeln('Symbol table count = ', nSymbols);
    Exit;
  end;

  if nVocab > DimVocab then begin
    Writeln('Model vocabulary exceeds DimVocab. Training aborted.');
    Writeln('nVocab   = ', nVocab);
    Writeln('DimVocab = ', DimVocab);
    Exit;
  end;

  TokenizedCorpusPresent := True;
  SymbolTablePresent := True;
  ModelPresent := True;

  Write('Loaded: Token list = ', TokenFileName, '; Model = ', ModelFileName);
  Writeln('; Raw tokens = ', RawTokenCount, '; Padded tokens = ', PaddedTokenCount, '; nVocab = ', nVocab, '.');

  WorkingName := CurrentBaseName;

  Writeln('Loaded checkpoint: ', ModelFileName, '.');

  if AskYesNo('Proceed to resumed training?', True) then begin

    WriteInfoLog(CurrentBaseName);

    RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

    if TrainSuccess then begin
      if AskYesNo('Save an additional model?', False) then
        MaybeSaveModel;
    end;
  end;

  if AskYesNo('Proceed to inference?', False) then
    RunInfer(WModelParams, WModelState, WAdamWState);
end;

// Resume a predefined GPT-2 best model.
procedure DoResumeBestGPT2Model(const ModelTitle, FolderName, DataBaseName: string);
begin
  Writeln;
  Writeln('--- ', ModelTitle, ' ---');

  SelectExistingWork(FolderName, DataBaseName);

  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  TokenFileName := TokenDir + DataBaseName + '.tok';
  ModelFileName := ModelDir + DataBaseName + '_best.model';

  // Verify all required files before replacing in-memory data.
  if not FileExists(TokenFileName) then begin
    Writeln('Token file not found: ', TokenFileName);
    Exit;
  end;

  if not FileExists(ModelFileName) then begin
    Writeln('Model file not found: ', ModelFileName);
    Exit;
  end;

  // Establish GPT-2 token values, especially PAD, before counting tokens.
  SetTokenizerMode(GPT2Tokenizer);
  EnsureGPT2VocabLoaded;

  // Load tokenized corpus.
  SetLength(TokenizedCorpus, 0);
  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);
  SetLoadedTokenCounts(TokenizedCorpus);

  if Length(TokenizedCorpus) < MinTokens then begin
    Writeln('Token list is too small. Length = ', Length(TokenizedCorpus), '.');
    Exit;
  end;

  // GPT-2 does not require a Wes symbol table.
  ResetWesTrie;
  SetLength(SymbolTable, 0);
  nSymbols := 0;
  FromSymbolTable := False;

  // Discard CUDA storage belonging to the preceding model.
  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState, WAdamWState);

  // Load saved parameters and checkpoint information.
  if not LoadModel(ModelFileName, WModelParams, WAdamWState) then begin
    Writeln('Model not loaded: ', ModelFileName);
    Exit;
  end;

  NewModel := False;
  ParamsNeedCopyToDevice := True;

  // LoadModel restores the model's nVocab.
  if nVocab > DimVocab then begin
    Writeln('Model vocabulary exceeds DimVocab. Training aborted.');
    Writeln('nVocab   = ', nVocab);
    Writeln('DimVocab = ', DimVocab);
    Exit;
  end;

  TokenizedCorpusPresent := True;
  SymbolTablePresent := False;
  ModelPresent := True;

  Write('Loaded: Token list = ', TokenFileName, '; Model = ', ModelFileName);
  Writeln('; Raw tokens = ', RawTokenCount, '; Padded tokens = ', PaddedTokenCount, '; nVocab = ', nVocab, '.');

  WorkingName := CurrentBaseName;

  Writeln('Loaded checkpoint: ', ModelFileName, '.');

  if AskYesNo('Proceed to resumed training?', True) then begin

    WriteInfoLog(CurrentBaseName);

    RunTrain(WModelParams, WModelState, WAdamWState, TokenizedCorpus);

    if TrainSuccess then begin
      if AskYesNo('Save an additional model?', False) then
        MaybeSaveModel;
    end;
  end;

  if AskYesNo('Proceed to inference?', False) then
    RunInfer(WModelParams, WModelState, WAdamWState);
end;

// Resume the predefined Gibbon 924 WesT model.
procedure DoGibbon924WModel;
begin
  DoResumeBestWesTModel('Gibbon 924 Best Model', 'gibbon924w', 'gibbon', WesTokenizer);
end;

// Resume the predefined Gibbon 924 GPT model.
procedure DoGibbon924GModel;
begin
  DoResumeBestGPT2Model('Gibbon 924 Best Model', 'gibbon924g', 'gibbon');
end;

// Resume the predefined Gibbon 924 WesT model.
procedure DoGibbon924UModel;
begin
  DoResumeBestWesTModel('Gibbon 924 Best Model', 'gibbon924u', 'gibbon', UDTokenizer);
end;

// Resume the predefined Churchill UD model.
procedure DoChurchill925UModel;
begin
  DoResumeBestWesTModel('Churchill 925 Best Model', 'churchill925u', 'churchill', UDTokenizer);
end;

// Resume the predefined Churchill Wes model.
procedure DoChurchill1006WModel;
begin
  DoResumeBestWesTModel('Churchill 1006 Best Model', 'churchill1006w', 'churchill', WesTokenizer);
end;

// Resume the predefined The Monkey''s Paw Wes model.
procedure DoTMP928WModel;
begin
  DoResumeBestWesTModel('The Monkey''s Paw 928 Best Model', 'TMP928w', 'TMP', WesTokenizer);
end;

// Resume the predefined Tragedy UD model.
procedure DoTragedy924UModel;
begin
  DoResumeBestWesTModel('Tragedy 924 Best Model', 'tragedy924u', 'tragedy', UDTokenizer);
end;

// Resume the predefined Tiny Stories 2 MB Wes model.
procedure DoTS2UModel;
begin
  DoResumeBestWesTModel('Tiny Stories 2MB Wes 104 Best Model', 'ts2mb104w', 'tinystories_2mb', WesTokenizer);
end;

// Resume the predefined Tiny Stories 5 MB UD model.
procedure DoTS5UModel;
begin
  DoResumeBestWesTModel('Tiny Stories 5MB WesUD 925 Best Model', 'ts5mb925u', 'tinystories_5mb_ud', UDTokenizer);
end;

// Resume the predefined Tiny Stories 10 MB Wes model.
procedure DoTS2WModel;
begin
  DoResumeBestWesTModel('Tiny Stories 2MB WesT 104 Best Model', 'ts2mb104w', 'tinystories_2mb', WesTokenizer);
end;

// Resume the predefined Tiny Stories 10 MB Wes model.
procedure DoTS10WModel;
begin
  DoResumeBestWesTModel('Tiny Stories 10MB WesT 102 Best Model', 'ts10mb102w', 'tinystories_10mb', WesTokenizer);
end;

// Resume the predefined Tiny Stories 60 MB Wes model.
procedure DoTS60WModel;
begin
  DoResumeBestWesTModel('Tiny Stories 60MB WesT 925 Best Model', 'ts60mb925w', 'tinystories_60mb', WesTokenizer);
end;

// Resume the predefined Tiny Stories 100 MB Wes model.
procedure DoTS100WModel;
begin
  DoResumeBestWesTModel('Tiny Stories 100MB WesT 927 Best Model', 'ts100mb927w', 'tinystories_100mb', WesTokenizer);
end;

// Resume the predefined Tiny Stories 20 MB GPT-2 model.
procedure DoTS20GModel;
begin
  DoResumeBestGPT2Model('Tiny Stories 20MB GPT2 930 Best Model', 'ts20mb930g', 'tinystories_20mb');
end;

// Resume the predefined Tiny Stories 20 MB WesT SplitSwiGLU model.
procedure DoTS20WModel;
begin
  DoResumeBestWesTModel('Tiny Stories 20MB WesT SplitSwiGLU 1014 Best Model', 'ts20mb1014wss', 'tinystories_20mb', WesTokenizer);
end;

// Resume the predefined Tiny Stories 20 MB WesUD model.
procedure DoTS20UDModel;
begin
  DoResumeBestGPT2Model('Tiny Stories 20MB WesUD 923 Best Model', 'ts20mb923u', 'tinystories_20mb');
end;

// Resume the predefined Tiny Stories 100 MB GPT-2 model.
procedure DoTS100GModel;
begin
  DoResumeBestGPT2Model('Tiny Stories 100MB GPT2 1005 Best Model', 'ts100mb1005g', 'tinystories_100mb');
end;

// Display and process the predefined-model menu.
procedure DoPredefinedWork;
var
  TChoice: string;
begin
  repeat
    Writeln('--- Predefined Corpora and Models ---');
    Writeln('B: Train a new model, using an existing symbol table with WesT on the Bela corpus.');
    Writeln('D: Train a new model with WesT on the Damned Thing token list and symbol table.');
    Writeln('GW: Resume the Gibbon 924 WesT best model.');
    Writeln('GU: Resume the Gibbon 924 UD best model.');
    Writeln('GG: Resume the Gibbon 924 GPT2 best model.');
    Writeln('CHU: Resume the Churchill UD best model.');
    Writeln('CHW: Resume the Churchill WesT best model.');
    Writeln('TMP: Resume the The Monkey''s Paw WesT best model.');
    Writeln('TRU: Resume the Tragedy UD best model.');
    Writeln('TS2W: Resume the Tiny Stories 2MB Wes model.');
    Writeln('TS5U: Resume the Tiny Stories 5MB WesUD model.');
    Writeln('TS10W: Resume the Tiny Stories 10MB WesT model.');
    Writeln('TS20G: Resume the Tiny Stories 20MB GPT-2 model.');
    Writeln('TS20W: Resume the Tiny Stories 20MB WesT SplitSwiGLU model.');
    Writeln('TS20U: Resume the Tiny Stories 20MB WesUD model.');
    Writeln('TS60W: Resume the Tiny Stories 60MB WesT model.');
    Writeln('TS100W: Resume the Tiny Stories 100MB WesT model.');
    Writeln('TS100G: Resume the Tiny Stories 100MB GPT-2 model.');
    Writeln('X: Return to main menu');

    TChoice := AskChoice('Selection', 'B/D/GW/GU/GG/CHU/CHW/TMP/TRU/TS2W/TS5U/TS10W/TS20G/TS20W/TS20U/TS60W/TS100G/X');
    case TChoice of
      'B', 'BELA': DoBelaModel;
      'D', 'DT': DoDamnedThingModel;
      'GW': DoGibbon924WModel;
      'GU': DoGibbon924UModel;
      'GG': DoGibbon924GModel;
      'CHU': DoChurchill925UModel;
      'CHW': DoChurchill1006WModel;
      'TMP': DoTMP928WModel;
      'TRU': DoTragedy924UModel;
      'TS2U': DoTS2UModel;
      'TS2W': DoTS2WModel;
      'TS5U': DoTS5UModel;
      'TS20U': DoTS20UDModel;
      'TS10W': DoTS10WModel;
      'TS20G': DoTS20GModel;
      'TS20W': DoTS20WModel;
      'TS60W': DoTS60WModel;
      'TS100W': DoTS100WModel;
      'TS100G': DoTS100GModel;
    end;
  until TChoice = 'X';
end;


{ Menus and optional commands }
// Display the main user menu.
procedure Options;
begin
  Writeln('Options:');
  Writeln('  T: Tokenize -- create a token list from a corpus or corpus file list.');
  Writeln('     Select WesTokenizer, UDTokenizer, or GPT2Tokenizer. Wes and UD may create a symbol table or use an existing one.');
  Writeln('  R: Train -- train a model on a token list.');
  Writeln('     Requires a token list and matching symbol table. Can start a new model or resume from a saved model.');
  Writeln('  I: Infer -- run inference.');
  Writeln('     Requires a saved model and matching symbol table.');
  Writeln('  J: Join symbol tables.');
  Writeln('     Requires two symbol tables.');
  Writeln('  M: Do predefined work -- train corpora or resume predefined Wes, UD, and GPT-2 models.');
  Writeln('  E: Enhance -- continue training an existing model from a corpus or list of corpora.');
  Writeln('     Uses the existing model vocabulary and preserves trained weights.');
  //Writeln('  E: Enhance -- continue training an existing model on a new corpus using its existing symbol table.');
  Writeln('  F: File/folder utilities.     P: Program information.     H: Help and options.     X: Exit.');
end;

// Display help, folder information, and optional commands.
procedure Help;
begin
  Options;

  Writeln('Work folder layout:');
  Writeln('  ', CorpusDir,  '   corpus input files');
  Writeln('  ', SymbolDir,  '   .sym files');
  Writeln('  ', MergeDir,   '   .mer files');
  Writeln('  ', TokenDir,   '   .tok files');
  Writeln('  ', ModelDir,   '   saved models');
  Writeln('  ', LogDir,     '   logs');
  Writeln('  ', ListDir,    '   file lists');
  Writeln;

  Writeln('Debug / display toggles available:');
  Writeln('  VTO / NVTO: VerboseTokenize on/off');
  Writeln('  DC / NDC:   DisplayCorpus on/off');
  Writeln('  DTW / NDTW: DisplayTokenWork on/off');
  Writeln('  DMW / NDMW: DisplayMergeWork on/off');
  Writeln('  DTV / NDTV:   DisplayTokenVerification on/off');
  Writeln('  DEBR / NDEBR: DisplayEachByteRead on/off');
  Writeln('  DW / NDW: DisplayWindow on/off');
  Writeln('  DNP / DP: DoNotPause on/off');
  Writeln('  SF / NSF: SaveFiles on/off');
  Writeln;
  Writeln('Training display:');
  Writeln('  DE:   Display epochs');
  Writeln('  DS:   Display stages');
  Writeln('  DSS:  Display substages');
  Writeln('  ND:   Reduce display');
  Writeln;
  Writeln('Parameters:');
  Writeln('  MM:   Maximum merges');
  Writeln('  PC:   Maximum pair count');
  Writeln('  LR:   Override learning rate');
  Writeln('  TEMP: Temperature');
  Writeln;
  Writeln('Normalization kind:');
  Writeln('  LN:   Layer Norm');
  Writeln('  RMS:  RMS Norm');
  Writeln;
  Writeln('Tokenizer kind:');
  Writeln('  UTOK:    Universal Dependencies');
  Writeln('  WTOK:    Wes');
  Writeln('  CTOK:    GPT-2');
  Writeln;
  Writeln('Gating kind:');
  Writeln('  NOG:     No gating');
  Writeln('  SELF:    Self gating');
  Writeln('  SSG:     Split SwiGLU (Activation = SiLU');
  Writeln;
  Writeln('Activation kind:');
  Writeln('  ReLU:    Rectified linear unit');
  Writeln('  GELU:    Gaussian error function linear unit');
  Writeln('  SiLU:    Sigmoid linear unit');
  Writeln('  LReLU:   Leaky rectified linear unit');
  Writeln('  ELU:     Error function linear unit');
  Writeln('  SPLUS:   Smoothed function linear unit');
  Writeln('  MISH:    Smoothed non-monotonic function linear unit');
end;

// Process optional display, training, tokenization, and parameter commands.
procedure HandleSettingCommand(const Cmd: string);
begin
  case Cmd of
    'VTO': begin
      VerboseTokenize := True;
      Writeln('Verbose tokenize is ', VerboseTokenize);
    end;
    'NVTO': begin
      VerboseTokenize := False;
      Writeln('Verbose tokenize is ', VerboseTokenize);
    end;
    'DC': begin
      DisplayCorpus := True;
      Writeln('Display corpus is ', DisplayCorpus);
    end;
    'NDC': begin
      DisplayCorpus := False;
      Writeln('Display corpus is ', DisplayCorpus);
    end;
    'DTW': begin
      DisplayTokenWork := True;
      Writeln('Display token work is ', DisplayTokenWork);
    end;
    'NDTW': begin
      DisplayTokenWork := False;
      Writeln('Display token work is ', DisplayTokenWork);
    end;
    'DMW': begin
      DisplayMergeWork := True;
      Writeln('Display merge work is ', DisplayMergeWork);
    end;
    'NDMW': begin
      DisplayMergeWork := False;
      Writeln('Display merge work is ', DisplayMergeWork);
    end;
    'DTV': begin
      DisplayTokenVerification := True;
      Writeln('Display token verification is ', DisplayTokenVerification);
    end;
    'NDTV': begin
      DisplayTokenVerification := False;
      Writeln('Display token verification is ', DisplayTokenVerification);
    end;
    'DEBR': begin
      DisplayEachByteRead := True;
      Writeln('Display each byte read is ', DisplayEachByteRead);
    end;
    'NDEBR': begin
      DisplayEachByteRead := False;
      Writeln('Display each byte read is ', DisplayEachByteRead);
    end;
    'VTR': begin
      VerboseTransform := True;
      DisplayStage := True;
      Writeln('Verbose transform is ', VerboseTransform);
    end;
    'NVTR': begin
      VerboseTransform := False;
      Writeln('Verbose transform is ', VerboseTransform);
    end;
    'DE': begin
      DisplayEpoch := True;
      DisplayStage := False;
      DisplaySubstage := False;
    end;
    'VI': begin
      VerboseInfer := True;
      Writeln('Verbose infer is ', VerboseInfer);
    end;
    'NVI': begin
      VerboseInfer := False;
      Writeln('Verbose infer is ', VerboseInfer);
    end;
    'DS': begin
      DisplayStage := True;
      DisplayEpoch := True;
      DisplaySubstage := False;
    end;
    'DSS': begin
      DisplaySubstage := True;
      DisplayStage := True;
      DisplayEpoch := True;
    end;
    'ND': begin
      DisplayEpoch := False;
      DisplayStage := False;
      DisplaySubstage := False;
    end;
    'DW': begin
      DisplayWindow := True;
      Writeln('Display window is ', DisplayWindow);
    end;
    'NDW': begin
      DisplayWindow := False;
      Writeln('Display window is ', DisplayWindow);
    end;
    'DNP': begin
      DoNotPause := True;
      Writeln('Do not pause is ', DoNotPause);
    end;
    'DP': begin
      DoNotPause := False;
      Writeln('Do not pause is ', DoNotPause);
    end;
    'SF': begin
      SaveFiles := True;
      Writeln('Save files is ', SaveFiles);
    end;
    'NSF': begin
      SaveFiles := False;
      Writeln('Save files is ', SaveFiles);
    end;
    'TEMP': begin
      Write('Inference temperature: ');
      Readln(ITemperature);
    end;
    'LR': begin
      Write('Override learning rate: ');
      Readln(OverrideLearningRate);
    end;
    'MM': begin
      Write('Maximum merges: ');
      Readln(MaxMerges);
    end;
    'PC': begin
      Write('Maximum pair count: ');
      Readln(MaxPairCount);
    end;
    'LN': begin
      if NormKind = RMSNorm then
        ReinitializeBeta(WModelParams, WAdamWState);  // Zero betas if changing to Layer Norm.
      NormKind := LayerNorm;
      Writeln('Normalization is ', NormKindName(NormKind));
    end;
    'RMS': begin
      NormKind := RMSNorm;
      Writeln('Normalization is ', NormKindName(NormKind));
    end;
    'UTOK': begin
      TokenizerKind := UDTokenizer;
      Writeln('Tokenizer is ', TokenizerKindName(TokenizerKind));
    end;
    'WTOK': begin
      TokenizerKind := WesTokenizer;
      Writeln('Tokenizer is ', TokenizerKindName(TokenizerKind));
    end;
    'CTOK': begin
      TokenizerKind := GPT2Tokenizer;
      Writeln('Tokenizer is ', TokenizerKindName(TokenizerKind));
    end;
    'NOG': begin
      MLPGatingKind := NoGating;
      Writeln('Gating function is ', MLPGatingKindName(MLPGatingKind));
    end;
    'SELF': begin
      MLPGatingKind := SelfGating;
      Writeln('Gating function is ', MLPGatingKindName(MLPGatingKind));
    end;
    'SSG': begin
      MLPGatingKind := SplitSwiGLU;
      Writeln('Gating function is ', MLPGatingKindName(MLPGatingKind));
    end;
    'RELU': begin
      ActivationKind := ReLU;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
    'GELU': begin
      ActivationKind := GELU;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
    'SILU': begin
      ActivationKind := SiLU;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
    'LRELU': begin
      ActivationKind := LeakyReLU;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
    'ELU': begin
      ActivationKind := ELU;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
    'SPLUS': begin
      ActivationKind := SoftPlus;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
    'MISH': begin
      ActivationKind := Mish;
      Writeln('Activation function is ', ActivationKindName(ActivationKind));
    end;
  end;
end;

{ Main program }
begin
  // Configure UTF-8 console input and output.
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  // Display the program identity.
  Writeln('WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com.');
  Writeln;

  // Initialize all the program subfolders.
  InitializeWesChatFolders;

  // Display key program variables used by the menus.
  ReportStartupKeyVariables;
  if YesToAll then
    Writeln('The program is set to respond automatically to all yes/no selections with yes.');
  Writeln;

  // Establish the root for work folders and ask for the active work folder.
  NormalizeExistingWorkRoot;
  Writeln('Work folders are created under ', ExistingWorkRoot); // Do not use YesToAll here.
  Write('Enter work folder name, blank for WesChatWork: ');
  Readln(WorkingDir);

  // Resolve and initialize the active work folder.
  WorkingDir := ResolveWorkFolder(WorkingDir);
  InitWorkFolders(WorkingDir);
  // Writeln('Work folder: ', WorkRoot);

  // Display the main menu.
  Options;

  // Process commands until the user exits.
  try
    while True do begin
      Write('W>');
      Readln(Ch);
      Ch := UpperCase(Trim(Ch));

      case Ch of
        'T': DoTokenize;
        'R': DoTrain;
        'I': DoInfer;
        'J': DoJoinSymbolTables;
        'M': DoPredefinedWork;
        'E': DoEnhanceModel;
        'F': DoFolderUtilities;
        'P': ReportProgramInfo;
        'H': Help;
        'X', 'EXIT': Break;

        'VTO', 'NVTO', 'DC', 'NDC', 'DTW', 'NDTW',
        'DMW', 'NDMW', 'DTV', 'NDTV', 'DEBR', 'NDEBR',
        'VTR', 'NVTR', 'VI', 'NVI',
        'DE', 'DS', 'DSS', 'ND',
        'DW', 'NDW', 'DNP', 'DP', 'SF', 'NSF',
        'TEMP', 'LR', 'MM', 'PC',
        'LN', 'RMS',
        'WTOK', 'UTOK', 'CTOK',
        'NOG','SELF','SSG',
        'RELU', 'GELU', 'SILU', 'LRELU', 'ELU', 'SPLUS', 'MISH':
          HandleSettingCommand(Ch);

        else
          Writeln('Invalid input. Enter H for help.');
      end;
    end;

  except
    on E: Exception do begin
      Writeln;
      Writeln('Fatal error: ', E.ClassName, ': ', E.Message);
      Writeln;
      HardPause;
    end;
  end;

  // Release CUDA resources before exit.
  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState, WAdamWState);

  // Release the GPT-2 vocabulary object before exit.
  if Assigned(Vocab) then
    Vocab.Free;
end.
