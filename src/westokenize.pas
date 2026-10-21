unit WesTokenize;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  Classes, Crt, DateUtils, FileUtil, Math, SysUtils,
  { WesChat units }
  Display, Global;

type
  { Token count types }
  TTokenCount = record
    Symbol: Integer;                                    // Token ID.
    Count: Integer;                                     // Number of occurrences.
  end;
  TTokenCounts = array of TTokenCount;

  { Trie types }
  PTrieNode = ^TTrieNode;
  TTrieNode = record
    Children: array[0..255] of PTrieNode;               // One child for each possible byte.
    TokenID: Integer;                                   // -1 when the node is not terminal.
  end;

  { Merged-token statistics types }
  TMergedTokenStat = record
    TokenID: Integer;
    Count: Integer;
  end;
  TMergedTokenStats = array of TMergedTokenStat;

const
  TokenProgressInterval = 1000;

var
  { Timing }
  ElapsedMS, Hours, Mins: Int64;
  Secs, MSecs: Double;

  { Tokenizer state }
  TrieHead: PTrieNode = nil;
  MergedTypes, UnmergedTypes: Integer;
  MergedInstances, UnmergedInstances: Integer;

  { Legacy/shared state }
  FileName: String;
  Magic: array[0..3] of Char = ('S', 'Y', 'M', 'T');
  i: Integer;

{ Public routines }

// Free the cached trie so it will be rebuilt from the current symbol table.
procedure ResetWesTrie;

// Build a trie from the current symbol table.
procedure BuildTrie(out Root: PTrieNode);

// Match the longest symbol in the trie at a corpus position.
function MatchLongest(root: PTrieNode; const text: TBVector; startPos: Integer; out tokenID, matchLen: Integer): Boolean;

// Tokenize corpus bytes with the current Wes or UD symbol table.
procedure TokenizeWesBytes(const Corpus: TBVector; var Tokens: TIVector);

// Run Wes tokenization and optional reporting.
procedure RunWesTokenize(const Corpus: TBVector; var TokenizedCorpus: TIVector);

// Report tokenization statistics.
procedure DisplayTokenProgress(const TokenCount, Position, Total: Integer);
procedure ReportStatistics(const TokenizedCorpus: TIVector);

// Display part or all of a token list.
procedure WriteTokenList(const TokenizedCorpus: TIVector; const Part: TPart = B);

// Save tokenization statistics to a log file.
procedure SaveTokenizationLog(const TokenizedCorpus: TIVector; const LogFileName: string);

implementation

{ Trie construction and matching }
// Insert one symbol into the trie.
procedure InsertTrieSymbol(root: PTrieNode; const s: string; id: Integer);
var
  node: PTrieNode;
  i: Integer;
  c: Byte;
begin
  node := root;

  for i := 1 to Length(s) do begin
    c := Ord(s[i]);

    if node^.Children[c] = nil then begin
      New(node^.Children[c]);
      FillChar(node^.Children[c]^, SizeOf(TTrieNode), 0);
      node^.Children[c]^.TokenID := -1;
    end;

    node := node^.Children[c];
  end;

  node^.TokenID := id;                                  // Mark this node as a complete symbol.
end;

// Build a trie from the current symbol table.
procedure BuildTrie(out Root: PTrieNode);
var
  i: Integer;
begin
  New(Root);
  FillChar(Root^, SizeOf(TTrieNode), 0);
  Root^.TokenID := -1;

  for i := 0 to High(SymbolTable) do begin
    if (i = BOS) or (i = EOS) or (i = PAD) or (i = UNK) then Continue;

    if SymbolTable[i] = '' then Continue;

    InsertTrieSymbol(Root, SymbolTable[i], i);
  end;
end;

// Free a trie and all of its child nodes.
procedure FreeTrie(var Node: PTrieNode);
var
  i: Integer;
begin
  if Node = nil then
    Exit;

  for i := 0 to 255 do
    FreeTrie(Node^.Children[i]);

  Dispose(Node);
  Node := nil;
end;

// Free the cached trie after the symbol table changes.
procedure ResetWesTrie;
begin
  FreeTrie(TrieHead);
end;

// Ensure that the cached trie exists.
procedure EnsureWesTrie;
begin
  if TrieHead = nil then
    BuildTrie(TrieHead);
end;

// Match the longest symbol beginning at a corpus position.
function MatchLongest(root: PTrieNode; const text: TBVector; startPos: Integer; out tokenID, matchLen: Integer): Boolean;
var
  node: PTrieNode;
  i: Integer;
  c: Byte;
  lastMatchID: Integer;
  lastMatchLen: Integer;
begin
  node := root;
  lastMatchID := -1;
  lastMatchLen := 0;

  i := startPos;

  while i < Length(text) do begin
    c := text[i];

    if node^.Children[c] = nil then Break;

    node := node^.Children[c];

    if node^.TokenID <> -1 then begin
      lastMatchID := node^.TokenID;
      lastMatchLen := i - startPos + 1;
    end;

    Inc(i);
  end;

  if lastMatchID <> -1 then begin
    tokenID := lastMatchID;
    matchLen := lastMatchLen;
    Result := True;
  end
  else
    Result := False;
end;

{ UD token support }
// Return the first learned merged-token ID for the current Wes-style tokenizer.
function CurrentFirstMergedToken: Integer;
begin
  if TokenizerKind = UDTokenizer then
    Result := UDTagBoundary
  else
    Result := UNK + 1;
end;

// Match a reserved UD tag such as |noun, |pl, or |past.
function MatchReservedUDTag(root: PTrieNode; const text: TBVector; startPos: Integer; out tokenID, matchLen: Integer): Boolean;
var
  Node: PTrieNode;
  i: Integer;
  c: Byte;
  LastMatchID, LastMatchLen: Integer;
begin
  Node := root;
  LastMatchID := -1;
  LastMatchLen := 0;
  i := startPos;

  while i < Length(text) do begin
    c := text[i];

    if Node^.Children[c] = nil then Break;

    Node := Node^.Children[c];

    // Record only reserved UD-tag terminals.
    if (Node^.TokenID >= FirstTagToken) and (Node^.TokenID < UDTagBoundary) then begin
      LastMatchID := Node^.TokenID;
      LastMatchLen := i - startPos + 1;
    end;

    Inc(i);
  end;

  if LastMatchID >= 0 then begin
    tokenID := LastMatchID;
    matchLen := LastMatchLen;
    Result := True;
  end
  else begin
    tokenID := -1;
    matchLen := 0;
    Result := False;
  end;
end;

// Return the UD text beginning at a corpus position for diagnostics.
function UDTextAtPosition(const Corpus: TBVector; StartPos: Integer): string;
var
  i: Integer;
begin
  Result := '';

  if (StartPos < 0) or (StartPos > High(Corpus)) then Exit;

  i := StartPos;

  // First byte should be '|'.
  if Corpus[i] = Ord('|') then begin
    Result := '|';
    Inc(i);
  end;

  // Tag ends at another | or whitespace.
  while (i <= High(Corpus)) and (Corpus[i] <> Ord('|')) and (Corpus[i] <> 9) and
        (Corpus[i] <> 10) and (Corpus[i] <> 13) and (Corpus[i] <> 32) do begin
    Result := Result + Chr(Corpus[i]);
    Inc(i);
  end;
end;

{ Core tokenization }

// Tokenize a corpus using the currently loaded symbol table.
procedure TokenizeFromSymbolTable(var TokenizedCorpus: TIVector; const Corpus: TBVector);
var
  i, BestSym, BestLen: Integer;
begin
  SetLength(TokenizedCorpus, 1);
  TokenizedCorpus[0] := BOS;
  i := 0;

  EnsureWesTrie;

  while i < Length(Corpus) do begin

    // Tiny Stories UTF-8 separator "■" becomes EOS.
    if (i + 2 < Length(Corpus)) and (Corpus[i] = $E2) and
       (Corpus[i + 1] = $96) and (Corpus[i + 2] = $A0) then begin
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := EOS;
      Inc(i, 3);
      Continue;
    end;

    if MatchLongest(TrieHead, Corpus, i, BestSym, BestLen) then begin
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := BestSym;
      Inc(i, BestLen);
    end
    else begin
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := Corpus[i];
      Inc(i);
    end;
  end;

  // Ensure corpus ends with EOS.
  if (Length(TokenizedCorpus) = 0) or
     (TokenizedCorpus[High(TokenizedCorpus)] <> EOS) then begin
    SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
    TokenizedCorpus[High(TokenizedCorpus)] := EOS;
  end;

  nTokenizedCorpus := Length(TokenizedCorpus);

  FreeTrie(TrieHead);

  if VerboseTokenize then begin
    Write('Input corpus (first ', DisplayLength, ' bytes) is: ');
    for i := 0 to Min(High(Corpus), DisplayLength) do
      Write(Corpus[i], ' ');
    Writeln;
  end;

  Writeln('Created ', nTokenizedCorpus, ' tokens.');

  if VerboseTokenize then begin
    Write('First 50 tokens of tokenized corpus: ');
    for i := 0 to Min(49, (nTokenizedCorpus - 1)) do
      Write(TokenizedCorpus[i], ' ');
    Writeln;
    Pause;
  end;
end;

// Tokenize corpus bytes with the current Wes or UD symbol table.
procedure TokenizeWesBytes(const Corpus: TBVector; var Tokens: TIVector);
var
  i, BestSym, BestLen: Integer;

  procedure AddToken(const Token: Integer);
  begin
    SetLength(Tokens, Length(Tokens) + 1);
    Tokens[High(Tokens)] := Token;

    if (Length(Tokens) mod TokenProgressInterval) = 0 then
      if not VerboseTokenize then
        DisplayTokenProgress(Length(Tokens), i, Length(Corpus));
  end;

begin
  EnsureWesTrie;

  if (TokenizerKind = UDTokenizer) and (Length(SymbolTable) < UDTagBoundary) then
    raise Exception.CreateFmt('TokenizerKind is UDTokenizer, but SymbolTable has only %d entries; at least %d are required.', [Length(SymbolTable), UDTagBoundary]);

  SetLength(Tokens, 1);
  Tokens[0] := BOS;
  i := 0;

  while i < Length(Corpus) do begin
    // Tiny Stories separator byte becomes EOS.
    if Corpus[i] = 254 then begin
      AddToken(EOS);
      Inc(i);
      Continue;
    end;

    // In UD mode, first test whether a vertical bar begins a reserved grammatical tag.
    if (TokenizerKind = UDTokenizer) and (Corpus[i] = Ord('|')) then begin
      if MatchReservedUDTag(TrieHead, Corpus, i, BestSym, BestLen) then begin
        AddToken(BestSym);
        Inc(i, BestLen);
        Continue;
      end;
    // An unrecognized vertical bar falls through to ordinary longest-symbol matching.
    end;

    // Apply ordinary Wes longest-symbol matching.
    if MatchLongest(TrieHead, Corpus, i, BestSym, BestLen) then begin
      AddToken(BestSym);
      Inc(i, BestLen);
    end
    else begin
      // Every raw byte 0..255 should normally exist in SymbolTable, but retain the byte fallback.
      AddToken(Corpus[i]);
      Inc(i);
    end;
  end;

  // Add EOS to the end, and display progress.
  AddToken(EOS);
  if not VerboseTokenize then
    DisplayTokenProgress(Length(Tokens), Length(Corpus), Length(Corpus));
  Writeln;
end;

{ Statistics and reporting }
// Calculate time statistics.
procedure CalculateTimeStatistics;
var
  RawMS, PauseMS: Int64;
begin
  RawMS := MilliSecondsBetween(t0, t1);
  PauseMS := Round(StopTime * 86400000.0);

  ElapsedMS := RawMS - PauseMS;

  if ElapsedMS <= 0 then
    ElapsedMS := 1;

  Hours := ElapsedMS div 3600000;
  Mins := (ElapsedMS mod 3600000) div 60000;
  Secs := (ElapsedMS mod 60000) / 1000.0;
end;

// Count the number of occurrences of each symbol.
procedure CountSymbols(const TokenizedCorpus: TIVector);
var
  Counts, Index: TIVector;
  i, j, k, n, TmpIndex: Integer;
begin
  // Allocate and zero Counts.
  SetLength(Counts, Length(SymbolTable));
  FillChar(Counts[0], Length(Counts) * SizeOf(Counts[0]), 0);

  // Count occurrences.
  for i := 0 to High(TokenizedCorpus) do
    Inc(Counts[TokenizedCorpus[i]]);

  // Build index array.
  n := Length(Counts);
  SetLength(Index, n);
  for i := 0 to n - 1 do
    Index[i] := i;

  // Selection sort index array by Counts[index] descending.
  for i := 0 to n - 2 do begin
    k := i;
    for j := i + 1 to n - 1 do
      if Counts[Index[j]] > Counts[Index[k]] then
        k := j;

    // Swap Index[i] and Index[k].
    TmpIndex := Index[i];
    Index[i] := Index[k];
    Index[k] := TmpIndex;
  end;

  // Print top 60.
  Writeln('Top 60 most frequent symbols: ');
  for i := 0 to 59 do begin
    k := Index[i];
    if Counts[k] > 0 then begin
      Write(i + 1: 8, ': Symbol ', k: 8, '  Count=', Counts[k]: 6, '  ', '"' + CleanUpSymbol(SymbolTable[k]) + '"': 15);
      if ((i + 1) mod 3) = 0 then
        Writeln;
    end;
  end;

  if VerboseTokenize and (TextRec(Output).Handle = StdOutputHandle) then Pause;
end;

// Display tokenization progress for Wes and GPT-2.
procedure DisplayTokenProgress(const TokenCount, Position, Total: Integer);
var
  Percent: Double;
begin
  if Total > 0 then
    Percent := 100.0 * Position / Total
  else
    Percent := 0.0;

  Write(#13);
  ClrEol;
  Write(TokenCount, ' tokens created; ', Percent:0:1, '% of input processed.');
  Flush(Output);
end;

// Report all statistics.
procedure ReportStatistics(const TokenizedCorpus: TIVector);

// Calculate the number of merged and unmerged symbol types and token instances.
procedure CalculateSymbolCount;
var
  i, T, MergeStart: Integer;
begin
  MergeStart := CurrentFirstMergedToken;

  MergedTypes := 0;
  UnmergedTypes := 0;

  // Count symbol types.
  for i := 0 to High(SymbolTable) do
    if i >= MergeStart then
      Inc(MergedTypes)
    else
      Inc(UnmergedTypes);

  // Count token instances.
  MergedInstances := 0;
  UnmergedInstances := 0;

  for i := 0 to High(TokenizedCorpus) do begin
    T := TokenizedCorpus[i];

    if T >= MergeStart then
      Inc(MergedInstances)
    else
      Inc(UnmergedInstances);
  end;
end;

// Count usage of every token ID.
procedure CountTokenUsage(const TokenizedCorpus: TIVector; nSymbols: Integer; var Counts: TIVector);
var
  i, t: Integer;
begin
  SetLength(Counts, nSymbols);

  for i := 0 to nSymbols - 1 do
    Counts[i] := 0;

  for i := 0 to High(TokenizedCorpus) do begin
    t := TokenizedCorpus[i];
    if (t >= 0) and (t < nSymbols) then
      Inc(Counts[t]);
  end;
end;

// Build a list of merged-token statistics.
procedure BuildMergedTokenStats(const Counts: TIVector; FirstMergedID: Integer; out Stats: TMergedTokenStats);
var
  i, k: Integer;
begin
  SetLength(Stats, 0);
  k := 0;

  for i := FirstMergedID to High(Counts) do begin
    SetLength(Stats, k + 1);
    Stats[k].TokenID := i;
    Stats[k].Count := Counts[i];
    Inc(k);
  end;
end;

// Sort merged-token statistics by descending count.
procedure SortMergedTokenStatsByCount(var Stats: TMergedTokenStats);
var
  i, j: Integer;
  Temp: TMergedTokenStat;
begin
  for i := 0 to High(Stats) - 1 do
    for j := i + 1 to High(Stats) do
      if Stats[j].Count > Stats[i].Count then begin
        Temp := Stats[i];
        Stats[i] := Stats[j];
        Stats[j] := Temp;
      end;
end;

{ Token usage reports }
// Report the N most frequent merged tokens.
procedure ReportTopMergedTokens(const Stats: TMergedTokenStats; N: Integer);
var
  i, Limit: Integer;
  S: String;
begin
  Writeln('--- Top ', N, ' Most Frequent Merged Tokens ---');

  if Length(Stats) = 0 then Exit;

  Limit := N;
  if Limit > Length(Stats) then
    Limit := Length(Stats);

  for i := 0 to Limit - 1 do begin
    S := CleanUpSymbol(SymbolTable[Stats[i].TokenID]);
    if Stats[i].Count > 0 then
      Writeln(i + 1:4, '  ID=', Stats[i].TokenID:6, '  Count=', Stats[i].Count:8, '  Symbol="', S, '"');
  end;
   if TextRec(Output).Handle = StdOutputHandle then Pause;
end;

// Report merged tokens that are never used.
procedure ReportUnusedMergedTokens(const Stats: TMergedTokenStats);
var
  i, Unused: Integer;
begin
  Unused := 0;

  for i := 0 to High(Stats) do
    if Stats[i].Count = 0 then
      Inc(Unused);

  Writeln('Merged tokens never used: ', Unused);
end;

// Report merged tokens that are used only once.
procedure ReportSingletonMergedTokens(const Stats: TMergedTokenStats);
var
  i, Singletons: Integer;
begin
  Singletons := 0;

  for i := 0 to High(Stats) do
    if Stats[i].Count = 1 then
      Inc(Singletons);

  Writeln('Merged tokens used only once: ', Singletons);
end;

// Report coverage contributed by the most frequent merged tokens.
procedure ReportTopMergeCoverage(const Stats: TMergedTokenStats; TopN: Integer);
var
  i, Limit: Integer;
  TotalMergedInstances, TopMergedInstances: Integer;
  Coverage: Single;
begin
  TotalMergedInstances := 0;
  for i := 0 to High(Stats) do
    Inc(TotalMergedInstances, Stats[i].Count);

  Limit := TopN;
  if Limit > Length(Stats) then
    Limit := Length(Stats);

  TopMergedInstances := 0;
  for i := 0 to Limit - 1 do
    Inc(TopMergedInstances, Stats[i].Count);

  if TotalMergedInstances > 0 then
    Coverage := 100.0 * TopMergedInstances / TotalMergedInstances
  else
    Coverage := 0.0;

  Writeln('Top ', Limit, ' merged tokens account for ', TopMergedInstances, ' / ', TotalMergedInstances,
    ' merged-token instances = ', Coverage:0:2, '%');
  Writeln;
end;

// Report token usage statistics.
procedure ReportTokenUsageStatistics;
var
  Counts: TIVector;
  Stats: TMergedTokenStats;
begin
  Writeln('--- Token Statistics ---');
  Writeln('Merged token instances: ', MergedInstances);
  Writeln('Unmerged token instances: ', UnmergedInstances);
  Writeln('Mean token length: ', nCorpus / nTokenizedCorpus: 6: 4);
  Writeln;
  CountTokenUsage(TokenizedCorpus, Length(SymbolTable), Counts);
  BuildMergedTokenStats(Counts, CurrentFirstMergedToken, Stats);
  SortMergedTokenStatsByCount(Stats);

  ReportTopMergedTokens(Stats, 30);
  ReportUnusedMergedTokens(Stats);
  ReportSingletonMergedTokens(Stats);
  ReportTopMergeCoverage(Stats, 30);
end;

// Report basic timing and file statistics.
procedure ReportBasicStatistics;
var
  i: Integer;
begin
  Writeln;
  Writeln('--- File Information ---');
  Writeln('Files used in symbol table: ');
  for i := 0 to High(CorpusFileNames) do
    Writeln(CorpusFileNames[i], '  ');
  Writeln;

  Writeln('--- Time Statistics ---');
  Writeln('Start time: ', DateTimetoStr(t0), '     End time: ', DateTimeToStr(t1));
  Writeln('Total elapsed time: ', Hours, ' hours, ', Mins, ' min ', Secs: 4: 4, ' sec');
  Writeln;
end;

// Report BPE statistics.
procedure ReportBPEStatistics;
begin
  Writeln('--- Tokenization Statistics ---');
  Writeln('Original corpus size: ', nCorpus, ' bytes.');
  Writeln('Encoded token count: ', nTokenizedCorpus);
  Writeln('Bytes per token: ', nCorpus / nTokenizedCorpus:0:4);
  if not FromSymbolTable then
    Writeln('Input bytes per second: ', nCorpus / (ElapsedMS / 1000):6:4);
  Writeln;
end;

begin
  CalculateTimeStatistics;
  CalculateSymbolCount;
  ReportBasicStatistics;
  if VerboseTokenize and (TextRec(Output).Handle = StdOutputHandle) then
    Pause;
  ReportBPEStatistics;
  ReportTokenUsageStatistics;
end;

{ Display and saving }

// Display the requested part of a tokenized corpus.
procedure WriteTokenList(const TokenizedCorpus: TIVector; const Part: TPart = B);
var
  i, iB, iE: Integer;
begin
  case Part of
    B: begin
      iB := 0;
      iE := Min(99, High(TokenizedCorpus));
    end;

    E: begin
      iB := Max(0, High(TokenizedCorpus) - 99);
      iE := High(TokenizedCorpus);
    end;

    F: begin
      iB := 0;
      iE := High(TokenizedCorpus);
    end;
  end;

  Write('Tokenized corpus, ');
  case Part of
    B: Write('First 100 tokens: ');
    E: Write('Last 100 tokens: ');
    F: Write('All tokens: ');
  end;
  Writeln;

  for i := iB to iE do
    Write(TokenizedCorpus[i], ' ');
  Writeln;
  Writeln('Tokenized corpus length = ', Length(TokenizedCorpus));
  Pause;
end;

// Save tokenization statistics to a log file.
procedure SaveTokenizationLog(const TokenizedCorpus: TIVector; const LogFileName: string);
var
  SaveOut: Text;
  OutName, OutDir: string;
  Redirected: Boolean;
begin
  OutName := Trim(LogFileName);

  if OutName = '' then begin
    if Trim(WorkingName) <> '' then
      OutName := WorkingName + '.log'
    else
      OutName := 'tokenization.log';
  end;

  if ExtractFilePath(OutName) = '' then begin
    if Trim(LogDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(LogDir) + OutName
    else
      OutName := IncludeTrailingPathDelimiter(GetCurrentDir) + OutName;
  end;

  if ExtractFileExt(OutName) = '' then
    OutName := OutName + '.log';

  OutDir := ExtractFilePath(OutName);

  try
    if Trim(OutDir) <> '' then
      ForceDirectories(OutDir);

    SaveOut := Output;
    Redirected := False;

    try
      Assign(Output, OutName);

      if FileExists(OutName) then
        Append(Output)
      else
        Rewrite(Output);

      Redirected := True;

      ReportStatistics(TokenizedCorpus);

    finally
      if Redirected then
        Close(Output);

      Output := SaveOut;
    end;

    Writeln('Tokenization log written: ', OutName);

  except
    on E: Exception do begin
      Output := SaveOut;

      Writeln('Error saving tokenization log: ', E.ClassName, ' ', E.Message);
      Writeln('Target log file = "', OutName, '"');
    end;
  end;
end;

{ Main tokenizer workflow }
// Run Wes tokenization and optional reporting.
procedure RunWesTokenize(const Corpus: TBVector; var TokenizedCorpus: TIVector);
begin
  t0 := Now;
  StopTime := 0;

  nCorpus := Length(Corpus);

  if not Training then
    FileName := 'Inference';

  TokenizeWesBytes(Corpus, TokenizedCorpus);

  nTokenizedCorpus := Length(TokenizedCorpus);
  nSymbols := Length(SymbolTable);

  Writeln('Created ', nTokenizedCorpus, ' tokens.');

  t1 := Now;

  if DisplayTokenWork and VerboseTokenize then begin
    Writeln('--- Token Frequencies ---');
    CountSymbols(TokenizedCorpus);
  end;

  if VerboseTokenize then
    ReportStatistics(TokenizedCorpus);

  // Optionally verify tokenization by reconstructing the corpus.
  if DisplayCorpusVerification then begin
    // Writeln('--- Reconstructed Corpus, Beginning 500 Bytes ---');
    // Writeln('Length = ', Length(TokenizedCorpus));
    WesDetokenizeTextToDisplay(TokenizedCorpus, B);
    Writeln;
  end;

  if DisplayTokenVerification then begin
    Write('First ', DisplayLength, ' tokens of tokenized corpus: ');
    for i := 0 to Min(DisplayLength, High(TokenizedCorpus)) do
      Write(TokenizedCorpus[i], ' ');
    Writeln;
    Pause;
  end;
end;

finalization
  FreeTrie(TrieHead);

end.
