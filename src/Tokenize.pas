unit Tokenize;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Classes,
  Crt,
  DateUtils,
  Display,
  FileUtil,
  Global,
  SysUtils;

type
  TSymbolTable = TRBSVector;           // Array of symbols. So index of array is a symbol string.
  PTokenNode = ^TTokenNode;            // Doubly-linked list.
  TTokenNode = record                  // Each node as a token, an integer corresponding to a symbol.
    Tok: Integer;
    Prev, Next: PTokenNode;
  end;

  type
  TPairSlotState = (psEmpty, psUsed);

  TPairHashEntry = record
    A, B: Integer;
    Count: Integer;
    State: TPairSlotState;
  end;

  TPairHash = record
    Entries: array of TPairHashEntry;
    Capacity: Integer;
    Used: Integer;
  end;

  // Old pair count code.
  TPairCount = record                  // Record of pair counts.
    A, B: Integer;                     // A and B are the pair.
    Count: Integer;                    // Count is how often they occur.
  end;
  TPairCounts = array of TPairCount;   // Array of pair counts.

  TMerge = record                      // Record for merger of two nodes.
    A, B: Integer;                     // Original pair.
    NewSym: Integer;                   // New integer for symbol.
  end;
  TMergeArray = array of TMerge;       // Array of merges.

  TTokenCount = record                 // Records count of tokens.
    Symbol: Integer;                   // Symbol for token.
    Count: Integer;                    // Number of times it occurs.
  end;
  TTokenCounts = array of TTokenCount; // Array of token counts.

  PTrieNode = ^TTrieNode;
  TTrieNode = record
    Children: array[0..255] of PTrieNode; // ASCII.
    TokenID: Integer;                     // -1 if not terminal.
  end;

// Token statistics.
  TMergedTokenStat = record
    TokenID: Integer;
    Count: Integer;
  end;

  TMergedTokenStats = array of TMergedTokenStat;
var
  StartSymbol: Integer = 260;                    // UTF-8 0.255, BOS, EOS, PAD, UNK is 259.
  nCorpus: Integer;
  TokenizedCorpus: TIVector;
  ElapsedMS, MElapsedMS: Int64;                  // For timing.
  MHours, Hours, MMIns, Mins: Int64;             // For timing.
  Secs, MSecs: Double;                           // For timing.
  BOS, EOS, PAD, UNK: Integer;                   // Extra symbols for control.
  Head, Tail: PTokenNode;                        // Start and end node of list of tokens.
  MergeCount: Integer;                           // Maximum allowed number of merges and actual number.
  Merges: TMergeArray;                           // Array recording the merges.
  FileName, Reconstructed: String;               // Saving data.
  //SymbolTable: TSymbolTable;                     // Table of symbols.
  Magic: array[0..3] of Char = ('S', 'Y', 'M', 'T');  // For saving symbol table.
  TrieHead: PTrieNode = nil;                     // Nodes for Trie.
  MergedTypes, UnmergedTypes: Integer;
  MergedInstances, UnmergedInstances: Integer;
  i: Integer;

procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
procedure WriteTokenList(const Part: TPart = B);
procedure LoadTokenList(const BinFileName: String);
//procedure LoadSymbolTable(const FileName: string);
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
procedure DisplaySymbolTable;
procedure TokenizeFromSymbolTable(const TextFileName: string; var Corpus: TBVector);
procedure BuildTrie(const SymbolTable: TRBSVector; out Root: PTrieNode);
procedure ReconstructText(Head: PTokenNode; out Text: String);
function MatchLongest(root: PTrieNode; const text: TBVector; startPos: Integer;
  out tokenID, matchLen: Integer): Boolean;
procedure DetokenizeToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
procedure ReportStatistics;
procedure RunTokenize(const Corpus: TBVector);

implementation

// Apply a learned symbol table to a raw byte corpus.
// Input:
//   SymbolTable: array of learned symbols, each symbol is an array of bytes.
//   nSymbols (aka nVocab): number of entries in SymbolTable.
//   Corpus: raw byte text.
// Output:
//   TokenizedCorpus: dynamic array of token IDs.

{ PIPELINE 1: create own symbol table }
{ Load the Corpus }
// Read the corpus as a stream of binary.
procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
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
  if VeryVerbose and VerboseTokenize then
    writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VeryVerbose and VerboseTokenize then
      if ShowEachByteRead then
        if B < 32 then
          Write('<', B, '>')
        else
          Write(Chr(B));
  end;
  CloseFile(F);
  if VeryVerbose and VerboseTokenize then begin
    writeln('ReadByteFile: ');
    for i := 0 to 150 do
      write(OneCorpus[i], ' ');
    Pause;
  end;
  if VeryVerbose and VerboseTokenize then
    writeln;

  // Display initial Corpus length.
  Writeln('Read ', Size, ' bytes from ', FileName);
end;

{ Construct the token linked list }
// To prevent special characters from merging.
function IsSpecial(T: Integer): Boolean;
begin
  Result := (T = BOS) or (T = EOS) or (T = PAD) or (T = UNK);
end;

// Insert BOS at beginning of token list.
procedure InsertBOS(var Head, Tail: PTokenNode);
var
  N: PTokenNode;
begin
  New(N);
  N^.Tok := BOS;
  N^.Prev := nil;
  N^.Next := Head;

  if Head <> nil then
    Head^.Prev := N
  else
    Tail := N;   // List was empty.

  Head := N;
end;

// Insert EOS at end of token list.
procedure InsertEOS(var Head, Tail: PTokenNode);
var
  N: PTokenNode;
begin
  New(N);
  N^.Tok := EOS;
  N^.Next := nil;
  N^.Prev := Tail;

  if Tail <> nil then
    Tail^.Next := N
  else
    Head := N;   // List was empty.

  Tail := N;
end;

// Build the initial token linked list from the Corpus.
procedure BuildTokenListFromCorpus(const Corpus: TBVector);
var
  i: Integer;
  Node, Prev: PTokenNode;
begin
  Head := nil;
  Tail := nil;
  Prev := nil;

  for i := 0 to High(Corpus) do begin
    New(Node);
    Node^.Tok := Corpus[i];
    Node^.Prev := Prev;
    Node^.Next := nil;

    if Prev <> nil then
      Prev^.Next := Node
    else
      Head := Node;

    Prev := Node;
  end;

  Tail := Prev;
end;

// Init hash code.
procedure InitPairHash(var H: TPairHash; InitialCapacity: Integer);
var
  i: Integer;
begin
  if InitialCapacity < 16 then
    InitialCapacity := 16;

  H.Capacity := InitialCapacity;
  H.Used := 0;
  SetLength(H.Entries, H.Capacity);

  for i := 0 to H.Capacity - 1 do begin
    H.Entries[i].A := 0;
    H.Entries[i].B := 0;
    H.Entries[i].Count := 0;
    H.Entries[i].State := psEmpty;
  end;
end;

function HashPair(A, B, Capacity: Integer): Integer;
var
  H: QWord;
begin
  H := QWord(Cardinal(A)) * 1000003 + QWord(Cardinal(B));
  Result := Integer(H mod QWord(Capacity));
end;

function FindSlot(const H: TPairHash; A, B: Integer): Integer;
var
  Idx: Integer;
begin
  Idx := HashPair(A, B, H.Capacity);

  while H.Entries[Idx].State = psUsed do begin
    if (H.Entries[Idx].A = A) and (H.Entries[Idx].B = B) then Exit(Idx);

    Idx := (Idx + 1) mod H.Capacity;
  end;

  Result := Idx;
end;

procedure PairIncHash(var H: TPairHash; A, B: Integer);
var
  Idx: Integer;
begin
  Idx := FindSlot(H, A, B);

  if H.Entries[Idx].State = psUsed then
    Inc(H.Entries[Idx].Count)
  else begin
    H.Entries[Idx].State := psUsed;
    H.Entries[Idx].A := A;
    H.Entries[Idx].B := B;
    H.Entries[Idx].Count := 1;
    Inc(H.Used);
  end;
end;

procedure PairDecHash(var H: TPairHash; A, B: Integer);
var
  Idx: Integer;
begin
  Idx := FindSlot(H, A, B);

  if (H.Entries[Idx].State = psUsed) and
     (H.Entries[Idx].A = A) and
     (H.Entries[Idx].B = B) then
  begin
    if H.Entries[Idx].Count > 0 then
      Dec(H.Entries[Idx].Count);
  end;
end;

function FindBestPairHash(const H: TPairHash; out A, B: Integer): Integer;
var
  i, Max: Integer;
begin
  Max := 0;
  A := -1;
  B := -1;

  for i := 0 to H.Capacity - 1 do
    if (H.Entries[i].State = psUsed) and (H.Entries[i].Count > Max) then begin
      Max := H.Entries[i].Count;
      A := H.Entries[i].A;
      B := H.Entries[i].B;
    end;

  Result := Max;
end;

// Init pairs hash routine.
procedure InitPairHashFromList(Head: PTokenNode; var H: TPairHash);
var
  Cur: PTokenNode;
begin
  Cur := Head;

  while (Cur <> nil) and (Cur^.Next <> nil) do begin
    if not (IsSpecial(Cur^.Tok) or IsSpecial(Cur^.Next^.Tok)) then
      PairIncHash(H, Cur^.Tok, Cur^.Next^.Tok);

    Cur := Cur^.Next;
  end;
end;

// Update pairs hash.
procedure UpdatePairsForMergeHash(Node: PTokenNode; NewTok: Integer; var H: TPairHash);
var
  A, B: Integer;
begin
  if (Node = nil) or (Node^.Next = nil) then Exit;

  A := Node^.Tok;
  B := Node^.Next^.Tok;

  // Remove (A, B).
  PairDecHash(H, A, B);

  // Remove (Prev, A).
  if Node^.Prev <> nil then
    PairDecHash(H, Node^.Prev^.Tok, A);

  // Remove (B, Next).
  if Node^.Next^.Next <> nil then
    PairDecHash(H, B, Node^.Next^.Next^.Tok);

  // Add (Prev, NewTok).
  if Node^.Prev <> nil then
    PairIncHash(H, Node^.Prev^.Tok, NewTok);

  // Add (NewTok, Next).
  if Node^.Next^.Next <> nil then
    PairIncHash(H, NewTok, Node^.Next^.Next^.Tok);
end;

{ Merge process in linked list }
// Merge two nodes in token linked list.
procedure MergeAt(var Head, Tail: PTokenNode; Node: PTokenNode; NewTok: Integer);
var
  Right: PTokenNode;
begin
  Right := Node^.Next;
  if Right = nil then Exit;

  // If merging away the tail, update Tail.
  if Right = Tail then
    Tail := Node;

  // Replace Node + Right with NewTok.
  Node^.Tok := NewTok;
  Node^.Next := Right^.Next;

  if Right^.Next <> nil then
    Right^.Next^.Prev := Node;

  Dispose(Right);
end;

// Merge for pairs hash.
procedure MergeAllPairsHash(var Head, Tail: PTokenNode; A, B, NewTok: Integer; var H: TPairHash);
var
  Cur: PTokenNode;
begin
  Cur := Head;

  while (Cur <> nil) and (Cur^.Next <> nil) do begin
    if not (IsSpecial(Cur^.Tok) or IsSpecial(Cur^.Next^.Tok)) then begin
      if (Cur^.Tok = A) and (Cur^.Next^.Tok = B) then begin
        UpdatePairsForMergeHash(Cur, NewTok, H);
        MergeAt(Head, Tail, Cur, NewTok);
        Cur := Cur^.Next;
      end
      else
        Cur := Cur^.Next;
    end
    else
      Cur := Cur^.Next;
  end;
end;

// Check procedure for hashing.
procedure CheckPairHashCounts(const H: TPairHash);
var
  i: Integer;
begin
  for i := 0 to H.Capacity - 1 do
    if (H.Entries[i].State = psUsed) and (H.Entries[i].Count < 0) then
      Writeln('Negative pair count: (', H.Entries[i].A, ',', H.Entries[i].B,
        ') Count=', H.Entries[i].Count);
end;

// Record the merge in the Merges array.
procedure RecordMerge(var Merges: TMergeArray; MergeIndex, A, B, NewSym: Integer);
begin
  if MergeIndex >= Length(Merges) then
    SetLength(Merges, MergeIndex + 1);

  Merges[MergeIndex].A := A;
  Merges[MergeIndex].B := B;
  Merges[MergeIndex].NewSym := NewSym;
end;

{ Symbol Table }
// Initialize the symbol table with special characters.
procedure InitSymbolTable;
var
  i: Integer;
begin
  // 0..255 = bytes.
  SetLength(SymbolTable, 256);
  for i := 0 to 255 do
    SymbolTable[i] := Chr(i);

  // Add BOS. 256.
  BOS := Length(SymbolTable);
  SetLength(SymbolTable, BOS + 1);
  SymbolTable[BOS] := '<BOS>';

  // Add EOS. 257.
  EOS := Length(SymbolTable);
  SetLength(SymbolTable, EOS + 1);
  SymbolTable[EOS] := '<EOS>';

  // Add PAD. 258.
  PAD := Length(SymbolTable);
  SetLength(SymbolTable, PAD + 1);
  SymbolTable[PAD] := '<PAD>';

  // Add UNK. 259.
  UNK := Length(SymbolTable);
  SetLength(SymbolTable, UNK + 1);
  SymbolTable[UNK] := '<UNK>';
end;

// After performing a merge, add a new merge symbol to the symbol table.
procedure AddMergeSymbol(NewTok, A, B: Integer);
begin

  if (A < 0) or (A >= Length(SymbolTable)) then
    writeln('Invalid symbol A=', A);

  if (B < 0) or (B >= Length(SymbolTable)) then
    writeln('Invalid symbol B=', B);

  // Debugging.
  // Writeln('AddMergeSymbol: NewTok=', NewTok, ' A=', A, ' B=', B, ' Len=', Length(Table));

  // Ensure the table is large enough.
  if NewTok >= Length(SymbolTable) then
    SetLength(SymbolTable, NewTok + 1);

  // Represent the new token as concatenation of its components.
  if Length(SymbolTable[A]) + Length(SymbolTable[B]) < 4096 then
    SymbolTable[NewTok] := SymbolTable[A] + SymbolTable[B]
  else
    SymbolTable[NewTok] := '';  // lazy expansion

end;

{ PIPELINE 2: Use existing symbol table }
// Load the symbol table from file.
procedure LoadSymbolTable(const FileName: string);
var
  F: file;
  Magic: array[0..3] of Char;
  i, Len: Integer;
  S: string;
begin
  Assign(F, FileName);
  Reset(F, 1);

  // Magic header.
  BlockRead(F, Magic, SizeOf(Magic));
  if (Magic[0] <> 'S') or (Magic[1] <> 'Y') or
     (Magic[2] <> 'M') or (Magic[3] <> 'T') then begin
    Close(F);
    writeln('Invalid symbol table file.');
    Pause;
    Exit;
  end;

  // Version.
  BlockRead(F, Version, 16);

  // Symbol count.
  BlockRead(F, nSymbols, SizeOf(nSymbols));
  SetLength(SymbolTable, NSymbols);

  // Special token IDs
  BlockRead(F, BOS, SizeOf(BOS));
  BlockRead(F, EOS, SizeOf(EOS));
  BlockRead(F, PAD, SizeOf(PAD));
  BlockRead(F, UNK, SizeOf(UNK));

  // Read symbols.
  for i := 0 to nSymbols - 1 do begin
    BlockRead(F, Len, SizeOf(Len));
    SetLength(S, Len);
    if Len > 0 then
      BlockRead(F, S[1], Len);
    SymbolTable[i] := S;
  end;

  Close(F);
  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;
  Writeln('Loaded ', nSymbols, ' symbols from ', FileName);
end;

// Trie procedures.
// Call Once.
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

  node^.TokenID := id;  // mark terminal
end;

// Build trie.
procedure BuildTrie(const SymbolTable: TRBSVector; out Root: PTrieNode);
var
  i: Integer;
begin
  New(Root);
  FillChar(Root^, SizeOf(TTrieNode), 0);
  Root^.TokenID := -1;

  for i := 0 to High(SymbolTable) do begin
    // Skip special tokens 256..259.
    if (i = BOS) or (i = EOS) or (i = PAD) or (i = UNK) then Continue;
    if SymbolTable[i] = '' then Continue;

    InsertTrieSymbol(Root, SymbolTable[i], i);
  end;
end;

function MatchLongest(root: PTrieNode;
  const text: TBVector;
  startPos: Integer;
  out tokenID,
  matchLen: Integer): Boolean;
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

  while (i < Length(text)) do  begin
    c := text[i];

    if node^.Children[c] = nil then break;

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

// Tokenize Corpus from SymbolTable loaded by program.
procedure TokenizeFromSymbolTable(const TextFileName: string; var Corpus: TBVector);
var
  i, BestSym, BestLen: Integer;
begin
  if FileExists(TextFileName) then
    ReadFileBytes(TextFileName, Corpus);

  SetLength(TokenizedCorpus, 0);

  i := 0;

  BuildTrie(SymbolTable, TrieHead);

  while i < nCorpus do begin
    if MatchLongest(TrieHead, Corpus, i, BestSym, BestLen) then begin
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := BestSym;
      Inc(i, BestLen);
    end
    else begin
      // fallback: single byte token
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := Corpus[i];
      Inc(i);
    end;
  end;

  nTokenizedCorpus := Length(TokenizedCorpus);

  writeln('Created ', nTokenizedCorpus, ' tokens from ', TextFileName);
  // Verify by reconstructing.
  if ShowVerification and VerboseTokenize and DisplayCorpus then begin
    writeln('--- Reconstructed Corpus ---');
    ReconstructText(Head, Reconstructed);
    Writeln(Reconstructed);
  end;

  if VerboseTokenize then Begin
    writeln('First 50 token of tokenized corpus');
    for i := 0 to 49 do
      write(TokenizedCorpus[i], ' ');
    writeln;
    Pause;
  end;

  nVocab := nSymbols;
  writeln('End of tokenization. Press <CR> to continue.');
end;

{ Apply the BPE encoder }
// Main training loop, traverse the merges.
procedure TrainBPEHash(var Head, Tail: PTokenNode; MaxMerges: Integer;
  var MergeCount, StartSymbol: Integer);
var
  m, BestCount, A, B: Integer;
  H: TPairHash;

  procedure ReadMergeIfKeyPressed;
  var
    key: Char;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X':
        begin
          Writeln('Exit requested. Stopping execution.');
          Pause;
          Halt;
        end;
      'b', 'B':
        begin
          Writeln('Break requested. Exiting loop.');
          Pause;
          BestCount := 0;   // Causes outer loop to stop.
        end;
      'v', 'V':
        begin
          VeryVerbose := not VeryVerbose;
          Writeln('Very verbose mode: ', VeryVerbose);
        end;
      'i', 'I':
        begin
          Writeln;
          ReportInfo;
          Pause;
        end;
      'p', 'P':
        begin
          Pause;
        end;
      'm', 'M':
        begin
          Writeln;
          Writeln('Maximum symbols = ', MaxVocab, '. Maximum merges = ', MaxMerges,
            '. Hash capacity = ', H.Capacity, '. Used slots = ', H.Used, '. Best count = ', BestCount, '.');
          Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
          Writeln('  P = Program information. M = Merging information. S = maximum Symbols. Merging...');
        end;
      's', 'S':
        begin
          Writeln;
          Write('Current maximum symbols = ', MaxVocab, '. Enter new maximum symbols: ');
          ReadLn(MaxVocab);
        end;
    end;
  end;

begin
  MergeCount := 0;

  Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
  Writeln('  P = Program information. M = Merging information. S = maximum Symbols. Merging...');
  Writeln;

  if ShowMergeWork then
    Writeln('--- List of Merges (Hash) ---');

  for m := 1 to MaxMerges do begin
    if PauseIfKeyPressed then
      ReadMergeIfKeyPressed;

    // Rebuild pair counts from current token list.
    InitPairHash(H, MaxPairCount * 2 + 1024);
    InitPairHashFromList(Head, H);

    // Optional: save partial symbol table.
    if SavePartialSymbolTable then
      if (Length(SymbolTable) mod PartialSymbolTableTrigger) = 0 then
        SaveSymbolTable(WorkingName + FormatDateTime('yyyy-mm-dd_hhnnss' + '.sym', Now), SymbolTable);

    // Stop if hash table got too full.
    if H.Used > MaxPairCount then begin
      Writeln('Stopping: pair table exceeded ', MaxPairCount, ' entries.');
      Break;
    end;

    BestCount := FindBestPairHash(H, A, B);

    // Stop if no useful merges remain.
    if BestCount < 2 then begin
      Writeln('Stopping: no more valid merges at iteration ', m, '.');
      Break;
    end;

    // Stop if symbol table is full.
    if Length(SymbolTable) >= MaxVocab then begin
      Writeln('Stopping: symbol table reached ', MaxVocab, ' entries.');
      Break;
    end;

    // Perform merge.
    MergeAllPairsHash(Head, Tail, A, B, StartSymbol, H);

    AddMergeSymbol(StartSymbol, A, B);
    RecordMerge(Merges, MergeCount, A, B, StartSymbol);

    Inc(MergeCount);
    Inc(StartSymbol);

    if ShowMergeWork then begin
      Write(MergeCount, ' Merged (', A:5, ',', B:5, ') -> (', StartSymbol - 1:5, ') #', BestCount);
      if (MergeCount mod 4) = 0 then
        Writeln
      else
        Write('  |  ');
    end;
  end;

  Writeln('Hash tokenization complete. Total merges: ', MergeCount, '.');
  Pause;
end;

{ Create the tokenized corpus from linked list }
// Create tokenized corpus.
procedure CreateTokenizedCorpus;
var
  Cur: PTokenNode;
  i: Integer;
begin
  i := 0;
  Cur := Head;
  SetLength(TokenizedCorpus, nTokenizedCorpus);
  while Cur <> nil do begin
    TokenizedCorpus[i] := Cur^.Tok;
    Inc(i);
    Cur := Cur^.Next;
  end;
end;

{ Display routines }
// Display the list of tokens from linked list and count them.
procedure DisplayCountTokenList(Head: PTokenNode; var k: Integer);
begin
  k := 0;
  // k counts the number of loops and thus the token number.
  if ShowTokenWork and VerboseTokenize then
    Writeln('--- Token List ---');

  while Head <> nil do begin      // Loop thru the nodes.
    if ShowTokenWork and VerboseTokenize then
      Write(Head^.Tok, ' ');      // Write each Head for the Token.
    Inc(k);
    Head := Head^.Next;
  end;
  if ShowTokenWork and VerboseTokenize then begin
    Writeln;
    writeln('Token list length = ', k);      // Write the total number of nodes.
    Pause;
  end;
end;

// Display the symbol table.
procedure DisplaySymbolTable;
var
  i, j: Integer;
  Disp: String;
begin
  Writeln('--- Symbol Table ---');           // Chr(183) is non-display char.
  for i := 0 to High(SymbolTable) do         // Loop thru each symbol in table.
    if SymbolTable[i] <> '' then begin
      Disp := SymbolTable[i];                // Use Disp so Table is not changed.
      for j := 1 to Length(SymbolTable[i]) do
        if (Ord(Disp[j]) < 32) or (Ord(Disp[j]) = 127) then Disp[j] := Chr(183);
      if Length(Disp) < 12 then begin
        write(i: 8, Disp: 15);
        if (i mod 5) = 4 then writeln;
      end
      else begin
        if not (i mod 5) = 4 then writeln;
        Writeln(i: 8, '     ', Disp);
      end;
      if (i > 0) and (i mod 100 = 99) then Pause;
    end;
  writeln('Symbol table length = ', Length(SymbolTable));
  writeln;
end;

// Display all symbols in the Corpus with their frequency.
{procedure DisplayAllTokenFrequencies(const Corpus: TBVector);
var
  Counts: array of Integer;
  TST: String;
  i, j, k, S, LS, MaxSymbol: Integer;
  TokenList: TTokenCounts;
  Temp: TTokenCount;
begin

  // Find the maximum symbol value.
  MaxSymbol := 0;
  for i := 0 to High(Corpus) do
    if Corpus[i] > MaxSymbol then MaxSymbol := Corpus[i];

  // Initialize counts array.
  SetLength(Counts, MaxSymbol + 1);
  for i := 0 to MaxSymbol do
    Counts[i] := 0;

  // Count occurrences of each symbol.
  for i := 0 to High(Corpus) do
    Counts[Corpus[i]] := Counts[Corpus[i]] + 1;

  // Build list of tokens with count > 0.
  SetLength(TokenList, 0);
  for i := 0 to MaxSymbol do
    if Counts[i] > 0 then begin
      SetLength(TokenList, Length(TokenList) + 1);
      TokenList[High(TokenList)].Symbol := i;
      TokenList[High(TokenList)].Count := Counts[i];
    end;

  // Sort descending by count.
  for i := 0 to High(TokenList) - 1 do
    for j := i + 1 to High(TokenList) do
      if TokenList[i].Count < TokenList[j].Count then begin
        Temp := TokenList[i];
        TokenList[i] := TokenList[j];
        TokenList[j] := Temp;
      end;

  // Print all symbols with frequency.
  for i := 0 to High(TokenList) do begin
      writeln(i: 4, '  ', SymbolTable[TokenList[i].Symbol], '   ', TokenList[i].Count);
  {  S := TokenList[i].Symbol;
    LS := Length(SymbolTable[S]);
    TST := SymbolTable[S];        // TST is temporary SymbolTable character.
    for j := 1 to LS do begin     // Used for displaying below.
      k := Ord(TST[j]);
      // Unknown character is a hex, not a dot, also char 183, for display.
      if (k >= 32) and (k <= 126) then
        Write(Chr(k))
      else
        Write('\x', IntToHex(k, 2));
      // if (k < 32) or (k > 126) then TST[j] := Chr(183);
    end;
    Write(i: 5, S: 5, ' ': (10 - LS), '*', TST, '*', TokenList[i].Count: 5, '          ');
    if (i mod 4 = 3) then writeln;}
  end;
end;}

{ Reconstruction check }
// Reconstruct original corpus from linked list and symbol table.
procedure ReconstructText(Head: PTokenNode; out Text: String);
var
  Cur: PTokenNode;
begin
  Text := '';
  Cur := Head;

  while Cur <> nil do begin
    if Cur^.Tok <= High(SymbolTable) then
      Text := Text + SymbolTable[Cur^.Tok]
    else
      Text := Text + Chr(183);    // Chr(183) is non-display char. Need this?

    Cur := Cur^.Next;
  end;
end;

{ Computations and reports }
// Calculate time statistics.
procedure CalculateTimeStatistics;
begin
  // Total elapsed time.
  ElapsedMS := MilliSecondsBetween(t0, t1) - Round(StopTime);
  Hours := ElapsedMS div 3600000;
  Mins := ElapsedMS div 60000;
  Secs := (ElapsedMS mod 60000) / 1000.0;
  // Merge eotal elapsed time.
  MElapsedMS := MilliSecondsBetween(Mt0, Mt1) - Round(StopTime);
  MHours := MElapsedMS div 3600000;
  MMins := MElapsedMS div 60000;
  MSecs := (MElapsedMS mod 60000) / 1000.0;
end;

// Calculate number of symbol types and instances.
procedure CalculateSymbolCount;
var
  i, T: Integer;
begin
  MergedTypes := 0;
  UnmergedTypes := 0;

  // Count symbol types.
  for i := 0 to High(SymbolTable) do
    if Length(SymbolTable[i]) > 1 then
      Inc(MergedTypes)
    else
      Inc(UnmergedTypes);

  // Count token instances.
  MergedInstances := 0;
  UnmergedInstances := 0;

  for i := 0 to High(TokenizedCorpus) do begin
    T := TokenizedCorpus[i];
    if Length(SymbolTable[T]) > 1 then
      Inc(MergedInstances)
    else
      Inc(UnmergedInstances);
  end;
end;

// Calculate and report statistics on the symbol table.
procedure ReportSymbolLengths(const SymbolTable: TSymbolTable);
var
  i, MaxLen, MaxIndex, SumLen: Integer;
  SymbolLengths: array[1..10] of Integer;
begin
  MaxLen := 0;
  MaxIndex := -1;
  SumLen := 0;
  FillChar(SymbolLengths, SizeOf(SymbolLengths), 0);

  for i := 0 to High(SymbolTable) do begin
    if Length(SymbolTable[i]) > MaxLen then begin
      MaxLen := Length(SymbolTable[i]);
      MaxIndex := i;
    end;
    SumLen := SumLen + Length(SymbolTable[i]);
    if (Length(SymbolTable[i]) <= 9) then
      Inc(SymbolLengths[Length(SymbolTable[i])])
    else
      Inc(SymbolLengths[10]);
  end;

  if maxIndex >= 0 then  begin
    writeln('Longest symbol:');
    writeln('  Index: ', maxIndex);
    writeln('  Length: ', maxLen);
    writeln('  Value: "', SymbolTable[maxIndex], '"');
    writeln('Mean symbol length: ', SumLen / Length(SymbolTable): 6: 4);
  end;
end;

// Count the number of occurrences of each symbol.
procedure CountSymbols(const SymbolTable: TSymbolTable);
var
  Counts, Index: TIVector;
  i, j, k, N, TmpIndex: Integer;
begin
  // Allocate and zero Counts.
  SetLength(Counts, Length(SymbolTable));
  FillChar(Counts[0], SizeOf(Counts[0]), 0);

  // Count occurrences.
  for i := 0 to High(TokenizedCorpus) do
    Inc(Counts[TokenizedCorpus[i]]);

  // Build index array.
  N := Length(Counts);
  SetLength(Index, N);
  for i := 0 to N - 1 do
    Index[i] := i;

  // Selection sort index array by Counts[index] descending.
  for i := 0 to N - 2 do begin
    k := i;
    for j := i + 1 to N-1 do
      if Counts[Index[j]] > Counts[Index[k]] then
        k := j;

    // Swap Index[i] and Index[k].
    TmpIndex := Index[i];
    Index[i] := Index[k];
    Index[k] := TmpIndex;
  end;

  // Print top 60.
  writeln('Top 60 most frequent symbols:');
  for i := 0 to 59 do begin
    k := Index[i];
    write(i + 1: 8, ': Symbol ', k: 8, '  Count=', Counts[k]: 6, '  ', '"' + SymbolTable[k] + '"': 15);
    if ((i + 1) mod 3) = 0 then
      writeln;
  end;

  Pause;
end;

// Calculate token statistics.
// Count token usage.
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

// Build a list of merged token states.
procedure BuildMergedTokenStats(const Counts: TIVector; FirstMergedToken: Integer;
  out Stats: TMergedTokenStats);
var
  i, k: Integer;
begin
  SetLength(Stats, 0);
  k := 0;

  for i := FirstMergedToken to High(Counts) do begin
    SetLength(Stats, k + 1);
    Stats[k].TokenID := i;
    Stats[k].Count := Counts[i];
    Inc(k);
  end;
end;

// Sort by descending count.
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

{ Report Statistics }
// Report N most frequent merged tokens.
procedure ReportTopMergedTokens(const Stats: TMergedTokenStats;
  const SymbolTable: TSymbolTable; N: Integer);
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
    S := SymbolTable[Stats[i].TokenID];
    Writeln(i + 1:4, '  ID=', Stats[i].TokenID:6, '  Count=',
      Stats[i].Count:8, '  Symbol="', S, '"');
  end;
  Writeln;
end;

// Report singleton merged tokens.
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

// Report merged tokens never used.
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

// Report coverage of top merges.
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
end;

// Report token statistics.
procedure ReportTokenUsageStatistics;
var
  Counts: TIVector;
  Stats: TMergedTokenStats;
  FirstMergedToken: Integer;
begin
  Writeln('--- Token Statistics ---');
  Writeln('Merged token instances: ', MergedInstances);
  Writeln('Unmerged token instances: ', UnmergedInstances);
  Writeln('Mean token length: ', nCorpus / nTokenizedCorpus: 6: 4);
  if not FromSymbolTable then
    Writeln('Number of merges performed: ', MergeCount);
  CountTokenUsage(TokenizedCorpus, Length(SymbolTable), Counts);
  BuildMergedTokenStats(Counts, FirstMergedToken, Stats);
  SortMergedTokenStatsByCount(Stats);

  ReportTopMergedTokens(Stats, SymbolTable, 30);
  ReportUnusedMergedTokens(Stats);
  ReportSingletonMergedTokens(Stats);
  ReportTopMergeCoverage(Stats, 30);
end;

// Report symbol statistics.
procedure ReportSymbolStatistics;
begin
  Writeln('--- Symbol Statistics ---');
  Writeln('Number of symbols: ', nSymbols);
  // Writeln('Vocabulary size: ', nVocab);
  Writeln('Number of merged symbols: ', MergedTypes);
  Writeln('Mean merged symbol length: ');
  Writeln('Vocabulary compression ratio (merged symbols ÷ total symbols): ', MergedTypes / nSymbols);
  Writeln('Number of unmerged symbols: ', UnmergedTypes);
  writeln;
end;

// Report basic statistics (time, file names).
procedure ReportBasicStatistics;
var
  i: Integer;
begin
  writeln;
  Writeln('--- File Information ---');
  writeln('Files used in symbol table: ');
  for i := 0 to High(CorpusFileNames) do
    writeln(CorpusFileNames[i], '  ');
  writeln;

  Writeln('--- Time Statistics ---');
  writeln('Start time: ', DateTimetoStr(t0), '     End time: ', DateTimeToStr(t1));
  Writeln('Total elapsed time: ', Hours, ' hours, ', Mins, ' min ', Secs: 4: 4, ' sec');
end;

// Report BPE statistics.
procedure ReportBPEStatistics;
begin
  Writeln('--- BPE Statistics ---');
  if not FromSymbolTable then begin
    Writeln('Elapsed time applying merges: ', MHours, ' hours, ', Mmins, ' min ', Msecs: 4: 4, ' sec');
  end;
  Writeln('Original text size (bytes/tokens): ', nCorpus);
  Writeln('Encoded text size (bytes/tokens): ', nTokenizedCorpus);
  Writeln('Compression ratio: ', nCorpus   / nTokenizedCorpus:0: 4);
  if not FromSymbolTable then
    Writeln('Tokens per second: ', nCorpus / (ElapsedMS / 1000): 6: 4);
  writeln;
  end;

// Report all statistics.
procedure ReportStatistics;
begin
  CalculateSymbolCount;
  CalculateTimeStatistics;

  ReportBasicStatistics;
  if VerboseTokenize and (TextRec(Output).Handle = StdOutputHandle) then
    Pause;
  ReportBPEStatistics;
  ReportSymbolStatistics;
  ReportTokenUsageStatistics;
  Pause;
end;

{ Save data from tokenization }
// Save metadata.
procedure SaveMetaData(const MetaFileName: String);
var
  SaveOut: Text;
begin
  // Save current Output.
  SaveOut := Output;

  // Redirect Output to F.
  Assign(Output, MetaFileName);
  Rewrite(Output);

  ReportStatistics;

  // Restore Output to console.
  Close(Output);
  Output := SaveOut;

  writeln('File ', MetaFileName, ' successfully saved.');
  writeln;
end;

// Save merge table.
procedure SaveMergeTable(const Merges: TMergeArray; MergeFileName: String);
var
  F: file;
  i, n: Integer;
begin
  Assign(F, MergeFileName);
  Rewrite(F, 1);

  n := Length(Merges);
  BlockWrite(F, n, SizeOf(n));

  for i := 0 to n - 1 do begin
    BlockWrite(F, Merges[i].A, SizeOf(Integer));
    BlockWrite(F, Merges[i].B, SizeOf(Integer));
    BlockWrite(F, Merges[i].NewSym, SizeOf(Integer));
  end;

  Close(F);
  writeln('File ', MergeFileName, ' successfully saved.');
end;

// Save symbol table.
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
var
  F: file;
  NumSymbols: Integer;
  i, Len: Integer;
begin
  Assign(F, SymbolFileName);
  Rewrite(F, 1);

  // Magic.
  BlockWrite(F, Magic, SizeOf(Magic));

  // Version.
  BlockWrite(F, Version, 16);

  // Symbol count.
  NumSymbols := Length(SymbolTable);
  BlockWrite(F, NumSymbols, SizeOf(NumSymbols));

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
  writeln('File ', SymbolFileName, ' successfully saved.');
end;

// Save the output token list to a .bin file.
procedure WriteTokenList(const Part: TPart = B);
var
  i, iB, iE: Integer;
begin
    Case Part of
    B: begin
      iB := 0;
      iE := 99;
    end;
    E: begin
      iB := High(TokenizedCorpus) - 99;
      iE := High(TokenizedCorpus);
    end;
    F: begin
      iB := 0;
      iE := High(TokenizedCorpus);
    end;
  end;

  write('Tokenized Corpus, ');
  Case Part of
    B: write('First 100 bytes: ');
    E: write('Last 100 bytes: ');
    F: write('All bytes: ');
  end;
  writeln;

  for i := ib to iE do
    Write(TokenizedCorpus[i], ' ');
  writeln;
  writeln('Tokenized corpus length =  ', Length(TokenizedCorpus));
  Pause
end;

// Save the output tokenized corpus to a .bin file.
procedure SaveTokenList(const BinFileName: String);
var
  F: file of Int32;
  v: Int32;
  i: Integer;
begin
  AssignFile(F, BinFileName);
  Rewrite(F);

  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    Write(F, v);
  end;

  CloseFile(F);
  writeln('File ', BinFileName, ' successfully saved.');
end;

// Append to the output token list.  Need to work on this.
procedure AppendTokenListToBin(out nTokens: Integer);
var
  F: file of Int32;
  FileName: String;
  Cur: PTokenNode;
  v: Int32;
begin
  writeln('Name of token file (.bin) to append to: ');
  Readln(FileName);
  nTokens := 0;

  AssignFile(F, FileName);
  if FileExists(FileName) then begin
    Reset(F);
    Seek(F, FileSize(F));
  end
  else
    Rewrite(F);

  Cur := Head;
  while Cur <> nil do begin
    v := Cur^.Tok;
    Write(F, v);
    Inc(nTokens);
    Cur := Cur^.Next;
  end;

  CloseFile(F);
  writeln('File ', FileName, ' successfully appended.');
end;

{ Load tables }
// Load tokenized corpus from a .bin file.
procedure LoadTokenList(const BinFileName: String);
var
  F: file of Int32;
  v: Int32;
  i, Count: Integer;
begin
  AssignFile(F, BinFileName);
  Reset(F);

  // Determine number of tokens in file.
  Count := FileSize(F);

  // Allocate TokenizedCorpus.
  SetLength(TokenizedCorpus, Count);

  // Read them back.
  for i := 0 to Count - 1 do begin
    Read(F, v);
    TokenizedCorpus[i] := v;
  end;

  CloseFile(F);
  nTokenizedCorpus := Length(TokenizedCorpus);
  Writeln('Loaded ', Count, ' tokens from ', BinFileName);
end;

// Load the merge table from file.
procedure LoadMergeTable(out Merges: TMergeArray);
var
  F: File;
  i, n: Integer;
  FileName: String;
begin
  writeln('Name for merge table: ');
  Readln(FileName);

  Assign(F, FileName);
  Reset(F, 1);

  BlockRead(F, n, SizeOf(n));
  SetLength(Merges, n);

  for i := 0 to n - 1 do begin
    BlockRead(F, Merges[i].A, SizeOf(Integer));
    BlockRead(F, Merges[i].B, SizeOf(Integer));
    BlockRead(F, Merges[i].NewSym, SizeOf(Integer));
  end;

  Close(F);
  writeln('File ', FileName, ' successfully loaded.');
end;

// Reconstruct and save corpus to file.
procedure ReconstructToFile(Head: PTokenNode; const Table: TSymbolTable; const ReconFileName: string);
var
  Cur: PTokenNode;
  F: File;
  s: string;
begin
  Assign(F, ReconFileName);
  Rewrite(F, 1);

  Cur := Head;
  while Cur <> nil do begin
    s := Table[Cur^.Tok];
    if Length(s) > 0 then
      BlockWrite(F, s[1], Length(s));
    Cur := Cur^.Next;
  end;

  Close(F);
  if SaveFiles then
    writeln('File ', ReconFileName, ' successfully saved.');
end;

// Reconstruct corpus from tokenized corpus.
procedure DetokenizeToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
var
  i, iB, iE, t, symIndex: Integer;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 499;
    end;
    E: begin
      iB := High(TokenizedCorpus) - 499;
      iE := High(TokenizedCorpus);
    end;
    F: begin
      iB := 0;
      iE := High(TokenizedCorpus);
    end;
  end;

  write('Detokenized Corpus, ');
  Case Part of
    B: write('First 500 bytes: ');
    E: write('Last 500 bytes: ');
    F: write('All bytes: ');
  end;
  writeln;
  Pause;
  for i := iB to iE do begin
    t := TokenizedCorpus[i];

    if t < 256 then begin
      // Raw byte.
      Write(Char(t));
    end
    else if t < 260 then begin
      // Special tokens.
      case t of
        256: Write('<BOS>');
        257: Write('<EOS>');
        258: Write('<UNK>');
        259: Write('<PAD>');
      end;
    end
    else begin
      // Symbol table entry.
      symIndex := t - 260;
      Write(SymbolTable[symIndex]);   // Or just SymbolTable[t]?? Says ChaptGPT.
    end;
  end;

  writeln;
end;

function CompareByLength(List: TStringList; Index1, Index2: Integer): Integer;
begin
  // Sort longest first
  Result := Length(List[Index2]) - Length(List[Index1]);
end;

// Run the tokenizer.
procedure RunTokenize(const Corpus: TBVector);
begin
  // Timing.
  t0 := Now;       // Start of timing for entire tokenization;
  StopTime := 0;   // Time to subtract from timing.

  // Create the TokenList.
  writeln('Maximum symbols = ', MaxVocab, '. Maximum merges = ', MaxMerges, '. Maximum pair counts = ', MaxPairCount, '. Tokenizing...');
  writeln('X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. P = Program information. M = Merging information. Merging...');
  BuildTokenListFromCorpus(Corpus);

  // Report the Token List.
  if ShowTokenWork and VerboseTokenize and VeryVerbose and DisplayCorpus then begin
    Writeln('--- Initial Token List ---');
    DisplayCountTokenList(Head, nCorpus);
  end;
  nCorpus := Length(Corpus);

  // First merge symbol is StartSymbol, 260.
  InitSymbolTable;

  // Run BPE.
  Mt0 := Now;
  TrainBPEHash(Head, Tail, MaxMerges, MergeCount, StartSymbol);
  Mt1 := Now;

  // Insert BOS and EOS.
  InsertBOS(Head, Tail);
  InsertEOS(Head, Tail);

  // Timing.
  t1 := Now;

  // Count nTokenizedCorpus (and display).
  DisplayCountTokenList(Head, nTokenizedCorpus);

  if ShowTokenWork and VerboseTokenize then begin
    Writeln('---  Token Frequencies ---');
    CountSymbols(SymbolTable);
  end;

  nSymbols := Length(SymbolTable);
  // Display symbol table.
  if VerboseTokenize then
    DisplaySymbolTable;
  nVocab := nSymbols;

  // Create the tokenized corpus.
  CreateTokenizedCorpus;

  // Report statistics.
  if VerboseTokenize then begin
    ReportStatistics;
    Pause;
  end;

  // Verify by reconstructing.
  if ShowVerification and VerboseTokenize and DisplayCorpus then begin
    writeln('--- Reconstructed Corpus ---');
    ReconstructText(Head, Reconstructed);
    Writeln(Reconstructed);
  end;

  // Save various files.
  if SaveFiles then begin
    writeln;
    ChDir(WorkingDir);
    writeln('--- Saving Files ---');
    SaveTokenList(WorkingName + '.tok');
    SaveSymbolTable(WorkingName + '.sym', SymbolTable);
    SaveMergeTable(Merges, WorkingName + '.mer');
    SaveMetaData(WorkingName + '.meta');
    ReconstructToFile(Head, SymbolTable, WorkingName + '.rcn');
    ChDir('..');
  end;

  writeln('End of tokenization.');
  Pause;
  //nTokens := nTokenizedCorpus;    // For embedding, need nTokens.

  if VerboseTokenize then Begin
    writeln('First 150 tokens of tokenized corpus:');
    for i := 0 to 149 do
      write(TokenizedCorpus[i], ' ');
    writeln;
    Pause;
  end;
end;

end.

