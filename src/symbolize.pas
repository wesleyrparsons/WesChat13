unit Symbolize;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com. }
{ Build and train WesTokenizer and UDTokenizer byte-pair symbol tables }

interface

uses
  { RTL and platform units }
  Classes, Crt, DateUtils, FileUtil, Math, SysUtils,
  { WesChat units }
  Display, Global, IOHandler;

type
  { Token linked-list types }
  PTokenNode = ^TTokenNode;
  TTokenNode = record
    Tok: Integer;
    Prev, Next: PTokenNode;
  end;

  { Pair-hash types }
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

  { Lazy pair max-heap types }
  TPairHeapEntry = record
    A, B: Integer;
    Count: Integer;
  end;

  TPairHeap = record
    Items: array of TPairHeapEntry;
    Count: Integer;
  end;

  { Merge types }
  TMerge = record
    A, B: Integer;
    NewSym: Integer;
  end;
  TMergeArray = array of TMerge;

var
  { Symbolization state }
  StartSymbol: Integer = 260;           // First merge symbol; reset by RunSymbolize after base symbols are initialized.
  FinalTokenCount: Integer;             // Number of tokens remaining after BPE merging.
  ElapsedMS: Int64;                     // Elapsed symbolization time in milliseconds.
  Hours, Mins: Int64;                   // Elapsed hours and minutes.
  Secs: Double;                         // Elapsed seconds.
  Head, Tail: PTokenNode;               // First and last nodes in the working token list.
  Merges: TMergeArray;                  // Recorded BPE merge operations.

// Read a corpus file as bytes.
procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);

// Save the learned merge table.
procedure SaveMergeTable(const Merges: TMergeArray; MergeFileName: String);

// Save symbolization statistics.
procedure SaveMetaData(const MetaFileName: String);

// Report symbolization statistics.
procedure ReportStatistics;

// Build a symbol table and BPE merge table from a corpus.
procedure RunSymbolize(const Corpus: TBVector);

implementation

{ UD tag support }
// Return the reserved token ID for a UD tag.
function ReservedTagToken(const S: string): Integer;
begin
  Result := -1;

  // Parts of speech.
  if S = '|noun' then Result := TokNoun
  else if S = '|verb' then Result := TokVerb
  else if S = '|adj' then Result := TokAdj
  else if S = '|adv' then Result := TokAdv
  else if S = '|prep' then Result := TokPrep
  else if S = '|det' then Result := TokDet
  else if S = '|pron' then Result := TokPron
  else if S = '|aux' then Result := TokAux
  else if S = '|sconj' then Result := TokSConj
  else if S = '|cconj' then Result := TokCConj
  else if S = '|part' then Result := TokPart
  else if S = '|intj' then Result := TokIntj
  else if S = '|num' then Result := TokNum
  else if S = '|propn' then Result := TokPropn
  else if S = '|x' then Result := TokX
  else if S = '|sym' then Result := TokSym
  else if S = '|punct' then Result := TokPunct

  // Number.
  else if S = '|sg' then Result := TokSing
  else if S = '|pl' then Result := TokPlur

  // Person.
  else if S = '|1p' then Result := TokPerson1
  else if S = '|2p' then Result := TokPerson2
  else if S = '|3p' then Result := TokPerson3

  // Case.
  else if S = '|nom' then Result := TokNom
  else if S = '|acc' then Result := TokAcc
  else if S = '|gen' then Result := TokGen
  else if S = '|dat' then Result := TokDat
  else if S = '|loc' then Result := TokLoc
  else if S = '|ins' then Result := TokIns
  else if S = '|voc' then Result := TokVoc

  // Gender.
  else if S = '|masc' then Result := TokMasc
  else if S = '|fem' then Result := TokFem
  else if S = '|neut' then Result := TokNeut
  else if S = '|common' then Result := TokCommon

  // Tense.
  else if S = '|past' then Result := TokPast
  else if S = '|pres' then Result := TokPres
  else if S = '|fut' then Result := TokFut

  // Mood.
  else if S = '|ind' then Result := TokMoodInd
  else if S = '|imp' then Result := TokMoodImp
  else if S = '|sub' then Result := TokMoodSub
  else if S = '|cond' then Result := TokMoodCond
  else if S = '|opt' then Result := TokMoodOpt

  // Verb form.
  else if S = '|fin' then Result := TokVerbFin
  else if S = '|inf' then Result := TokVerbInf
  else if S = '|ger' then Result := TokVerbGer
  else if S = '|conv' then Result := TokVerbConv
  else if S = '|participle' then Result := TokVerbPart

  // Voice.
  else if S = '|act' then Result := TokVoiceAct
  else if S = '|pass' then Result := TokVoicePass
  else if S = '|mid' then Result := TokVoiceMid

  // Aspect.
  else if S = '|impf' then Result := TokAspectImp
  else if S = '|perf' then Result := TokAspectPerf
  else if S = '|prog' then Result := TokAspectProg
  else if S = '|prosp' then Result := TokAspectProsp

  // Degree.
  else if S = '|pos' then Result := TokDegreePos
  else if S = '|cmp' then Result := TokDegreeCmp
  else if S = '|sup' then Result := TokDegreeSup
  else if S = '|abs' then Result := TokDegreeAbs

  // Definiteness.
  else if S = '|def' then Result := TokDefiniteDef
  else if S = '|indef' then Result := TokDefiniteInd

  // Pronoun type.
  else if S = '|art' then Result := TokPronArt
  else if S = '|dem' then Result := TokPronDem
  else if S = '|int' then Result := TokPronInt
  else if S = '|prs' then Result := TokPronPrs
  else if S = '|rel' then Result := TokPronRel
  else if S = '|indpron' then Result := TokPronInd
  else if S = '|negpron' then Result := TokPronNeg
  else if S = '|tot' then Result := TokPronTot

  // Possessive.
  else if S = '|poss' then Result := TokPoss

  // Reflexive.
  else if S = '|refl' then Result := TokRefl

  // Polarity.
  else if S = '|neg' then Result := TokPolarityNeg
  else if S = '|positive' then Result := TokPolarityPos

  // Numeral type.
  else if S = '|card' then Result := TokNumCard
  else if S = '|ord' then Result := TokNumOrd
  else if S = '|frac' then Result := TokNumFrac
  else if S = '|mult' then Result := TokNumMult
  else if S = '|sets' then Result := TokNumSets
  else if S = '|dist' then Result := TokNumDist

  // Numeral form.
  else if S = '|digit' then Result := TokNumDigit
  else if S = '|numword' then Result := TokNumWord
  else if S = '|roman' then Result := TokNumRoman

  // Miscellaneous lexical properties.
  else if S = '|abbr' then Result := TokAbbr
  else if S = '|foreign' then Result := TokForeign
  else if S = '|typo' then Result := TokTypo

  // Animacy.
  else if S = '|anim' then Result := TokAnim
  else if S = '|inan' then Result := TokInan
  else if S = '|human' then Result := TokHuman
  else if S = '|nonhuman' then Result := TokNonHuman;
end;

// Initialize reserved UD tags in the symbol table.
procedure InitUDTagSymbols;
var
  i: Integer;
begin
  SetLength(SymbolTable, UDTagBoundary);

  // Mark unused reserved positions.
  for i := FirstTagToken to UDTagBoundary - 1 do
    SymbolTable[i] := '<RES' + IntToStr(i) + '>';

  // Parts of speech.
  SymbolTable[TokNoun] := '|noun';
  SymbolTable[TokVerb] := '|verb';
  SymbolTable[TokAdj] := '|adj';
  SymbolTable[TokAdv] := '|adv';
  SymbolTable[TokPrep] := '|prep';
  SymbolTable[TokDet] := '|det';
  SymbolTable[TokPron] := '|pron';
  SymbolTable[TokAux] := '|aux';
  SymbolTable[TokSConj] := '|sconj';
  SymbolTable[TokCConj] := '|cconj';
  SymbolTable[TokPart] := '|part';
  SymbolTable[TokIntj] := '|intj';
  SymbolTable[TokNum] := '|num';
  SymbolTable[TokPropn] := '|propn';
  SymbolTable[TokX] := '|x';
  SymbolTable[TokSym] := '|sym';
  SymbolTable[TokPunct] := '|punct';

  // Number.
  SymbolTable[TokSing] := '|sg';
  SymbolTable[TokPlur] := '|pl';

  // Person.
  SymbolTable[TokPerson1] := '|1p';
  SymbolTable[TokPerson2] := '|2p';
  SymbolTable[TokPerson3] := '|3p';

  // Case.
  SymbolTable[TokNom] := '|nom';
  SymbolTable[TokAcc] := '|acc';
  SymbolTable[TokGen] := '|gen';
  SymbolTable[TokDat] := '|dat';
  SymbolTable[TokLoc] := '|loc';
  SymbolTable[TokIns] := '|ins';
  SymbolTable[TokVoc] := '|voc';

  // Gender.
  SymbolTable[TokMasc] := '|masc';
  SymbolTable[TokFem] := '|fem';
  SymbolTable[TokNeut] := '|neut';
  SymbolTable[TokCommon] := '|common';

  // Tense.
  SymbolTable[TokPast] := '|past';
  SymbolTable[TokPres] := '|pres';
  SymbolTable[TokFut] := '|fut';

  // Mood.
  SymbolTable[TokMoodInd] := '|ind';
  SymbolTable[TokMoodImp] := '|imp';
  SymbolTable[TokMoodSub] := '|sub';
  SymbolTable[TokMoodCond] := '|cond';
  SymbolTable[TokMoodOpt] := '|opt';

  // Verb form.
  SymbolTable[TokVerbFin] := '|fin';
  SymbolTable[TokVerbInf] := '|inf';
  SymbolTable[TokVerbGer] := '|ger';
  SymbolTable[TokVerbPart] := '|participle';
  SymbolTable[TokVerbConv] := '|conv';

  // Voice.
  SymbolTable[TokVoiceAct] := '|act';
  SymbolTable[TokVoicePass] := '|pass';
  SymbolTable[TokVoiceMid] := '|mid';

  // Aspect.
  SymbolTable[TokAspectImp] := '|impf';
  SymbolTable[TokAspectPerf] := '|perf';
  SymbolTable[TokAspectProg] := '|prog';
  SymbolTable[TokAspectProsp] := '|prosp';

  // Degree.
  SymbolTable[TokDegreePos] := '|pos';
  SymbolTable[TokDegreeCmp] := '|cmp';
  SymbolTable[TokDegreeSup] := '|sup';
  SymbolTable[TokDegreeAbs] := '|abs';

  // Definiteness.
  SymbolTable[TokDefiniteDef] := '|def';
  SymbolTable[TokDefiniteInd] := '|indef';

  // Pronoun type.
  SymbolTable[TokPronArt] := '|art';
  SymbolTable[TokPronDem] := '|dem';
  SymbolTable[TokPronInt] := '|int';
  SymbolTable[TokPronPrs] := '|prs';
  SymbolTable[TokPronRel] := '|rel';
  SymbolTable[TokPronInd] := '|indpron';
  SymbolTable[TokPronNeg] := '|negpron';
  SymbolTable[TokPronTot] := '|tot';

  // Possessive.
  SymbolTable[TokPoss] := '|poss';

  // Reflexive.
  SymbolTable[TokRefl] := '|refl';

  // Polarity.
  SymbolTable[TokPolarityNeg] := '|neg';
  SymbolTable[TokPolarityPos] := '|positive';

  // Numeral type.
  SymbolTable[TokNumCard] := '|card';
  SymbolTable[TokNumOrd] := '|ord';
  SymbolTable[TokNumFrac] := '|frac';
  SymbolTable[TokNumMult] := '|mult';
  SymbolTable[TokNumSets] := '|sets';
  SymbolTable[TokNumDist] := '|dist';

  // Numeral form.
  SymbolTable[TokNumDigit] := '|digit';
  SymbolTable[TokNumWord] := '|numword';
  SymbolTable[TokNumRoman] := '|roman';

  // Miscellaneous lexical properties.
  SymbolTable[TokAbbr] := '|abbr';
  SymbolTable[TokForeign] := '|foreign';
  SymbolTable[TokTypo] := '|typo';

  // Animacy.
  SymbolTable[TokAnim] := '|anim';
  SymbolTable[TokInan] := '|inan';
  SymbolTable[TokHuman] := '|human';
  SymbolTable[TokNonHuman] := '|nonhuman';
end;

{ Corpus input }
// Read a corpus file as a binary byte stream.
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

  // Display the corpus as it is read.
  if VerboseTokenize then
    Writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VerboseTokenize then
      if DisplayEachByteRead then
          Write(Chr(B));
  end;
  CloseFile(F);
  if VerboseTokenize then
    Writeln;

  if DisplayCorpus then begin
    Writeln('First ', DisplayLength, ' bytes of corpus: ');
    for i := 0 to Min(DisplayLength, High(OneCorpus)) do
      Write(OneCorpus[i], ' ');
    Pause;
    Writeln;
  end;

  // Display the corpus length.
  Writeln('Read ', Size, ' bytes from ', FileName, '.');
end;

{ Token linked list }
// Return True when a token must not participate in BPE merges.
function IsSpecial(T: Integer): Boolean;
begin
  Result := (T = BOS) or (T = EOS) or (T = PAD) or (T = UNK);

  if TokenizerKind = UDTokenizer then
    Result := Result or ((T >= FirstTagToken) and (T < UDTagBoundary));
end;

// Build the initial token linked list from the corpus.
procedure BuildTokenListFromCorpus(const Corpus: TBVector);
var
  i, j, TagToken: Integer;
  Prev: PTokenNode;
  TagString: string;

  procedure AppendToken(AToken: Integer);
  var
    Node: PTokenNode;
  begin
    New(Node);

    Node^.Tok := AToken;
    Node^.Prev := Prev;
    Node^.Next := nil;

    if Prev <> nil then
      Prev^.Next := Node
    else
      Head := Node;

    Prev := Node;
  end;

begin
  Head := nil;
  Tail := nil;
  Prev := nil;
  i := 0;

  while i <= High(Corpus) do begin

    // For UD tokenization, recognize each |tag as one reserved token.
    if (TokenizerKind = UDTokenizer) and (Corpus[i] = Ord('|')) then begin
      TagString := '|';
      j := i + 1;

      // A tag ends at the next | or whitespace.
      while (j <= High(Corpus)) and (Corpus[j] <> Ord('|')) and (Corpus[j] <> 9)
        and (Corpus[j] <> 10) and (Corpus[j] <> 13) and (Corpus[j] <> 32) do begin
        TagString := TagString + Chr(Corpus[j]);
        Inc(j);
      end;

      TagToken := ReservedTagToken(TagString);

      if TagToken >= 0 then begin
        AppendToken(TagToken);
        i := j;
        Continue;
      end;

      // Not a reserved UD tag. Treat the | as an ordinary corpus byte.
      AppendToken(Corpus[i]);
      Inc(i);
      Continue;
    end;

    // Tiny Stories separator byte 254 becomes EOS.
    if Corpus[i] = 254 then
      AppendToken(EOS)
    else
      AppendToken(Corpus[i]);

    Inc(i);
  end;

  Tail := Prev;
end;

// Free the token linked list.
procedure FreeTokenList(var Head, Tail: PTokenNode);
var
  Cur, Next: PTokenNode;
begin
  Cur := Head;

  while Cur <> nil do begin
    Next := Cur^.Next;
    Dispose(Cur);
    Cur := Next;
  end;

  Head := nil;
  Tail := nil;
end;

{ Pair hash }
// Initialize the open-addressed pair hash.
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

// Hash a token pair into the pair table.
function HashPair(A, B, Capacity: Integer): Integer;
var
  H: QWord;
begin
  H := QWord(Cardinal(A)) * 1000003 + QWord(Cardinal(B));
  Result := Integer(H mod QWord(Capacity));
end;

// Find the slot containing or available for a token pair.
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

// Return True when a pair is eligible for BPE merging.
function PairAllowed(A, B: Integer): Boolean;
begin
  Result := not (IsSpecial(A) or IsSpecial(B));
end;

// Return the current count for a token pair.
function PairGetCount(const H: TPairHash; A, B: Integer): Integer;
var
  Idx: Integer;
begin
  if H.Capacity <= 0 then begin
    Result := 0;
    Exit;
  end;

  Idx := FindSlot(H, A, B);

  if (H.Entries[Idx].State = psUsed) and (H.Entries[Idx].A = A) and (H.Entries[Idx].B = B) then
    Result := H.Entries[Idx].Count
  else
    Result := 0;
end;

// Increment a pair count in the hash table.
function PairIncHash(var H: TPairHash; A, B: Integer): Integer;
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

  Result := H.Entries[Idx].Count;
end;

// Decrement a pair count in the hash table.
function PairDecHash(var H: TPairHash; A, B: Integer): Integer;
var
  Idx: Integer;
begin
  Result := 0;
  Idx := FindSlot(H, A, B);

  if (H.Entries[Idx].State = psUsed) and (H.Entries[Idx].A = A) and (H.Entries[Idx].B = B) then begin
    if H.Entries[Idx].Count > 0 then
      Dec(H.Entries[Idx].Count);
    Result := H.Entries[Idx].Count;
  end;
end;

// Initialize pair counts from the current linked list.
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

{ Pair max-heap }
// Initialize the lazy max-heap.
procedure InitPairHeap(var Heap: TPairHeap; InitialCapacity: Integer);
begin
  if InitialCapacity < 16 then
    InitialCapacity := 16;

  SetLength(Heap.Items, InitialCapacity);
  Heap.Count := 0;
end;

// Compare two heap entries for max-heap ordering.
function HeapEntryGreater(const L, R: TPairHeapEntry): Boolean;
begin
  if L.Count <> R.Count then
    Result := L.Count > R.Count
  else if L.A <> R.A then
    Result := L.A < R.A
  else
    Result := L.B < R.B;
end;

// Swap two heap entries.
procedure HeapSwap(var X, Y: TPairHeapEntry);
var
  T: TPairHeapEntry;
begin
  T := X;
  X := Y;
  Y := T;
end;

// Push a pair count onto the lazy max-heap.
procedure HeapPush(var Heap: TPairHeap; A, B, Count: Integer);
var
  I, Parent: Integer;
begin
  if Count <= 0 then Exit;

  if Heap.Count >= Length(Heap.Items) then begin
    if Length(Heap.Items) = 0 then
      SetLength(Heap.Items, 16)
    else
      SetLength(Heap.Items, Length(Heap.Items) * 2);
  end;

  I := Heap.Count;
  Heap.Items[I].A := A;
  Heap.Items[I].B := B;
  Heap.Items[I].Count := Count;
  Inc(Heap.Count);

  while I > 0 do begin
    Parent := (I - 1) div 2;
    if not HeapEntryGreater(Heap.Items[I], Heap.Items[Parent]) then Break;

    HeapSwap(Heap.Items[I], Heap.Items[Parent]);
    I := Parent;
  end;
end;

// Pop the largest entry from the max-heap.
function HeapPop(var Heap: TPairHeap; out Entry: TPairHeapEntry): Boolean;
var
  I, Left, Right, Best: Integer;
begin
  if Heap.Count <= 0 then begin
    Result := False;
    Exit;
  end;

  Entry := Heap.Items[0];
  Dec(Heap.Count);

  if Heap.Count > 0 then begin
    Heap.Items[0] := Heap.Items[Heap.Count];

    I := 0;
    while True do begin
      Left := I * 2 + 1;
      Right := Left + 1;
      Best := I;

      if (Left < Heap.Count) and HeapEntryGreater(Heap.Items[Left], Heap.Items[Best]) then
        Best := Left;

      if (Right < Heap.Count) and HeapEntryGreater(Heap.Items[Right], Heap.Items[Best]) then
        Best := Right;

      if Best = I then
        Break;

      HeapSwap(Heap.Items[I], Heap.Items[Best]);
      I := Best;
    end;
  end;

  Result := True;
end;

// Rebuild the heap from current pair-hash counts.
procedure InitPairHeapFromHash(const H: TPairHash; var Heap: TPairHeap);
var
  I: Integer;
begin
  InitPairHeap(Heap, H.Used + 1024);

  for I := 0 to H.Capacity - 1 do
    if (H.Entries[I].State = psUsed) and (H.Entries[I].Count > 0) then
      HeapPush(Heap, H.Entries[I].A, H.Entries[I].B, H.Entries[I].Count);
end;

// Increment a pair count and add its new count to the heap.
procedure PairIncHashHeap(var H: TPairHash; var Heap: TPairHeap; A, B: Integer);
var
  C: Integer;
begin
  if not PairAllowed(A, B) then Exit;

  C := PairIncHash(H, A, B);
  HeapPush(Heap, A, B, C);
end;

// Decrement a pair count and preserve its current count in the heap.
procedure PairDecHashHeap(var H: TPairHash; var Heap: TPairHeap; A, B: Integer);
var
  C: Integer;
begin
  if not PairAllowed(A, B) then Exit;

  C := PairDecHash(H, A, B);
  if C > 0 then
    HeapPush(Heap, A, B, C);
end;

// Return the highest valid current pair count from the lazy heap.
function FindBestPairHeap(const H: TPairHash; var Heap: TPairHeap; out A, B: Integer): Integer;
var
  E: TPairHeapEntry;
  CurrentCount: Integer;
begin
  A := -1;
  B := -1;
  Result := 0;

  while HeapPop(Heap, E) do begin
    CurrentCount := PairGetCount(H, E.A, E.B);

    if CurrentCount <= 0 then Continue;

    if CurrentCount = E.Count then begin
      A := E.A;
      B := E.B;
      Result := CurrentCount;
      Exit;
    end;

    // Stale heap entry. Push the current count and keep looking.
    HeapPush(Heap, E.A, E.B, CurrentCount);
  end;
end;

{ Merge operations }
// Update neighboring pair counts before merging a pair.
procedure UpdatePairsForMergeHash(Node: PTokenNode; NewTok: Integer; var H: TPairHash; var Heap: TPairHeap);
var
  A, B: Integer;
begin
  if (Node = nil) or (Node^.Next = nil) then Exit;

  A := Node^.Tok;
  B := Node^.Next^.Tok;

  // Remove (A, B).
  PairDecHashHeap(H, Heap, A, B);

  // Remove (Prev, A).
  if Node^.Prev <> nil then
    PairDecHashHeap(H, Heap, Node^.Prev^.Tok, A);

  // Remove (B, Next).
  if Node^.Next^.Next <> nil then
    PairDecHashHeap(H, Heap, B, Node^.Next^.Next^.Tok);

  // Add (Prev, NewTok).
  if Node^.Prev <> nil then
    PairIncHashHeap(H, Heap, Node^.Prev^.Tok, NewTok);

  // Add (NewTok, Next).
  if Node^.Next^.Next <> nil then
    PairIncHashHeap(H, Heap, NewTok, Node^.Next^.Next^.Tok);
end;

// Merge a node and its right neighbor in the token linked list.
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

// Merge every occurrence of one selected token pair.
procedure MergeAllPairsHash(var Head, Tail: PTokenNode; A, B, NewTok: Integer; var H: TPairHash; var Heap: TPairHeap);
var
  Cur: PTokenNode;
begin
  Cur := Head;

  while (Cur <> nil) and (Cur^.Next <> nil) do begin
    if not (IsSpecial(Cur^.Tok) or IsSpecial(Cur^.Next^.Tok)) then begin
      if (Cur^.Tok = A) and (Cur^.Next^.Tok = B) then begin
        UpdatePairsForMergeHash(Cur, NewTok, H, Heap);
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

// Record one learned BPE merge.
procedure RecordMerge(var Merges: TMergeArray; MergeIndex, A, B, NewSym: Integer);
begin
  if MergeIndex >= Length(Merges) then
    SetLength(Merges, MergeIndex + 1);

  Merges[MergeIndex].A := A;
  Merges[MergeIndex].B := B;
  Merges[MergeIndex].NewSym := NewSym;
end;

{ Symbol table }
// Initialize byte, special, and optional UD symbols.
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
  if TokenizerKind = UDTokenizer then
    InitUDTagSymbols;
end;

// Add one learned merge symbol to the symbol table.
procedure AddMergeSymbol(NewTok, A, B: Integer);
begin
  if NewTok >= Length(SymbolTable) then
    SetLength(SymbolTable, NewTok + 1);

  SymbolTable[NewTok] := SymbolTable[A] + SymbolTable[B];
end;

// Display number of symbols created.
procedure DisplaySymbolProgress(const CurrentSymbols, MaximumSymbols: Integer);
begin
  Write(#13);
  ClrEol;
  Write(CurrentSymbols, ' symbols created out of a maximum of ', MaximumSymbols, '.');
  Flush(Output);
end;

{ BPE training }
// Train BPE merges using the pair hash and lazy max-heap.
procedure TrainBPEHash(var Head, Tail: PTokenNode; MaxMerges, MaxSymbols: Integer; var MergeCount, StartSymbol: Integer);
var
  m, BestCount, A, B: Integer;
  f, BaseName: string;
  BreakRequested: Boolean;
  H: TPairHash;
  Heap: TPairHeap;

  procedure ReadMergeIfKeyPressed;
  var
    key: Char;
  begin
    key := CheckForControlKey;
    case key of
      'b', 'B': begin
        Writeln('Break requested. Exiting loop.');
        BreakRequested := True;
      end;
      'i', 'I': begin
        Writeln;
        ReportProgramInfo;
        Pause;
      end;
      'm', 'M': begin
        Writeln;
        Writeln('Maximum symbols = ', MaxSymbols, '. Current symbols = ', Length(SymbolTable),
          '. Maximum merges = ', MaxMerges, '. Hash capacity = ', H.Capacity, '. Used slots = ', H.Used, '. Heap entries = ', Heap.Count, '. Best count = ', BestCount, '.');
        Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. I = program Information. ');
        Writeln('P = Pause. M = Merging information. S = Save. Symbolizing and merging...');
        Pause;
      end;
      'p', 'P': begin        // Pause work.
        Write('Paused... ');
        Pause;
      end;
      's', 'S': begin
        try
          if Trim(WorkingName) = '' then
            BaseName := 'symboltable'
          else
            BaseName := ChangeFileExt(ExtractFileName(WorkingName), '');

          if Trim(BaseName) = '' then
            BaseName := 'symboltable';

          if Trim(SymbolDir) = '' then begin
            SymbolDir := IncludeTrailingPathDelimiter(GetCurrentDir) +
              'WesChatWork' + DirectorySeparator + 'symbols' + DirectorySeparator;
            ForceDirectories(SymbolDir);
          end;

          // Normalize the symbol directory.
          SymbolDir := IncludeTrailingPathDelimiter(SymbolDir);

          if not DirectoryExists(SymbolDir) then begin
            Writeln('Creating symbol directory: ', SymbolDir);
            ForceDirectories(SymbolDir);
          end;

          f := SymbolDir + BaseName + '_' + FormatDateTime('yyyy-mm-dd_hhnnss', Now) + '.sym';

          Writeln('Saving symbol table to: ', f);

          SaveSymbolTable(f, SymbolTable);

          Pause;
        except
          on E: Exception do begin
            Writeln('Error saving symbol table: ', E.ClassName, ' ', E.Message);
            Writeln('SymbolDir = "', SymbolDir, '"');
            Writeln('WorkingName = "', WorkingName, '"');
            Writeln('BaseName = "', BaseName, '"');
            Writeln('Target file = "', f, '"');
            Pause;
          end;
        end;
      end;
      'v', 'V': begin
        VerboseTokenize := not VerboseTokenize;
        Writeln('Verbose tokenize mode: ', VerboseTokenize);
        Pause;
      end;
      'x', 'X': begin
        Writeln('Exit requested. Stopping execution.');
        Pause;
        Halt;
      end;
    end;
  end;

begin
  MergeCount := 0;
  BreakRequested := False;

  Write(DateTimeToStr(Now), '  B = Break out of merge loop. I = program Information. M = Merging information. P = Pause. ');
  Writeln('S = Save. V = toggle Verbose mode. X = Exit program. Symbolizing and merging...');
  //Writeln;

  if DisplayMergeWork then
    Writeln('--- List of Merges (Hash) ---');

  // Build pair counts once, then maintain them incrementally.
  InitPairHash(H, MaxPairCount * 2 + 1024);
  InitPairHashFromList(Head, H);
  InitPairHeapFromHash(H, Heap);

  // Merge loop.
  for m := 1 to MaxMerges do begin
    // Check for key pressed.
    if PauseIfKeyPressed then
      ReadMergeIfKeyPressed;

    // Check for break request.
    if BreakRequested then Break;

    // Stop if symbol table got too large.
    if Length(SymbolTable) >= MaxSymbols then begin
      Writeln('Stopping: symbol table reached ', MaxSymbols, ' entries.');
      Break;
    end;

    // The open-address hash table never truly deletes zero-count pairs.
    // If it gets crowded, rebuild it from the current linked list.
    if H.Used > (H.Capacity * 7) div 10 then begin
      InitPairHash(H, H.Capacity * 2);
      InitPairHashFromList(Head, H);
      InitPairHeapFromHash(H, Heap);
    end;

    BestCount := FindBestPairHeap(H, Heap, A, B);

    // Stop if no useful merges remain.
    if BestCount < 2 then begin
      Writeln;
      Writeln('Stopping: no more valid merges at iteration ', m, '.');
      Break;
    end;

    // Perform merge.
    MergeAllPairsHash(Head, Tail, A, B, StartSymbol, H, Heap);

    // Lazy heap entries accumulate. Compact occasionally by rebuilding
    // the heap from the current hash counts.
    if Heap.Count > (H.Used * 8 + 100000) then
      InitPairHeapFromHash(H, Heap);

    AddMergeSymbol(StartSymbol, A, B);
    RecordMerge(Merges, MergeCount, A, B, StartSymbol);

    // Display new symbols if not verbose tokenize.
    if not VerboseTokenize then
      if (Length(SymbolTable) mod 50) = 0 then
        DisplaySymbolProgress(Length(SymbolTable), MaxSymbols);

    Inc(MergeCount);
    Inc(StartSymbol);

    if DisplayMergeWork then begin
      Write(MergeCount, ' Merged (', A:5, ',', B:5, ') -> (', StartSymbol - 1:5, ') #', BestCount);
      if (MergeCount mod 4) = 0 then
        Writeln
      else
        Write('  |  ');
    end;
  end;

  Writeln('Hash tokenization complete. Total merges: ', MergeCount, '.');
end;

{ Statistics and reports }
// Calculate elapsed symbolization time.
procedure CalculateTimeStatistics;
var
  PauseMS: Int64;
begin
  // StopTime is a TDateTime measured in days.
  PauseMS := Round(StopTime * 86400000.0);

  ElapsedMS := MilliSecondsBetween(t0, t1) - PauseMS;

  if ElapsedMS < 1 then
    ElapsedMS := 1;

  Hours := ElapsedMS div 3600000;
  Mins := (ElapsedMS mod 3600000) div 60000;
  Secs := (ElapsedMS mod 60000) / 1000.0;
end;

// Calculate and report symbol statistics.
procedure SymbolStats;
var
  n, i, j, L, MinLen, MaxLen, SumLen: Integer;
  Lengths, Histogram: TIVector;
  MaxPossibleLen: Integer;
  Median: Single;
begin
  n := Length(SymbolTable);
  if n = 0 then begin
    WriteLn('Symbol table is empty.');
    Exit;
  end;

  Writeln('--- Symbols Statistics ---');
  Writeln('Number of raw byte symbols: ', 256);
  Writeln('Number of special symbols: ', 4);
  if TokenizerKind = UDTokenizer then begin
    Writeln('Number of reserved UD tag slots: ', UDTagBoundary - FirstTagToken);
    Writeln('Number of merged symbols: ', nSymbols - UDTagBoundary);
  end
  else
    Writeln('Number of merged symbols: ', nSymbols - 260);

  // Compute symbol lengths, minimum, maximum, and sum.
  SetLength(Lengths, n);

  MinLen := MaxInt;
  MaxLen := 0;
  SumLen := 0;

  for i := 0 to n - 1 do begin
    L := Length(SymbolTable[i]);  // Byte length.
    Lengths[i] := L;

    if L < MinLen then MinLen := L;
    if L > MaxLen then MaxLen := L;

    SumLen := SumLen + L;
  end;

  // Min / Max.
  WriteLn('Minimum symbol length: ', MinLen);
  WriteLn('Maximum symbol length: ', MaxLen);

  // Histogram.
  MaxPossibleLen := MaxLen;
  SetLength(Histogram, MaxPossibleLen + 1);
  for i := 0 to MaxPossibleLen do
    Histogram[i] := 0;

  for i := 0 to n - 1 do
    Inc(Histogram[Lengths[i]]);

  WriteLn;
  WriteLn('Histogram of symbol lengths:');
  for i := 0 to MaxPossibleLen do
    if Histogram[i] > 0 then
      WriteLn('Length ', i: 2, ': ', Histogram[i]);

  // Sort symbol lengths before calculating the median.
  for i := 1 to n - 1 do begin
    L := Lengths[i];
    j := i - 1;
    while (j >= 0) and (Lengths[j] > L) do begin
      Lengths[j + 1] := Lengths[j];
      Dec(j);
    end;
    Lengths[j + 1] := L;
  end;

  if (n mod 2) = 1 then
    Median := Lengths[n div 2]
  else
    Median := 0.5 * (Lengths[n div 2 - 1] + Lengths[n div 2]);

  WriteLn;
  WriteLn('Mean symbol length: ', SumLen / n: 0: 4);
  WriteLn('Median symbol length: ', Median: 0: 4);
end;

// Count nodes in the token linked list.
function CountTokenNodes(Head: PTokenNode): Integer;
var
  Cur: PTokenNode;
begin
  Result := 0;
  Cur := Head;

  while Cur <> nil do begin
    Inc(Result);
    Cur := Cur^.Next;
  end;
end;

// Report the longest symbol.
procedure ReportSymbolLengths;
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

  if MaxIndex >= 0 then begin
    Writeln('Longest symbol:');
    Writeln('  Index: ', MaxIndex);
    Writeln('  Length: ', MaxLen);
    Writeln('  Value: "', SymbolTable[MaxIndex], '"');
  end;
end;


// Report basic file, time, and corpus statistics.
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
  Writeln('Number of symbols: ', nSymbols);
  Writeln('Corpus size: ', nCorpus, ' bytes');
  if not FromSymbolTable then begin
    Writeln('Bytes processed per second: ', nCorpus / (ElapsedMS / 1000): 6: 4);
    Writeln;
  end;
end;

// Report all symbolization statistics.
procedure ReportStatistics;
begin
  ReportBasicStatistics;
  SymbolStats;
  ReportSymbolLengths;

  if VerboseTokenize and (TextRec(Output).Handle = StdOutputHandle) then
    Pause;
end;

{ Saving symbolization data }
// Save symbolization statistics to a .sym.tok file without changing the current directory.
procedure SaveMetaData(const MetaFileName: String);
var
  SaveOut: Text;
  OutName, OutDir: String;
  Redirected: Boolean;
begin
  OutName := Trim(MetaFileName);

  // If caller passes blank filename, build a default.
  if OutName = '' then begin
    if Trim(WorkingName) <> '' then
      OutName := ChangeFileExt(ExtractFileName(WorkingName), '') + '.sym.tok'
    else
      OutName := 'symbolize.sym.tok';
  end;

  // If caller passed no path, save into LogDir if available.
  if ExtractFilePath(OutName) = '' then begin
    if Trim(LogDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(LogDir) + OutName
    else
      OutName := IncludeTrailingPathDelimiter(GetCurrentDir) + OutName;
  end;

  // Add .sym.tok extension if missing.
  if ExtractFileExt(OutName) = '' then
    OutName := OutName + '.sym.tok';

  OutDir := ExtractFilePath(OutName);

  try
    // Create destination folder, but never call ForceDirectories('').
    if Trim(OutDir) <> '' then
      ForceDirectories(OutDir);

    // Save current console Output.
    SaveOut := Output;
    Redirected := False;

    try
      // Redirect Output to metadata file.
      Assign(Output, OutName);
      Rewrite(Output);
      Redirected := True;

      ReportStatistics;

    finally
      // Always restore console Output.
      if Redirected then
        Close(Output);

      Output := SaveOut;
    end;

    Writeln('File ', OutName, ' successfully saved.');
    // Writeln;

  except
    on E: Exception do begin
      // Try to restore Output even if something failed early.
      Output := SaveOut;

      Writeln('Error saving metadata: ', E.ClassName, ' ', E.Message);
      Writeln('Target metadata file = "', OutName, '"');
      Writeln;
    end;
  end;
end;

// Save the binary merge table without changing the current directory.
procedure SaveMergeTable(const Merges: TMergeArray; MergeFileName: String);
var
  F: file;
  i, n: Integer;
  OutName, OutDir: String;
  FileOpen: Boolean;
begin
  OutName := Trim(MergeFileName);

  // If caller passes blank filename, build a default.
  if OutName = '' then begin
    if Trim(WorkingName) <> '' then
      OutName := WorkingName + '.mer'
    else
      OutName := 'symbolize.mer';
  end;

  // If caller passed no path, save into SymbolDir if available.
  // If you later add MergeDir, use MergeDir here instead.
  if ExtractFilePath(OutName) = '' then begin
    if Trim(MergeDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(MergeDir) + OutName
    else if Trim(SymbolDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(SymbolDir) + OutName
    else
      OutName := IncludeTrailingPathDelimiter(GetCurrentDir) + OutName;
  end;

  // Add .mer extension if missing.
  if ExtractFileExt(OutName) = '' then
    OutName := OutName + '.mer';

  OutDir := ExtractFilePath(OutName);
  FileOpen := False;

  try
    // Create destination folder, but never call ForceDirectories('').
    if Trim(OutDir) <> '' then
      ForceDirectories(OutDir);

    Assign(F, OutName);
    Rewrite(F, 1);
    FileOpen := True;

    n := Length(Merges);
    BlockWrite(F, n, SizeOf(n));

    for i := 0 to n - 1 do begin
      BlockWrite(F, Merges[i].A, SizeOf(Integer));
      BlockWrite(F, Merges[i].B, SizeOf(Integer));
      BlockWrite(F, Merges[i].NewSym, SizeOf(Integer));
    end;

    Close(F);
    FileOpen := False;

    Writeln('File ', OutName, ' successfully saved.');

  except
    on E: Exception do begin
      if FileOpen then
        Close(F);

      Writeln('Error saving merge table: ', E.ClassName, ' ', E.Message);
      Writeln('Target merge file = "', OutName, '"');
    end;
  end;
end;

{ Main symbolization workflow }
// Run symbolization and BPE training.
procedure RunSymbolize(const Corpus: TBVector);
begin
  // Reset for new run.
  MergeCount := 0;
  SetLength(Merges, 0);

  // Timing.
  t0 := Now;       // Start of timing for entire tokenization;
  StopTime := 0;   // Time to subtract from timing.

  // Initialize base byte symbols plus special tokens.
  InitSymbolTable;
  StartSymbol := Length(SymbolTable);
  nSymbols := Length(SymbolTable);

  // Convert byte corpus to initial token list.
  BuildTokenListFromCorpus(Corpus);
  nCorpus := Length(Corpus);

  Writeln('Symbolizing and merging started.');
  Write('Maximum symbols = ', MaxSymbols, '; Initial symbols = ', nSymbols, '; Maximum merges = ', MaxMerges, '; Maximum pair counts = ', MaxPairCount, '; Tokenizer Kind = ', TokenizerKindName(TokenizerKind), '.');
  if TokenizerKind = UDTokenizer then
    Writeln(' UD tags are from ', FirstTagToken, ' to ', UDTagBoundary - 1, '.')
  else
    Writeln;

  TrainBPEHash(Head, Tail, MaxMerges, MaxSymbols, MergeCount, StartSymbol);

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;
  FinalTokenCount := CountTokenNodes(Head);

  // Finish timing before reports or pauses.
  t1 := Now;
  CalculateTimeStatistics;

  // Display the symbol table.
  if VerboseTokenize then
    DisplayByteSymbolTable(SymbolTable);

  // Report statistics.
  if VerboseTokenize then
    ReportStatistics;

  FreeTokenList(Head, Tail);
  Writeln('Symbolizing and merging ended.');
  nMerges := MergeCount;
end;

end.
