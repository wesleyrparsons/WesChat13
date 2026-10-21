unit GPT2Tokenize;

{$mode objfpc}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  { RTL and platform units }
  Classes, Fpjson, Jsonparser, SysUtils,
  { WesChat units }
  Display, Global, WesTokenize;

procedure EnsureGPT2VocabLoaded;
procedure EnsureGPT2MergesLoaded;
procedure RunGPT2TokenizeFile(const FileName: string; var OutTokens: TIVector);
procedure RunGPT2TokenizeString(const InputString: string; var TokenizedCorpus: TIVector);
procedure LoadVocab(const FileName: string; Vocab: TStringList);
procedure LoadMerges(const FileName: string; Merges: TStringList);
function DisplayToken(const S: UnicodeString): AnsiString;
function DecodeGPT2Token(const TokenID: Integer): UnicodeString;
function DecodeGPT2Tokens(const Tokens: TIVector): UnicodeString;

implementation

type
  TRBStringArray = array of RawByteString;
  TUStringArray = array of UnicodeString;

var
  InputBytes: RawByteString;
  EncodedText: UnicodeString;
  GPT2Merges: TStringList = nil;

// Ensure that vocab1.json is loaded.
procedure EnsureGPT2VocabLoaded;
begin
  if Vocab = nil then
    Vocab := TStringList.Create;

  if Vocab.Count = 0 then
    LoadVocab(VocabFileName, Vocab);
end;

// Ensure the merges file is loaded.
procedure EnsureGPT2MergesLoaded;
begin
  if GPT2Merges = nil then
    GPT2Merges := TStringList.Create;

  if GPT2Merges.Count = 0 then
    LoadMerges(MergeFileName, GPT2Merges);
end;

// Load a file of bytes.
function LoadFileRaw(const FileName: string): RawByteString;
var
  FS: TFileStream;
  i: Integer;
begin
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
  Result := '';
  try
    SetLength(Result, FS.Size);
    if FS.Size > 0 then
      FS.ReadBuffer(Result[1], FS.Size);
  finally
    FS.Free;
  end;

 if VeryVerboseTokenize then begin
    Writeln('Display rawbytes read:');
    for i := 500 to high(result) do
      Writeln(i: 5, ord(Result[i]): 5, '   *', Result[i], '* ');
    Readln;
  end;
end;

// Encode bytes to ChatGPT2 unicode.
function GPT2ByteToUnicode(B: Byte): WideChar;
var
  i, n: Integer;
begin
  if ((B >= 33) and (B <= 126)) or
     ((B >= 161) and (B <= 172)) or
      (B >= 174) then begin
    Result := WideChar(B);
    Exit;
  end;

  n := 0;
  for i := 0 to B - 1 do
    if not (((i >= 33) and (i <= 126)) or
            ((i >= 161) and (i <= 172)) or
            ((i >= 174) and (i <= 255))) then
      Inc(n);

  Result := WideChar(256 + n);
end;

// Decode GPT2 tokens.
function GPT2UnicodeToByte(const C: WideChar; out B: Byte): Boolean;
var
  i: Integer;
begin
  for i := 0 to 255 do
    if GPT2ByteToUnicode(Byte(i)) = C then begin
      B := Byte(i);
      Result := True;
      Exit;
    end;

  B := 0;
  Result := False;
end;

function DecodeGPT2Tokens(const Tokens: TIVector): UnicodeString;
var
  Raw: RawByteString;
  PieceUTF8: UTF8String;
  PieceUnicode: UnicodeString;
  i, j, TokenID: Integer;
  B: Byte;
begin
  Raw := '';

  if not Assigned(Vocab) then begin
    Result := '<VOCAB NOT LOADED>';
    Exit;
  end;

  for i := 0 to High(Tokens) do begin
    TokenID := Tokens[i];

    if (TokenID < 0) or (TokenID >= Vocab.Count) then begin
      Raw := Raw + UTF8Encode('<BADTOKEN:' + IntToStr(TokenID) + '>');
      Continue;
    end;

    PieceUTF8 := UTF8String(Vocab[TokenID]);
    PieceUnicode := UTF8Decode(PieceUTF8);

    for j := 1 to Length(PieceUnicode) do begin
      if GPT2UnicodeToByte(PieceUnicode[j], B) then
        Raw := Raw + AnsiChar(B)
      else
        Raw := Raw + UTF8Encode(PieceUnicode[j]);
    end;
  end;

  Result := UTF8Decode(UTF8String(Raw));
end;

function DecodeGPT2Token(const TokenID: Integer): UnicodeString;
var
  OneToken: TIVector;
begin
  SetLength(OneToken, 1);
  OneToken[0] := TokenID;

  Result := DecodeGPT2Tokens(OneToken);
end;

function EncodeBytesToUnicode(const x: RawByteString): UnicodeString;
var
  i: Integer;
begin
  SetLength(Result, Length(x));

  for i := 1 to Length(x) do
    Result[i] := GPT2ByteToUnicode(Byte(x[i]));
end;

function DisplayToken(const S: UnicodeString): AnsiString;
var
  i, cp: Integer;
  b: Byte;
begin
  Result := '';

  for i := 1 to Length(S) do begin
    cp := Ord(S[i]);

    // ASCII printable.
    if (cp >= 32) and (cp <= 126) then
      Result := Result + AnsiChar(cp)

    // GPT2 fallback range.
    else if (cp >= $0100) and (cp <= $01FF) then begin
      b := cp - $0100;

    // One rule for control bytes 0..32
      if b <= 32 then
        Result := Result + Chr(b)
      else
        Result := Result + '?';

    end;
  end;
end;

// New NextToken version, 3/9/2026.
function IsWordChar(ch: WideChar): Boolean;
begin
  Result :=
    ((ch >= 'A') and (ch <= 'Z')) or
    ((ch >= 'a') and (ch <= 'z')) or
    ((ch >= '0') and (ch <= '9'));
end;

function IsPunct(ch: WideChar): Boolean;
begin
  Result := not IsWordChar(ch) and (ch <> ' ') and (ch <> #$0120);
end;

function NextToken(const S: UnicodeString; var idx: Integer): UnicodeString;
var
  start: Integer;
  ch: WideChar;
begin
  Result := '';
  if idx > Length(S) then Exit;

  ch := S[idx];

  { 1. word with leading space marker (Ġ) }
  if ch = #$0120 then begin
    start := idx;
    Inc(idx);

    while (idx <= Length(S)) and IsWordChar(S[idx]) do
      Inc(idx);

    Result := Copy(S, start, idx - start);
    Exit;
  end;

  { 2. plain word }
  if IsWordChar(ch) then begin
    start := idx;

    while (idx <= Length(S)) and IsWordChar(S[idx]) do
      Inc(idx);

    Result := Copy(S, start, idx - start);
    Exit;
  end;

  { 3. punctuation }
  if IsPunct(ch) then begin
    start := idx;

    while (idx <= Length(S)) and IsPunct(S[idx]) do
      Inc(idx);

    Result := Copy(S, start, idx - start);
    Exit;
  end;

  { 4. skip spaces }
  Inc(idx);
end;

procedure DisplayVocab(const a, b: Integer);
var
  i, j: Integer;
  s: string;
begin
 Writeln('Vocab ', a, ' to ', b);
  for i := a to b do begin
    Write(i, ' ', 'Vocab[i]: ', Vocab[i], ' ');
    Write('Raw bytes: ');
    s := Vocab[i];
    for j := 1 to Length(s) do
      Write(Ord(s[j]), ' ');
    Writeln;
  end;
end;

// Load vocab1.json.
procedure LoadVocab(const FileName: string; Vocab: TStringList);
var
  Raw: RawByteString;
  JSON: TJSONData;
  Obj: TJSONObject;
  FS: TFileStream;
  i, TokenID, MaxTokenID: Integer;
begin
  Vocab.Clear;
  Vocab.Sorted := False;
  Vocab.Duplicates := dupAccept;
  Vocab.CaseSensitive := True;
  Vocab.OwnsObjects := False;

  if FileExists(FileName) then begin
    Writeln('Vocabulary file ', FileName, ' found.');
  end
  else begin
    Writeln('ERROR: Vocabulary file not found: ', FileName);
    Exit;
  end;

  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Raw, FS.Size);

    if FS.Size > 0 then
      FS.ReadBuffer(Raw[1], FS.Size);
  finally
    FS.Free;
  end;

  JSON := GetJSON(Raw);
  try
    if not (JSON is TJSONObject) then begin
      Writeln('Invalid vocabulary file JSON.');
      Exit;
    end;

    Obj := TJSONObject(JSON);

    // Find the largest token ID.
    MaxTokenID := -1;

    for i := 0 to Obj.Count - 1 do begin
      TokenID := Obj.Items[i].AsInteger;

      if TokenID > MaxTokenID then
        MaxTokenID := TokenID;
    end;

    // Create one list position for every token ID.
    Vocab.Capacity := MaxTokenID + 1;

    for i := 0 to MaxTokenID do
      Vocab.Add('');

    // Place each token at its actual ID.
    for i := 0 to Obj.Count - 1 do begin
      TokenID := Obj.Items[i].AsInteger;
      if (TokenID >= 0) and (TokenID <= MaxTokenID) then
        Vocab[TokenID] := Obj.Names[i];
    end;

  finally
    Writeln('Vocabulary file ', FileName, ' loaded.');
    JSON.Free;
  end;

  Writeln('End of loading vocabulary. Length of Vocab: ', Vocab.Count, '.');
end;

procedure LoadMerges(const FileName: string; Merges: TStringList);
var
  SL: TStringList;
  Line: string;
  Parts: TRBStringArray;
  i: Integer;
begin
  if not FileExists(FileName) then begin
    Writeln('ERROR: GPT-2 merges file not found: ', ExpandFileName(FileName));
    Pause;
    Halt;
  end;

  Merges.Clear;
  Merges.CaseSensitive := True;
  Merges.Sorted := True;
  Merges.Duplicates := dupIgnore;

  SL := TStringList.Create;
  try
    SL.LoadFromFile(FileName, TEncoding.UTF8);

    for i := 0 to SL.Count - 1 do begin
      Line := Trim(SL[i]);

      if (Line = '') or (Line[1] = '#') then
        Continue;

      Parts := Line.Split([' ']);

      if Length(Parts) = 2 then
        Merges.AddObject(Parts[0] + ' ' + Parts[1],
          TObject(PtrInt(Merges.Count)));
    end;
  finally
    SL.Free;
  end;

  Writeln('End of loading merges. Length of merges: ', Merges.Count, '. Tokeninizing...');
end;

function UTF8CharLen(P: PChar): Integer;
var
  B: Byte;
begin
  B := Byte(P^);
  if B < $80 then Result := 1
  else if (B and $E0) = $C0 then Result := 2
  else if (B and $F0) = $E0 then Result := 3
  else if (B and $F8) = $F0 then Result := 4
  else Result := 1;
end;

procedure BPE(const Word: string; const Merges: TStringList; var Output: TStringArray);
var
  Symbols: TStringArray;
  Pair, BestPair: string;
  i, j, Rank, BestRank: Integer;
  Found: Boolean;
var
  p, len: Integer;
begin
  { 1. Initialize symbols as UTF8 bytes }
  begin
    p := 1;
    SetLength(Symbols, 0);
    while p <= Length(Word) do begin
      len := UTF8CharLen(@Word[p]);
      SetLength(Symbols, Length(Symbols)+1);
      Symbols[High(Symbols)] := Copy(Word, p, len);
      Inc(p, len);
    end;
  end;

  { 2. Main merge loop }
  while True do begin
    BestRank := MaxInt;
    BestPair := '';
    Found := False;

    { 2a. Find the best-ranked mergeable pair }
    for i := 0 to High(Symbols)-1 do begin
      Pair := Symbols[i] + ' ' + Symbols[i+1];   { GPT2 uses space delimiter }

      j := Merges.IndexOf(Pair);
      if j >= 0 then begin
        Rank := PtrInt(Merges.Objects[j]);
        if Rank < BestRank then begin
          BestRank := Rank;
          BestPair := Pair;
          Found := True;
        end;
      end;
    end;

    if not Found then Break;

    { 2b. Apply the best merge everywhere }
    i := 0;
    while i < High(Symbols) do begin
      Pair := Symbols[i] + ' ' + Symbols[i + 1];
      if Pair = BestPair then begin
        Symbols[i] := Symbols[i] + Symbols[i + 1];  { merge into UTF‑8 string }

        { shift left }
        for j := i+1 to High(Symbols) - 1 do
          Symbols[j] := Symbols[j + 1];

        SetLength(Symbols, Length(Symbols) - 1);
      end
      else
        Inc(i);
    end;
  end;

  Output := Symbols;
end;

procedure ShowRaw(const x: string);
var
  j: Integer;
begin
  Write('Raw: ');
  for j := 1 to Length(x) do
    Write(ord(x[j]), ' ');
  Writeln;
end;

// Tokenize a text file.
procedure TokenizeGPT2Text(const Input: RawByteString; const Vocab, Merges: TStringList; var TokenIDs: TIVector);
var
  Words: TUStringArray;
  Pieces: TStringArray;
  tok: UnicodeString;
  i, j, k, idx, Count, iWord: Integer;
begin
  Count := 0;
  SetLength(TokenIDs, 0);

  EncodedText := EncodeBytesToUnicode(Input);

  // Because NextToken needs a leading $0120.
  EncodedText := WideChar($0120) + EncodedText;

  i := 1;
  iWord := 0;
  while True do begin
    tok := NextToken(EncodedText, i);
    if tok = '' then Break;

    SetLength(Words, iWord + 1);
    Words[iWord] := tok;
    Inc(iWord);
  end;

  // Remove the $0120 that was added.
  if (Length(Words) > 0) and (Length(Words[0]) > 0) then
    Words[0] := Copy(Words[0], 2, Length(Words[0]) - 1);

  // Byte-pair encoding.
  for j := 0 to High(Words) do begin
    BPE(UTF8Encode(Words[j]), Merges, Pieces);

    for k := 0 to High(Pieces) do begin
      idx := Vocab.IndexOf(Pieces[k]);

      if idx >= 0 then begin
        Inc(Count);
        SetLength(TokenIDs, Count);
        TokenIDs[Count - 1] := idx;
        if (Count mod TokenProgressInterval) = 0 then
          if not VerboseTokenize and not VeryVerboseTokenize then
            DisplayTokenProgress(Count, j + 1, Length(Words));
      end;
    end;
  end;

  if Length(Words) > 0 then
    if not VerboseTokenize and not VeryVerboseTokenize then
      DisplayTokenProgress(Count, Length(Words), Length(Words));
  Writeln;
end;

procedure RunGPT2TokenizeText(const Input: RawByteString; var OutTokens: TIVector);
begin
  EnsureGPT2VocabLoaded;
  EnsureGPT2MergesLoaded;

  TokenizeGPT2Text(Input, Vocab, GPT2Merges, OutTokens);
end;

procedure TokenizeFile(const Corpus: string; const Vocab, Merges: TStringList; var TokenIDs: TIVector);
begin
  InputBytes := LoadFileRaw(Corpus);
  TokenizeGPT2Text(InputBytes, Vocab, Merges, TokenIDs);
end;

function DecodeToken(const s: UTF8String): UnicodeString;
var
  u: UnicodeString;
  i: Integer;
  c: Word;
begin
  u := UTF8Decode(s);
  Result := '';

  for i := 1 to Length(u) do begin
    c := Ord(u[i]);

    case c of
      $0120: Result := Result + WideChar($0020);  // Ġ -> space
      $010D: Result := Result + WideChar($000D);  // č -> CR
      $010A: Result := Result + WideChar($000A);  // Ċ -> LF
      $0009: Result := Result + WideChar($0009);  // tab
    else
      if c < 128 then
        Result := Result + WideChar(c)
      else if (c >= $0100) and (c <= $01FF) then
        Result := Result + WideChar('?')
      else
        Result := Result + u[i];
    end;
  end;
end;

// Main work flow.
procedure RunGPT2TokenizeFile(const FileName: string; var OutTokens: TIVector);
begin
  EnsureGPT2VocabLoaded;
  EnsureGPT2MergesLoaded;

  TokenizeFile(FileName, Vocab, GPT2Merges, OutTokens);

end;

procedure RunGPT2TokenizeString(const InputString: string; var TokenizedCorpus: TIVector);
begin
  try
    EnsureGPT2VocabLoaded;
    EnsureGPT2MergesLoaded;

    TokenizeGPT2Text(RawByteString(InputString), Vocab, GPT2Merges, TokenizedCorpus);

    finally
      if GPT2Merges <> nil then begin
        GPT2Merges.Free;
        GPT2Merges := nil;
      end;
  end;
end;

end.

