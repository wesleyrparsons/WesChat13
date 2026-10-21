unit UDTag;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com.}

interface

uses
  { RTL and platform units }
  Classes, Pipes, Process, SysUtils;

{ Public routines }

// Convert an entire corpus file to UD-prefixed text.
procedure UDTagFile(const UDPipeFile, ModelFile, InputFile, OutputFileName: string);

// Convert a string to UD-prefixed text, especially for inference queries.
function UDTagText(const UDPipeFile, ModelFile, InputText: string): string;

implementation

{ UD tag mapping }

// Map a Universal Dependencies part-of-speech value to a WesChat tag.
function MapPOS(const UPOS: string): string;
begin
  case UPOS of
    'NOUN':  Result := 'noun';
    'VERB':  Result := 'verb';
    'ADJ':   Result := 'adj';
    'ADV':   Result := 'adv';
    'ADP':   Result := 'prep';
    'DET':   Result := 'det';
    'PRON':  Result := 'pron';
    'AUX':   Result := 'aux';
    'SCONJ': Result := 'sconj';
    'CCONJ': Result := 'cconj';
    'PART':  Result := 'part';
    'INTJ':  Result := 'intj';
    'NUM':   Result := 'num';
    'PROPN': Result := 'propn';
    'X':     Result := 'x';
    'SYM':   Result := 'sym';
    'PUNCT': Result := 'punct';
  else
    Result := LowerCase(UPOS);
  end;
end;

// Map one Universal Dependencies morphological feature to a WesChat tag.
function MapFeat(const F: string): string;
begin
  Result := '';

  // Number.
  if F = 'Number=Sing' then Result := 'sg'
  else if F = 'Number=Plur' then Result := 'pl'

  // Person.
  else if F = 'Person=1' then Result := '1p'
  else if F = 'Person=2' then Result := '2p'
  else if F = 'Person=3' then Result := '3p'

  // Case.
  else if F = 'Case=Nom' then Result := 'nom'
  else if F = 'Case=Acc' then Result := 'acc'
  else if F = 'Case=Gen' then Result := 'gen'
  else if F = 'Case=Dat' then Result := 'dat'
  else if F = 'Case=Loc' then Result := 'loc'
  else if F = 'Case=Ins' then Result := 'ins'
  else if F = 'Case=Voc' then Result := 'voc'

  // Gender.
  else if F = 'Gender=Masc' then Result := 'masc'
  else if F = 'Gender=Fem' then Result := 'fem'
  else if F = 'Gender=Neut' then Result := 'neut'
  else if F = 'Gender=Com' then Result := 'common'

  // Tense.
  else if F = 'Tense=Past' then Result := 'past'
  else if F = 'Tense=Pres' then Result := 'pres'
  else if F = 'Tense=Fut' then Result := 'fut'

  // Mood.
  else if F = 'Mood=Ind' then Result := 'ind'
  else if F = 'Mood=Imp' then Result := 'imp'
  else if F = 'Mood=Sub' then Result := 'sub'
  else if F = 'Mood=Cnd' then Result := 'cond'
  else if F = 'Mood=Opt' then Result := 'opt'

  // Verb form.
  else if F = 'VerbForm=Fin' then Result := 'fin'
  else if F = 'VerbForm=Inf' then Result := 'inf'
  else if F = 'VerbForm=Ger' then Result := 'ger'
  else if F = 'VerbForm=Part' then Result := 'participle'
  else if F = 'VerbForm=Conv' then Result := 'conv'

  // Voice.
  else if F = 'Voice=Act' then Result := 'act'
  else if F = 'Voice=Pass' then Result := 'pass'
  else if F = 'Voice=Mid' then Result := 'mid'

  // Aspect.
  else if F = 'Aspect=Imp' then Result := 'impf'
  else if F = 'Aspect=Perf' then Result := 'perf'
  else if F = 'Aspect=Prog' then Result := 'prog'
  else if F = 'Aspect=Prosp' then Result := 'prosp'

  // Degree.
  else if F = 'Degree=Pos' then Result := 'pos'
  else if F = 'Degree=Cmp' then Result := 'cmp'
  else if F = 'Degree=Sup' then Result := 'sup'
  else if F = 'Degree=Abs' then Result := 'abs'

  // Definiteness.
  else if F = 'Definite=Def' then Result := 'def'
  else if F = 'Definite=Ind' then Result := 'indef'

  // Pronoun type.
  else if F = 'PronType=Art' then Result := 'art'
  else if F = 'PronType=Dem' then Result := 'dem'
  else if F = 'PronType=Int' then Result := 'int'
  else if F = 'PronType=Prs' then Result := 'prs'
  else if F = 'PronType=Rel' then Result := 'rel'
  else if F = 'PronType=Ind' then Result := 'indpron'
  else if F = 'PronType=Neg' then Result := 'negpron'
  else if F = 'PronType=Tot' then Result := 'tot'

  // Possessive.
  else if F = 'Poss=Yes' then Result := 'poss'

  // Reflexive.
  else if F = 'Reflex=Yes' then Result := 'refl'

  // Polarity.
  else if F = 'Polarity=Neg' then Result := 'neg'
  else if F = 'Polarity=Pos' then Result := 'positive'

  // Numeral type.
  else if F = 'NumType=Card' then Result := 'card'
  else if F = 'NumType=Ord' then Result := 'ord'
  else if F = 'NumType=Frac' then Result := 'frac'
  else if F = 'NumType=Mult' then Result := 'mult'
  else if F = 'NumType=Sets' then Result := 'sets'
  else if F = 'NumType=Dist' then Result := 'dist'

  // Numeral form.
  else if F = 'NumForm=Digit' then Result := 'digit'
  else if F = 'NumForm=Word' then Result := 'numword'
  else if F = 'NumForm=Roman' then Result := 'roman'

  // Miscellaneous lexical properties.
  else if F = 'Abbr=Yes' then Result := 'abbr'
  else if F = 'Foreign=Yes' then Result := 'foreign'
  else if F = 'Typo=Yes' then Result := 'typo'

  // Animacy.
  else if F = 'Animacy=Anim' then Result := 'anim'
  else if F = 'Animacy=Inan' then Result := 'inan'
  else if F = 'Animacy=Hum' then Result := 'human'
  else if F = 'Animacy=Nhum' then Result := 'nonhuman';
end;

// Convert one word and its UD information to prefixed WesChat tag text.
function ConvertTag(const Word, UPOS, Feats: string; UnknownFeatures: TStringList): string;
var
  Parts: TStringList;
  i: Integer;
  Mapped, TagPrefix: string;
begin
  // Put all UD information before the word.
  TagPrefix := '|' + MapPOS(UPOS);

  if (Feats <> '') and (Feats <> '_') then begin
    Parts := TStringList.Create;
    try
      Parts.Delimiter := '|';
      Parts.StrictDelimiter := True;
      Parts.DelimitedText := Feats;

      for i := 0 to Parts.Count - 1 do begin
        Mapped := MapFeat(Parts[i]);

        if Mapped <> '' then
          TagPrefix := TagPrefix + '|' + Mapped
        else if Parts[i] <> '' then
          UnknownFeatures.Add(Parts[i]);
      end;

    finally
      Parts.Free;
    end;
  end;

  Result := TagPrefix + ' ' + Word;
end;

{ CoNLL-U conversion }

// Return True for a normal integer CoNLL-U token ID.
function IsNormalTokenID(const S: string): Boolean;
var
  TokenID: Integer;
begin
  // Reject multiword rows such as 1-2 and empty nodes such as 3.1.
  Result := TryStrToInt(S, TokenID);
end;

// Convert one normal CoNLL-U token line to prefixed UD text.
function ConvertConlluTokenLine(const Line: string; Cols, UnknownFeatures: TStringList; out TaggedToken: string): Boolean;
var
  Word, UPOS, Feats: string;
begin
  Result := False;
  TaggedToken := '';

  if Line = '' then
    Exit;

  if Line[1] = '#' then
    Exit;

  Cols.DelimitedText := Line;

  if Cols.Count < 6 then
    Exit;

  if not IsNormalTokenID(Cols[0]) then
    Exit;

  Word := Cols[1];
  UPOS := Cols[3];
  Feats := Cols[5];

  TaggedToken := ConvertTag(Word, UPOS, Feats, UnknownFeatures);
  Result := True;
end;

// Convert one CoNLL-U line and write it to the tagged corpus file.
procedure ProcessConlluLineToFile(const Line: string; var OutputFile: TextFile; Cols, UnknownFeatures: TStringList; var FirstToken: Boolean);
var
  TaggedToken: string;
begin
  // A blank CoNLL-U line marks a sentence boundary.
  if Line = '' then begin
    if not FirstToken then begin
      WriteLn(OutputFile);
      FirstToken := True;
    end;

    Exit;
  end;

  if not ConvertConlluTokenLine(Line, Cols, UnknownFeatures, TaggedToken) then
    Exit;

  if not FirstToken then
    Write(OutputFile, ' ');

  Write(OutputFile, TaggedToken);
  FirstToken := False;
end;

// Convert CoNLL-U text in memory to prefixed UD text.
function ConvertConlluText(const ConlluText: string; UnknownFeatures: TStringList): string;
var
  Lines, Cols: TStringList;
  i: Integer;
  Sentence, TaggedToken: string;
begin
  Result := '';
  Sentence := '';

  Lines := TStringList.Create;
  Cols := TStringList.Create;

  try
    Lines.Text := ConlluText;

    Cols.Delimiter := #9;
    Cols.StrictDelimiter := True;

    for i := 0 to Lines.Count - 1 do begin

      if Lines[i] = '' then begin
        if Sentence <> '' then begin
          if Result <> '' then
            Result := Result + LineEnding;

          Result := Result + Sentence;
          Sentence := '';
        end;

        Continue;
      end;

      if ConvertConlluTokenLine(Lines[i], Cols, UnknownFeatures, TaggedToken) then begin
        if Sentence <> '' then
          Sentence := Sentence + ' ';

        Sentence := Sentence + TaggedToken;
      end;
    end;

    if Sentence <> '' then begin
      if Result <> '' then
        Result := Result + LineEnding;

      Result := Result + Sentence;
    end;

  finally
    Cols.Free;
    Lines.Free;
  end;
end;

// Report morphological features that WesChat does not recognize.
procedure PrintUnknownFeatures(UnknownFeatures: TStringList);
var
  i: Integer;
begin
  if UnknownFeatures.Count = 0 then begin
    WriteLn('All UDPipe morphological features were recognized.');
    Exit;
  end;

  WriteLn;
  WriteLn('Unrecognized UDPipe morphological features:');

  for i := 0 to UnknownFeatures.Count - 1 do
    WriteLn('  ', UnknownFeatures[i]);

  WriteLn('These features were omitted from the tagged output.');
end;

{ UDPipe helpers }

// Verify that the UDPipe executable and language model are available.
procedure CheckUDPipeFiles(const UDPipeFile, ModelFile: string);
begin
  if not FileExists(UDPipeFile) then
    raise Exception.CreateFmt('UDPipe executable not found: %s', [UDPipeFile]);

  if not FileExists(ModelFile) then
    raise Exception.CreateFmt('UDPipe model not found: %s', [ModelFile]);
end;

// Append all currently available pipe data to a byte string.
procedure AddPipeData(Stream: TInputPipeStream; var S: RawByteString);
const
  BufferSize = 65536;
var
  Buffer: array[0..BufferSize - 1] of Byte;
  BytesRead: LongInt;
  Chunk: RawByteString;
begin
  while Stream.NumBytesAvailable > 0 do begin
    BytesRead := Stream.Read(Buffer, SizeOf(Buffer));

    if BytesRead <= 0 then Exit;

    SetString(Chunk, PAnsiChar(@Buffer[0]), BytesRead);
    S := S + Chunk;
  end;
end;

{ File tagging }

// Run UDPipe on a corpus file and save the prefixed UD representation.
procedure UDTagFile(const UDPipeFile, ModelFile, InputFile, OutputFileName: string);
const
  BufferSize = 65536;
var
  P: TProcess;
  OutputFile: TextFile;
  Cols, UnknownFeatures: TStringList;
  Buffer: array[0..BufferSize - 1] of Byte;
  BytesRead: LongInt;
  Chunk, Pending, ErrorText: RawByteString;
  Line: RawByteString;
  LinePos: SizeInt;
  FirstToken: Boolean;
begin
  CheckUDPipeFiles(UDPipeFile, ModelFile);

  if not FileExists(InputFile) then
    raise Exception.CreateFmt('Input corpus not found: %s', [InputFile]);

  P := TProcess.Create(nil);
  Cols := TStringList.Create;
  UnknownFeatures := TStringList.Create;

  try
    Cols.Delimiter := #9;
    Cols.StrictDelimiter := True;

    UnknownFeatures.Sorted := True;
    UnknownFeatures.Duplicates := dupIgnore;

    AssignFile(OutputFile, OutputFileName);
    Rewrite(OutputFile);

    try
      P.Executable := UDPipeFile;
      P.Parameters.Add('--tokenize');
      P.Parameters.Add('--tag');
      P.Parameters.Add('--output=conllu');
      P.Parameters.Add(ModelFile);
      P.Parameters.Add(InputFile);
      P.Options := [poUsePipes];

      P.Execute;

      Pending := '';
      ErrorText := '';
      FirstToken := True;

      while P.Running or (P.Output.NumBytesAvailable > 0) or (P.Stderr.NumBytesAvailable > 0) do begin

        if P.Output.NumBytesAvailable > 0 then begin
          BytesRead := P.Output.Read(Buffer, SizeOf(Buffer));

          if BytesRead > 0 then begin
            SetString(Chunk, PAnsiChar(@Buffer[0]), BytesRead);
            Pending := Pending + Chunk;

            repeat
              LinePos := Pos(#10, Pending);

              if LinePos = 0 then
                Break;

              Line := Copy(Pending, 1, LinePos - 1);
              Delete(Pending, 1, LinePos);

              if (Length(Line) > 0) and (Line[Length(Line)] = #13) then
                Delete(Line, Length(Line), 1);

              ProcessConlluLineToFile(string(Line), OutputFile, Cols, UnknownFeatures, FirstToken);
            until False;
          end;
        end;

        AddPipeData(P.Stderr, ErrorText);

        if (P.Output.NumBytesAvailable = 0) and (P.Stderr.NumBytesAvailable = 0) and P.Running then
          Sleep(1);
      end;

      if Pending <> '' then begin
        if (Length(Pending) > 0) and (Pending[Length(Pending)] = #13) then
          Delete(Pending, Length(Pending), 1);

        ProcessConlluLineToFile(string(Pending), OutputFile, Cols, UnknownFeatures, FirstToken);
      end;

      if not FirstToken then
        WriteLn(OutputFile);

      P.WaitOnExit;

      if P.ExitStatus <> 0 then
        raise Exception.CreateFmt('UDPipe terminated with error code %d. %s', [P.ExitStatus, Trim(string(ErrorText))]);

    finally
      CloseFile(OutputFile);
    end;

    PrintUnknownFeatures(UnknownFeatures);

  finally
    UnknownFeatures.Free;
    Cols.Free;
    P.Free;
  end;
end;

{ Text tagging }

// Run UDPipe on an in-memory string and return the prefixed UD representation.
function UDTagText(const UDPipeFile, ModelFile, InputText: string): string;
var
  P: TProcess;
  UnknownFeatures: TStringList;
  RawOutput, ErrorText: RawByteString;
begin
  Result := '';

  if InputText = '' then
    Exit;

  CheckUDPipeFiles(UDPipeFile, ModelFile);

  P := TProcess.Create(nil);
  UnknownFeatures := TStringList.Create;

  try
    UnknownFeatures.Sorted := True;
    UnknownFeatures.Duplicates := dupIgnore;

    P.Executable := UDPipeFile;
    P.Parameters.Add('--tokenize');
    P.Parameters.Add('--tag');
    P.Parameters.Add('--output=conllu');
    P.Parameters.Add(ModelFile);

    // No input filename: UDPipe reads the query from stdin.
    P.Options := [poUsePipes];

    P.Execute;

    // Send the query to UDPipe through stdin.
    if Length(InputText) > 0 then
      P.Input.WriteBuffer(InputText[1], Length(InputText));

    // Important: tell UDPipe that no more input is coming.
    P.CloseInput;

    RawOutput := '';
    ErrorText := '';

    while P.Running or (P.Output.NumBytesAvailable > 0) or (P.Stderr.NumBytesAvailable > 0) do begin

      AddPipeData(P.Output, RawOutput);
      AddPipeData(P.Stderr, ErrorText);

      if (P.Output.NumBytesAvailable = 0) and (P.Stderr.NumBytesAvailable = 0) and P.Running then
        Sleep(1);
    end;

    P.WaitOnExit;

    if P.ExitStatus <> 0 then
      raise Exception.CreateFmt('UDPipe terminated with error code %d. %s', [P.ExitStatus, Trim(string(ErrorText))]);

    Result := ConvertConlluText(string(RawOutput), UnknownFeatures);

    if UnknownFeatures.Count > 0 then
      PrintUnknownFeatures(UnknownFeatures);

  finally
    UnknownFeatures.Free;
    P.Free;
  end;
end;

end.
