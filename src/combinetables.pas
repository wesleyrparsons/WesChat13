unit CombineTables;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  SysUtils,
  { WesChat units }
  Global, IOHandler;

procedure MergeSymbolTables(out CombinedTable: TSymbolTable);

implementation

type
  TSymbolTableArray = array of TSymbolTable;

{ Symbol-table validation }
// Return True if a symbol table uses the fixed UD reserved-token layout.
function IsUDSymbolTable(const Table: TSymbolTable): Boolean;
begin
  Result := (Length(Table) >= UDTagBoundary) and (Table[TokNoun] = '|noun') and (Table[TokVerb] = '|verb') and (Table[TokPunct] = '|punct');
end;

// Require one symbol to occur at its fixed token ID.
procedure RequireFixedSymbol(const Table: TSymbolTable; const FileName: string; const TokenID: Integer; const Expected: RawByteString);
begin
  if (TokenID < 0) or (TokenID > High(Table)) then
    raise Exception.CreateFmt('Symbol table "%s" is too short for required token ID %d.', [FileName, TokenID]);

  if Table[TokenID] <> Expected then
    raise Exception.CreateFmt('Symbol table "%s" has "%s" at token ID %d; expected "%s".',
      [FileName, string(Table[TokenID]), TokenID, string(Expected)]);
end;

// Validate fixed byte and special-token IDs shared by Wes and UD symbol tables.
procedure ValidateBaseSymbols(const Table: TSymbolTable; const FileName: string);
var
  i: Integer;
begin
  if Length(Table) < TokNUL + 1 then
    raise Exception.CreateFmt('Symbol table "%s" has only %d symbols; at least %d are required.', [FileName, Length(Table), TokNUL + 1]);

  for i := 0 to 255 do
    if Table[i] <> RawByteString(Chr(i)) then
      raise Exception.CreateFmt('Symbol table "%s" does not contain byte %d at token ID %d.', [FileName, i, i]);

  RequireFixedSymbol(Table, FileName, TokBOS, '<BOS>');
  RequireFixedSymbol(Table, FileName, TokEOS, '<EOS>');
  RequireFixedSymbol(Table, FileName, TokPAD, '<PAD>');
  RequireFixedSymbol(Table, FileName, TokNUL, '<UNK>');
end;

// Validate every currently defined UD tag at its fixed token ID.
procedure ValidateUDSymbols(const Table: TSymbolTable; const FileName: string);
begin
  if Length(Table) < UDTagBoundary then
    raise Exception.CreateFmt('UD symbol table "%s" has only %d symbols; at least %d are required.', [FileName, Length(Table), UDTagBoundary]);

  RequireFixedSymbol(Table, FileName, TokNoun, '|noun');
  RequireFixedSymbol(Table, FileName, TokVerb, '|verb');
  RequireFixedSymbol(Table, FileName, TokAdj, '|adj');
  RequireFixedSymbol(Table, FileName, TokAdv, '|adv');
  RequireFixedSymbol(Table, FileName, TokPrep, '|prep');
  RequireFixedSymbol(Table, FileName, TokDet, '|det');
  RequireFixedSymbol(Table, FileName, TokPron, '|pron');
  RequireFixedSymbol(Table, FileName, TokAux, '|aux');
  RequireFixedSymbol(Table, FileName, TokSConj, '|sconj');
  RequireFixedSymbol(Table, FileName, TokCConj, '|cconj');
  RequireFixedSymbol(Table, FileName, TokPart, '|part');
  RequireFixedSymbol(Table, FileName, TokIntj, '|intj');
  RequireFixedSymbol(Table, FileName, TokNum, '|num');
  RequireFixedSymbol(Table, FileName, TokPropn, '|propn');
  RequireFixedSymbol(Table, FileName, TokX, '|x');
  RequireFixedSymbol(Table, FileName, TokSym, '|sym');
  RequireFixedSymbol(Table, FileName, TokPunct, '|punct');

  RequireFixedSymbol(Table, FileName, TokSing, '|sg');
  RequireFixedSymbol(Table, FileName, TokPlur, '|pl');

  RequireFixedSymbol(Table, FileName, TokPerson1, '|1p');
  RequireFixedSymbol(Table, FileName, TokPerson2, '|2p');
  RequireFixedSymbol(Table, FileName, TokPerson3, '|3p');

  RequireFixedSymbol(Table, FileName, TokNom, '|nom');
  RequireFixedSymbol(Table, FileName, TokAcc, '|acc');
  RequireFixedSymbol(Table, FileName, TokGen, '|gen');
  RequireFixedSymbol(Table, FileName, TokDat, '|dat');
  RequireFixedSymbol(Table, FileName, TokLoc, '|loc');
  RequireFixedSymbol(Table, FileName, TokIns, '|ins');
  RequireFixedSymbol(Table, FileName, TokVoc, '|voc');

  RequireFixedSymbol(Table, FileName, TokMasc, '|masc');
  RequireFixedSymbol(Table, FileName, TokFem, '|fem');
  RequireFixedSymbol(Table, FileName, TokNeut, '|neut');
  RequireFixedSymbol(Table, FileName, TokCommon, '|common');

  RequireFixedSymbol(Table, FileName, TokPast, '|past');
  RequireFixedSymbol(Table, FileName, TokPres, '|pres');
  RequireFixedSymbol(Table, FileName, TokFut, '|fut');

  RequireFixedSymbol(Table, FileName, TokMoodInd, '|ind');
  RequireFixedSymbol(Table, FileName, TokMoodImp, '|imp');
  RequireFixedSymbol(Table, FileName, TokMoodSub, '|sub');
  RequireFixedSymbol(Table, FileName, TokMoodCond, '|cond');
  RequireFixedSymbol(Table, FileName, TokMoodOpt, '|opt');

  RequireFixedSymbol(Table, FileName, TokVerbFin, '|fin');
  RequireFixedSymbol(Table, FileName, TokVerbInf, '|inf');
  RequireFixedSymbol(Table, FileName, TokVerbGer, '|ger');
  RequireFixedSymbol(Table, FileName, TokVerbPart, '|participle');
  RequireFixedSymbol(Table, FileName, TokVerbConv, '|conv');

  RequireFixedSymbol(Table, FileName, TokVoiceAct, '|act');
  RequireFixedSymbol(Table, FileName, TokVoicePass, '|pass');
  RequireFixedSymbol(Table, FileName, TokVoiceMid, '|mid');

  RequireFixedSymbol(Table, FileName, TokAspectImp, '|impf');
  RequireFixedSymbol(Table, FileName, TokAspectPerf, '|perf');
  RequireFixedSymbol(Table, FileName, TokAspectProg, '|prog');
  RequireFixedSymbol(Table, FileName, TokAspectProsp, '|prosp');

  RequireFixedSymbol(Table, FileName, TokDegreePos, '|pos');
  RequireFixedSymbol(Table, FileName, TokDegreeCmp, '|cmp');
  RequireFixedSymbol(Table, FileName, TokDegreeSup, '|sup');
  RequireFixedSymbol(Table, FileName, TokDegreeAbs, '|abs');

  RequireFixedSymbol(Table, FileName, TokDefiniteDef, '|def');
  RequireFixedSymbol(Table, FileName, TokDefiniteInd, '|indef');

  RequireFixedSymbol(Table, FileName, TokPronArt, '|art');
  RequireFixedSymbol(Table, FileName, TokPronDem, '|dem');
  RequireFixedSymbol(Table, FileName, TokPronInt, '|int');
  RequireFixedSymbol(Table, FileName, TokPronPrs, '|prs');
  RequireFixedSymbol(Table, FileName, TokPronRel, '|rel');
  RequireFixedSymbol(Table, FileName, TokPronInd, '|indpron');
  RequireFixedSymbol(Table, FileName, TokPronNeg, '|negpron');
  RequireFixedSymbol(Table, FileName, TokPronTot, '|tot');

  RequireFixedSymbol(Table, FileName, TokPoss, '|poss');
  RequireFixedSymbol(Table, FileName, TokRefl, '|refl');

  RequireFixedSymbol(Table, FileName, TokPolarityNeg, '|neg');
  RequireFixedSymbol(Table, FileName, TokPolarityPos, '|positive');

  RequireFixedSymbol(Table, FileName, TokNumCard, '|card');
  RequireFixedSymbol(Table, FileName, TokNumOrd, '|ord');
  RequireFixedSymbol(Table, FileName, TokNumFrac, '|frac');
  RequireFixedSymbol(Table, FileName, TokNumMult, '|mult');
  RequireFixedSymbol(Table, FileName, TokNumSets, '|sets');
  RequireFixedSymbol(Table, FileName, TokNumDist, '|dist');

  RequireFixedSymbol(Table, FileName, TokNumDigit, '|digit');
  RequireFixedSymbol(Table, FileName, TokNumWord, '|numword');
  RequireFixedSymbol(Table, FileName, TokNumRoman, '|roman');

  RequireFixedSymbol(Table, FileName, TokAbbr, '|abbr');
  RequireFixedSymbol(Table, FileName, TokForeign, '|foreign');
  RequireFixedSymbol(Table, FileName, TokTypo, '|typo');

  RequireFixedSymbol(Table, FileName, TokAnim, '|anim');
  RequireFixedSymbol(Table, FileName, TokInan, '|inan');
  RequireFixedSymbol(Table, FileName, TokHuman, '|human');
  RequireFixedSymbol(Table, FileName, TokNonHuman, '|nonhuman');
end;

{ File-list helpers }
// Return True if a filename already contains an absolute path.
function IsAbsoluteFileName(const FileName: string): Boolean;
begin
  Result := (ExtractFileDrive(FileName) <> '') or
            ((Length(FileName) > 0) and ((FileName[1] = '\') or (FileName[1] = '/')));
end;

// Resolve a filename from the list relative to the list file's directory.
function ResolveListedFileName(const ListedName, ListDirectory: string): string;
begin
  if IsAbsoluteFileName(ListedName) then
    Result := ExpandFileName(ListedName)
  else
    Result := ExpandFileName(IncludeTrailingPathDelimiter(ListDirectory) + ListedName);
end;

// Read symbol-table filenames from a text file.
procedure ReadSymbolTableList(const ListFileName: string; out FileNames: TSVector);
var
  F: TextFile;
  Line, ResolvedName, ListDirectory: string;
  Count: Integer;
begin
  SetLength(FileNames, 0);

  if not FileExists(ListFileName) then
    raise Exception.CreateFmt('Symbol-table list file not found: %s', [ListFileName]);

  ListDirectory := ExtractFilePath(ExpandFileName(ListFileName));

  AssignFile(F, ListFileName);
  Reset(F);
  try
    Count := 0;

    while not EOF(F) do begin
      ReadLn(F, Line);
      Line := Trim(Line);

      if Line = '' then Continue;

      if (Length(Line) >= 2) and (Line[1] = '"') and (Line[Length(Line)] = '"') then
        Line := Copy(Line, 2, Length(Line) - 2);

      ResolvedName := ResolveListedFileName(Line, ListDirectory);

      if not FileExists(ResolvedName) then
        raise Exception.CreateFmt('Symbol table listed in "%s" was not found: %s', [ListFileName, ResolvedName]);

      SetLength(FileNames, Count + 1);
      FileNames[Count] := ResolvedName;
      Inc(Count);
    end;

  finally
    CloseFile(F);
  end;

  if Length(FileNames) = 0 then
    raise Exception.CreateFmt('Symbol-table list file "%s" contains no symbol-table filenames.', [ListFileName]);
end;

{ Symbol-table merging }
// Return True if an exact symbol is already present in the combined table.
function SymbolAlreadyPresent(const Table: TSymbolTable; const Symbol: RawByteString): Boolean;
var
  i: Integer;
begin
  for i := 0 to High(Table) do
    if Table[i] = Symbol then begin
      Result := True;
      Exit;
    end;

  Result := False;
end;

// Append one learned symbol unless it is already present.
procedure AddUniqueLearnedSymbol(var CombinedTable: TSymbolTable; const Symbol: RawByteString; var DuplicateCount, EmptyCount: Integer);
var
  NewIndex, CapacityLimit: Integer;
begin
  if Symbol = '' then begin
    Inc(EmptyCount);
    Exit;
  end;

  if SymbolAlreadyPresent(CombinedTable, Symbol) then begin
    Inc(DuplicateCount);
    Exit;
  end;

  CapacityLimit := MaxSymbols;
  if DimVocab < CapacityLimit then CapacityLimit := DimVocab;

  if Length(CombinedTable) >= CapacityLimit then
    raise Exception.CreateFmt('Combined symbol table would exceed the model/tokenizer capacity of %d symbols.', [CapacityLimit]);

  NewIndex := Length(CombinedTable);
  SetLength(CombinedTable, NewIndex + 1);
  CombinedTable[NewIndex] := Symbol;
end;

// Merge symbol tables while preserving all fixed token IDs.
procedure MergeSymbolTables(out CombinedTable: TSymbolTable);
var
  i, j, MergeStart, DuplicateCount, EmptyCount: Integer;
  ListFileName, KindName: string;
  FileNames: TSVector;
  Tables: TSymbolTableArray;
  TableIsUD, CombinedIsUD: Boolean;
begin
  SetLength(CombinedTable, 0);

  Write('Enter name of symbol-table file list: ');
  Readln(ListFileName);
  ListFileName := Trim(ListFileName);

  if ListFileName = '' then
    raise Exception.Create('No symbol-table list filename was entered.');

  ReadSymbolTableList(ListFileName, FileNames);
  SetLength(Tables, Length(FileNames));

  CombinedIsUD := False;

  for i := 0 to High(FileNames) do begin
    LoadSymbolTable(FileNames[i], Tables[i]);
    ValidateBaseSymbols(Tables[i], FileNames[i]);

    TableIsUD := IsUDSymbolTable(Tables[i]);

    if TableIsUD then
      ValidateUDSymbols(Tables[i], FileNames[i]);

    if i = 0 then
      CombinedIsUD := TableIsUD
    else if TableIsUD <> CombinedIsUD then
      raise Exception.Create('Cannot combine Wes and UD symbol tables in one table. Combine tables of the same tokenizer kind.');

    Writeln('  File processed: ', FileNames[i], '; symbols read: ', Length(Tables[i]));
  end;

  // Rebuild the fixed prefix from the first validated table without changing token IDs.
  if CombinedIsUD then begin
    SetLength(CombinedTable, UDTagBoundary);
    for i := 0 to UDTagBoundary - 1 do
      CombinedTable[i] := Tables[0][i];
    MergeStart := UDTagBoundary;
    KindName := 'UD';
  end
  else begin
    SetLength(CombinedTable, TokNUL + 1);
    for i := 0 to TokNUL do
      CombinedTable[i] := Tables[0][i];
    MergeStart := TokNUL + 1;
    KindName := 'Wes';
  end;

  DuplicateCount := 0;
  EmptyCount := 0;

  // Append unique learned symbols in deterministic file-list and original-token-ID order.
  for i := 0 to High(Tables) do
    for j := MergeStart to High(Tables[i]) do
      AddUniqueLearnedSymbol(CombinedTable, Tables[i][j], DuplicateCount, EmptyCount);

  Writeln;
  Writeln('Combined ', KindName, ' symbol table created.');
  Writeln('  Input tables: ', Length(Tables));
  Writeln('  Fixed symbols: ', MergeStart);
  Writeln('  Learned symbols: ', Length(CombinedTable) - MergeStart);
  Writeln('  Duplicate symbols skipped: ', DuplicateCount);
  if EmptyCount > 0 then
    Writeln('  Empty symbols skipped: ', EmptyCount);
  Writeln('  Total symbols: ', Length(CombinedTable));

  if CombinedIsUD then
    Writeln('Save this table with a _ud.sym suffix so the main program can identify it as a UD table.')
  else
    Writeln('Save this table with a _w.sym or _wes.sym suffix so the main program can identify it as a Wes table.');

  Writeln('Retokenize the corpus before using this combined table; existing token lists and models use different learned-token IDs.');
end;

end.
