unit Infer;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  Classes, Math, SysUtils,
  { WesChat units }
  Display, Global, GPT2Tokenize, Matrix, OutputHead, TransformForward, UDTag, Util, WesTokenize;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);

implementation

{ UD output formatting }
// Return displayable text for one space-delimited portion of prefix-tagged UD text.
function GetUDWord(const TaggedToken: string): string;
begin
  if TaggedToken = '' then begin
    Result := '';
    Exit;
  end;

  // A literal vertical bar is ordinary text.
  if TaggedToken = '|' then begin
    Result := '|';
    Exit;
  end;

  // Prefix UD tag clusters are not displayed.
  if TaggedToken[1] = '|' then begin
    Result := '';
    Exit;
  end;

  // Ordinary lexical text is displayed unchanged.
  Result := TaggedToken;
end;

// True if Word should not have a space before it.
function UDNoSpaceBefore(const Word: string): Boolean;
begin
  Result := (Word = '.') or (Word = ',') or (Word = ';') or (Word = ':') or (Word = '!') or (Word = '?') or (Word = '%') or (Word = ')') or (Word = ']') or (Word = '}') or (Word = '…') or

  // English contractions produced as separate UD tokens.
    (Word = 'n''t') or (Word = '''s') or (Word = '''re') or (Word = '''ve') or (Word = '''ll') or (Word = '''d') or (Word = '''m');
end;

// True if the following word should not have a space before it.
function UDNoSpaceAfter(const Word: string): Boolean;
begin
  Result := (Word = '(') or (Word = '[') or (Word = '{') or (Word = '$') or (Word = '£') or (Word = '€') or (Word = '“') or (Word = '‘');
end;

// Strip UD tags from one line and restore normal word spacing.
function StripUDTagsFromLine(const Line: string): string;
var
  i, StartPos: Integer;
  TaggedToken, Word, PreviousWord: string;
  DoubleQuoteOpen: Boolean;

  procedure AddWord(const AWord: string);
  begin
    if AWord = '' then Exit;

    // Straight double quote needs opening/closing context.
    if AWord = '"' then begin
      if DoubleQuoteOpen then begin
        // Closing quote: no space before it.
        Result := Result + AWord;
        DoubleQuoteOpen := False;
      end
      else begin
        // Opening quote.
        if (Result <> '') and not UDNoSpaceAfter(PreviousWord) then
          Result := Result + ' ';

        Result := Result + AWord;
        DoubleQuoteOpen := True;
      end;

      PreviousWord := AWord;
      Exit;
    end;

    // Curly closing quotes never get a space before them.
    if (AWord = '”') or (AWord = '’') then begin
      Result := Result + AWord;
      PreviousWord := AWord;
      Exit;
    end;

    if Result = '' then
      Result := AWord
    else if UDNoSpaceBefore(AWord) then
      Result := Result + AWord
    else if UDNoSpaceAfter(PreviousWord) then
      Result := Result + AWord
    else if DoubleQuoteOpen and (PreviousWord = '"') then
      Result := Result + AWord
    else
      Result := Result + ' ' + AWord;

    PreviousWord := AWord;
  end;

begin
  Result := '';
  PreviousWord := '';
  DoubleQuoteOpen := False;
  i := 1;

  while i <= Length(Line) do begin

    // Skip spaces.
    while (i <= Length(Line)) and (Line[i] = ' ') do
      Inc(i);

    if i > Length(Line) then Break;

    StartPos := i;

    while (i <= Length(Line)) and (Line[i] <> ' ') do
      Inc(i);

    TaggedToken := Copy(Line, StartPos, i - StartPos);
    Word := GetUDWord(TaggedToken);
    AddWord(Word);
  end;
end;

// Strip UD tags from possibly multi-line tagged text.
function StripUDTagsFromText(const TaggedText: string): string;
var
  Lines: TStringList;
  i: Integer;
  S: string;
begin
  Result := '';

  Lines := TStringList.Create;
  try
    Lines.Text := TaggedText;

    for i := 0 to Lines.Count - 1 do begin
      S := StripUDTagsFromLine(Lines[i]);

      if i > 0 then
        Result := Result + LineEnding;

      Result := Result + S;
    end;

  finally
    Lines.Free;
  end;
end;

// Reconstruct Wes-token output, remove UD tags, and display normal text.
procedure WriteUDInferenceTokens(const Tokens: TIVector);
var
  i, Tok: Integer;
  TaggedText, DisplayText: string;
begin
  TaggedText := '';

  // Reconstruct exactly what the Wes tokenizer represents.
  for i := 0 to High(Tokens) do begin
    Tok := Tokens[i];

    if (Tok = BOS) or (Tok = EOS) or (Tok = PAD) or (Tok = UNK) then Continue;

    if (Tok >= 0) and (Tok < Length(SymbolTable)) then
      TaggedText := TaggedText + SymbolTable[Tok];
  end;

  DisplayText := StripUDTagsFromText(TaggedText);
  Write(UTF8Encode(ConsoleText(UTF8Decode(DisplayText))));
end;

{ Token decoding and display }

// Decode one token for inference diagnostics.
function DecodeInferenceToken(const TokenID: Integer): UnicodeString;
begin
  case TokenizerKind of
    WesTokenizer, UDTokenizer:
      Result := Decode(TokenID);

    GPT2Tokenizer:
      Result := DecodeGPT2Token(TokenID);
  else
    Result := 'BAD TOKEN';
  end;
end;

procedure WriteInferenceTokens(const Tokens: TIVector);
var
  i: Integer;
  S: UnicodeString;
begin
  S := '';

  case TokenizerKind of
    WesTokenizer, UDTokenizer:
      for i := 0 to High(Tokens) do
        S := S + Decode(Tokens[i]);

    GPT2Tokenizer:
      S := DecodeGPT2Tokens(Tokens);
  end;

  Write(UTF8Encode(ConsoleText(S)));
end;

{ Sampling and diagnostics }
// Compute probability entropy in nats.
function ProbabilityEntropy(const Probs: array of Single; const Count: Integer): Double;
var
  i: Integer;
  P: Double;
begin
  Result := 0.0;

  for i := 0 to Count - 1 do begin
    P := Probs[i];

    if P > 0.0 then
      Result := Result - P * Ln(P);
  end;
end;

// Data reporting for inference.
procedure ReportInferenceDiagnostics(const Tok: Integer; const RawProb, AdjProb,
  SampleProb, Top1Prob, Top2Prob, TopKMass, Entropy, EOSProb: Double);
begin
  Write('Token = ', Tok, ' "', UTF8Encode(DecodeInferenceToken(Tok)), '"    RawP = ', RawProb: 0: 6, ' AdjP = ', AdjProb: 0: 6, ' SampP = ', SampleProb: 0: 6);
  Writeln('    Top1 = ', Top1Prob:0:6, ' Top2 = ', Top2Prob:0:6, ' Top5Mass = ', TopKMass:0:6, ' Entropy = ', Entropy:0:4, ' EOS = ', EOSProb:0:6);
end;

// Sample the top probs.
function SampleTopK(const TopTokVector: array of Integer; const TopProbVector: array of Single; out TopTok: Integer; out TopKSampleProb: Single): Integer;
var
  i, LastValid: Integer;
  Total, R, Accum: Double;
begin
  TopTok := EOS;
  TopKSampleProb := 0.0;
  Result := EOS;

  if (Length(TopTokVector) = 0) or (Length(TopProbVector) = 0) then Exit;

  if Length(TopTokVector) <> Length(TopProbVector) then begin
    Writeln('SampleTopK: token and probability vector lengths differ.');
    Exit;
  end;

  Total := 0.0;
  LastValid := -1;

  // Sum the valid top-K probabilities.
  for i := 0 to High(TopProbVector) do begin
    if (TopTokVector[i] < 0) or (TopProbVector[i] <= 0.0) then Continue;

    Total := Total + TopProbVector[i];
    LastValid := i;
  end;

  // No valid candidate.
  if (Total <= 0.0) or (LastValid < 0) then Exit;

  R := Random * Total;
  Accum := 0.0;

  // Select according to the probabilities within the top-K set.
  for i := 0 to High(TopProbVector) do begin
    if (TopTokVector[i] < 0) or (TopProbVector[i] <= 0.0) then Continue;

    Accum := Accum + TopProbVector[i];

    if R <= Accum then begin
      TopTok := TopTokVector[i];

      // Normalized probability within the top-K sampling pool.
      TopKSampleProb := TopProbVector[i] / Total;

      Result := TopTok;
      Exit;
    end;
  end;

  // Floating-point fallback.
  TopTok := TopTokVector[LastValid];
  TopKSampleProb := TopProbVector[LastValid] / Total;
  Result := TopTok;
end;

{ Inference input construction }
// Build a padded inference input vector from the most recent context tokens.
procedure BuildInferenceInputTokens(var InputTokens: TIDimVector; const QueryTokenized: TIVector; const SeqLen: Integer; out LastPos: Integer);
var
  i, CopyLen, SrcStart: Integer;
begin
  // Fill everything with PAD.
  for i := 0 to SeqLen - 1 do
    InputTokens[i] := PAD;

  if Length(QueryTokenized) >= SeqLen then begin
    // Use most recent SeqLen tokens.
    SrcStart := Length(QueryTokenized) - SeqLen;
    CopyLen := SeqLen;
    LastPos := SeqLen - 1;
  end
  else begin
    SrcStart := 0;
    CopyLen := Length(QueryTokenized);
    LastPos := CopyLen - 1;
  end;

  for i := 0 to CopyLen - 1 do
    InputTokens[i] := QueryTokenized[SrcStart + i];
end;

{ Single-token inference }
// Run one forward pass and sample one new token.
procedure InferOneToken(var WModelParams: TWModelParams; var WModelState: TWModelState; const Step: Integer; const QueryTokenized: TIVector; var QueryToken: Integer; var AdjustedProb: Single);
const
  Scale = Sqrt(ModelDim);         // Transformer-style embedding scaling by sqrt(d_model).
  KSample = 5;                    // Top n probable tokens.
var
  j, Blk, LastPos, BestTok, TopTok: Integer;
  TopTokVector: array[0..KSample - 1] of Integer;
  TopProbVector: array[0..KSample - 1] of Single;
  TopKSampleProb, RawProb, AdjProb, RawEOSProb, Top1Prob, Top2Prob: Single;
  TopKMass, Entropy: Double;
  RawProbs: array of Single;

  procedure TopKProbs;
  var
    t, p, j: Integer;
  begin
    // Initialize.
    for t := 0 to KSample - 1 do begin
      TopTokVector[t] := -1;
      TopProbVector[t] := -1.0;
    end;

    // Find top K.
    for j := 0 to nVocab - 1 do with WModelState do begin
      for p := 0 to KSample - 1 do
        if Probs[LastPos, j] > TopProbVector[p] then begin

          // Shift down.
          for t := KSample - 1 downto p + 1 do begin
            TopTokVector[t] := TopTokVector[t - 1];
            TopProbVector[t] := TopProbVector[t - 1];
          end;

          TopTokVector[p] := j;
          TopProbVector[p] := Probs[LastPos, j];
          Break;
        end;
    end;

    // Optionally display the top picks.
    if VerboseInfer then begin
      Write('Top probability candidates: ');
      for j := 0 to KSample - 1 do begin
        Write(TopProbVector[j]: 9: 7, ' ', TopTokVector[j], ' ');

        if TopTokVector[j] >= 0 then
          Write(UTF8Encode(DecodeInferenceToken(TopTokVector[j])))
        else
          Write('BADTOKEN');
        Write('    ');
      end;

      Writeln;
    end;
    Top1Prob := TopProbVector[0];
    Top2Prob := TopProbVector[1];

    TopKMass := 0.0;
    for j := 0 to KSample - 1 do
      if TopProbVector[j] > 0.0 then
        TopKMass := TopKMass + TopProbVector[j];
  end;

begin
  if VerboseTransform then with WModelParams do begin
    cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display Embeddings.Value prior to Transform.', Embeddings.Value, B);
  end;

  with WModelState do begin
    // Build the input vector.
    BuildInferenceInputTokens(InputTokens, QueryTokenized, SeqLen, LastPos);

    if VerboseInfer then begin
      Write('Step ', Step, '. Context tokens = ', Length(QueryTokenized), '.');
      Write(' Last 20 InputTokens: ');
      for j := Max(0, LastPos - 19) to LastPos do
        Write(InputTokens[j], ' ');
      Writeln;
    end;

    for j := 0 to LastPos do
        if (InputTokens[j] < 0) or (InputTokens[j] >= nVocab) then
          Writeln('BAD TOKEN at ', j, ': ', InputTokens[j], '. nVocab = ', nVocab, '.');

    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    // Build X only for block 0.
    LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);

    // Scale only block 0 input.
    // Optional transformer-style embedding scaling by sqrt(d_model).
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

    // Forward pass through stacked transformer blocks.
    for Blk := 0 to nBlock - 1 do begin
      if VerboseTransform then begin
        cudaMemcpy(@StateBlock[Blk].X.Value[0, 0], StateBlock[Blk].X.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X.Value before transform.', StateBlock[Blk].X.Value, B);
      end;

      RunTransformForward(WModelParams, WModelState, Blk);

      // Feed this block's output into the next block's input.
      if Blk < nBlock - 1 then
        cudaMemcpy(WModelState.StateBlock[Blk + 1].X.dValue, WModelState.StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    // Compute vocabulary probabilities.
    RunOutputForward(WModelParams, WModelState);
    cudaMemcpy(@Probs[LastPos, 0], dProbs + LastPos * DimVocab, nVocab * SizeOf(Single), cudaMemcpyDeviceToHost);

    SetLength(RawProbs, nVocab);
    for j := 0 to nVocab - 1 do
      RawProbs[j] := Probs[LastPos, j];
    Entropy := ProbabilityEntropy(RawProbs, nVocab);
    if (EOS >= 0) and (EOS < nVocab) then
      RawEOSProb := RawProbs[EOS]
    else
      RawEOSProb := 0.0;

    if DebugCudaChecks then
      CheckCudaError('Copy final probability row for inference.');

    // Suppress non-generative special tokens while leaving EOS available.
    case TokenizerKind of
      WesTokenizer, UDTokenizer: begin
        if (BOS >= 0) and (BOS < nVocab) then Probs[LastPos, BOS] := 0.0;
        if (PAD >= 0) and (PAD < nVocab) then Probs[LastPos, PAD] := 0.0;
        if (UNK >= 0) and (UNK < nVocab) then Probs[LastPos, UNK] := 0.0;
      end;

      GPT2Tokenizer: begin
        if (GPT2BOS >= 0) and (GPT2BOS < nVocab) then Probs[LastPos, GPT2BOS] := 0.0;
        if (GPT2PAD >= 0) and (GPT2PAD < nVocab) then Probs[LastPos, GPT2PAD] := 0.0;
        if (GPT2UNK >= 0) and (GPT2UNK < nVocab) then Probs[LastPos, GPT2UNK] := 0.0;
      end;
    end;

    // Discourage immediate repetition before finding the top K.
    if Length(QueryTokenized) > 0 then begin
      j := QueryTokenized[High(QueryTokenized)];
      if (j >= 0) and (j < nVocab) then
        Probs[LastPos, j] := Probs[LastPos, j] * 0.25;
    end;

    // Find the adjusted top-K candidates.
    TopKProbs;

    // Sample one token from the top-K candidates.
    BestTok := SampleTopK(TopTokVector, TopProbVector, TopTok, TopKSampleProb);

    RawProb := RawProbs[BestTok];
    AdjProb := Probs[LastPos, BestTok];

    QueryToken := BestTok;
    AdjustedProb := AdjProb;

    if VerboseInfer then
      ReportInferenceDiagnostics(BestTok, RawProb, AdjProb, TopKSampleProb, Top1Prob, Top2Prob, TopKMass, Entropy, RawEOSProb);

  end;
end;

{ Interactive inference }
// Run repeated interactive queries; for each query, generate and display cumulative responses.
procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
var
  i, Step, QueryToken, ResponseInterval, OldNVocab: Integer;
  OldTraining, OldVerboseTransform, OldSaveTokenizationFiles, OwnsCuda, StartedCudaHere: Boolean;
  QueryTokenized, WorkTokens, QueryOutput: TIVector;
  QueryInput: TBVector;
  AdjustedProb: Single;
  MaxNewTokens: Integer = 1000;
  QueryString, TaggedQueryString, TokenizeString, ResponseCommand: string;
  LastDisplayedCount: Integer;

  // Read commands until the user enters a real query or requests exit.
  function ReadInferenceQuery(out AQuery: string): Boolean;
  var
    Code, NewValue: Integer;
    ValueText: string;
  begin
    Result := False;

    while True do begin
      Writeln('V = toggle Verbose mode. D = toggle Detail mode. I = program Information. L = response Length. M = Max new tokens. X = eXit inference.');
      Write('Enter query, blank to return: ');
      ReadLn(AQuery);
      AQuery := Trim(AQuery);

      if (AQuery = '') or SameText(AQuery, 'X') then begin
        Writeln('>>Leaving inference.');
        Exit;
      end;

      if SameText(AQuery, 'V') then begin
        VerboseInfer := not VerboseInfer;
        Writeln('Verbose inference = ', VerboseInfer, '.');
        Continue;
      end;

      if SameText(AQuery, 'D') then begin
        DetailInfer := not DetailInfer;
        Writeln('Detail inference = ', DetailInfer, '.');
        Continue;
      end;

      if SameText(AQuery, 'I') then begin
        ReportProgramInfo;
        Writeln('Model ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), ': nTC = ', nTokenizedCorpus, '; nCorpus = ', nCorpus, '; nVocab = ', nVocab,
          '; DimVocab = ', DimVocab, '; SeqLen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock = ', nBlock, '; Proj = ', Proj);
        Continue;
      end;

      if SameText(AQuery, 'M') then begin
        Write('Enter maximum number of new tokens per query: ');

        repeat
          ReadLn(ValueText);
          Val(ValueText, NewValue, Code);

          if (Code <> 0) or (NewValue < 1) then
            Writeln('Invalid number, try again.');
        until (Code = 0) and (NewValue >= 1);

        MaxNewTokens := NewValue;

        if ResponseInterval > MaxNewTokens then
          ResponseInterval := MaxNewTokens;

        Writeln('Maximum new tokens = ', MaxNewTokens, '.');
        Continue;
      end;

      if SameText(AQuery, 'L') then begin
        Write('Enter cumulative response length in tokens: ');

        repeat
          ReadLn(ValueText);
          Val(ValueText, NewValue, Code);

          if (Code <> 0) or (NewValue < 1) then
            Writeln('Invalid length, try again.');
        until (Code = 0) and (NewValue >= 1);

        ResponseInterval := NewValue;

        if ResponseInterval > MaxNewTokens then
          ResponseInterval := MaxNewTokens;

        Writeln('Cumulative response length = ', ResponseInterval, ' tokens.');
        Continue;
      end;

      Result := True;
      Exit;
    end;
  end;

  // Display all output generated so far for the current query.
  procedure DisplayCumulativeResponse;
  begin
    if Length(QueryOutput) = 0 then Exit;

    // Write('Response through token ', Length(QueryOutput), ': ');

    if TokenizerKind = UDTokenizer then
      WriteUDInferenceTokens(QueryOutput)
    else
      WriteInferenceTokens(QueryOutput);

    Writeln;
    LastDisplayedCount := Length(QueryOutput);
  end;

  // Return True to continue generating the current query.
  function ContinueCurrentQuery: Boolean;
  begin
    Write('Hit <Enter> to continue this query or N for a new query: ');
    ReadLn(ResponseCommand);
    Result := not SameText(Trim(ResponseCommand), 'N');
  end;

begin
  OldTraining := Training;
  OldVerboseTransform := VerboseTransform;
  OldSaveTokenizationFiles := SaveTokenizationFiles;
  OldNVocab := nVocab;

  OwnsCuda := False;
  StartedCudaHere := False;

  // Set inference defaults.
  DetailInfer := False;
  VerboseTransform := False;
  ResponseInterval := 100;

  try
    Training := False;
    SaveTokenizationFiles := False;

    if TokenizerKind in [WesTokenizer, UDTokenizer] then
      if nVocab <> Length(SymbolTable) then
        raise Exception.CreateFmt('Inference vocabulary mismatch: model nVocab=%d, symbol table length=%d.', [nVocab, Length(SymbolTable)]);

    if nVocab > DimVocab then begin
      Writeln('nVocab > DimVocab. Aborting inference...');
      Exit;
    end;

    InitializeTransformerState(WModelState);

    OwnsCuda := not CudaAllocated;

    if OwnsCuda then begin
      StartCuda(WModelParams, WModelState, WAdamWState);
      StartedCudaHere := True;
    end;

    if OwnsCuda or ParamsNeedCopyToDevice then begin
      CopyParamsToDevice(WModelParams);
      ParamsNeedCopyToDevice := False;
    end;

    CopyInvFreqToDevice(WModelState);

    Writeln('Starting inference...');

    { Query loop }
    while ReadInferenceQuery(QueryString) do begin

      // Prepare the query text for the active tokenizer.
      TokenizeString := QueryString;

      // UD models receive prefix-tagged text before native Wes tokenization.
      if TokenizerKind = UDTokenizer then begin
        try
          TaggedQueryString := UDTagText(UDPipeFileName, UDModelFileName, QueryString);
        except
          on E: Exception do begin
            Writeln('UD tagging error: ', E.Message);
            Continue;
          end;
        end;

        if TaggedQueryString = '' then begin
          Writeln('UDPipe returned an empty tagged query.');
          Continue;
        end;

        TokenizeString := TaggedQueryString;

        if VerboseInfer then begin
          Writeln('Original query: ', QueryString);
          Writeln('UD tagged query: ', TaggedQueryString);
        end;
      end;

      // Tokenize the query using either the native Wes/UD engine or GPT-2.
      if TokenizerKind in [WesTokenizer, UDTokenizer] then begin
        SetLength(QueryInput, Length(TokenizeString));

        for i := 0 to Length(TokenizeString) - 1 do
          QueryInput[i] := Ord(TokenizeString[i + 1]);

        if VerboseInfer then begin
          Write(Length(QueryInput), ' query bytes: ');
          for i := 0 to High(QueryInput) do
            Write(QueryInput[i], ' ');
          Writeln;
        end;

        TokenizeWesBytes(QueryInput, QueryTokenized);
      end
      else
        RunGPT2TokenizeString(QueryString, QueryTokenized);

      // The model must generate the next EOS itself; do not leave tokenizer-added EOS in the prompt.
      if (Length(QueryTokenized) > 0) and (QueryTokenized[High(QueryTokenized)] = EOS) then
        SetLength(QueryTokenized, Length(QueryTokenized) - 1);

      if Length(QueryTokenized) = 0 then begin
        Writeln('No input tokens remain after removing EOS.');
        Continue;
      end;

      if VerboseInfer and (TokenizerKind in [WesTokenizer, UDTokenizer]) then
        TCFull(QueryTokenized);

      SetLength(QueryOutput, 0);
      WorkTokens := Copy(QueryTokenized);
      QueryToken := -1;
      AdjustedProb := 0.0;
      LastDisplayedCount := 0;

      { Generation loop }
      Step := 1;

      while Step <= MaxNewTokens do begin

        // Generate one additional token from all context accumulated so far.
        InferOneToken(WModelParams, WModelState, Step, WorkTokens, QueryToken, AdjustedProb);

        // EOS terminates the current query but is not appended to the displayed response.
        if QueryToken = EOS then begin
          if LastDisplayedCount <> Length(QueryOutput) then
            DisplayCumulativeResponse;

          Writeln('<EOS>: token generation ended.');
          Break;
        end;

        // Append the token both to the response and to context for the next inference step.
        SetLength(QueryOutput, Length(QueryOutput) + 1);
        QueryOutput[High(QueryOutput)] := QueryToken;

        SetLength(WorkTokens, Length(WorkTokens) + 1);
        WorkTokens[High(WorkTokens)] := QueryToken;

        if DetailInfer then begin
          Write('WorkTokens: <<');
          WriteInferenceTokens(WorkTokens);
          Writeln('>>');
          Writeln('Selected adjusted probability = ', AdjustedProb:0:7, '.');
          PauseNNL;
        end;

        // At each response interval, print the complete response generated for this query so far.
        if (Length(QueryOutput) mod ResponseInterval) = 0 then begin
          DisplayCumulativeResponse;

          if Step < MaxNewTokens then
            if not ContinueCurrentQuery then Break;
        end;

        Inc(Step);
      end;

      // Display any final partial response that did not land exactly on an interval boundary.
      if LastDisplayedCount <> Length(QueryOutput) then
        DisplayCumulativeResponse;

      if (QueryToken <> EOS) and (Step > MaxNewTokens) then
        Writeln('Maximum of ', MaxNewTokens, ' new tokens reached for this query.');

      // Show the exact raw token reconstruction when detailed diagnostics are enabled.
      if DetailInfer then begin
        if TokenizerKind = UDTokenizer then
          Write('Raw UD tagged output: ')
        else
          Write('Raw generated output: ');

        WriteInferenceTokens(QueryOutput);
        Writeln;
      end;
    end;

  finally
    try
      if StartedCudaHere then
        EndCuda(WModelParams, WModelState, WAdamWState);
    finally
      Training := OldTraining;
      VerboseTransform := OldVerboseTransform;
      SaveTokenizationFiles := OldSaveTokenizationFiles;
      nVocab := OldNVocab;
    end;
  end;
end;

end.
