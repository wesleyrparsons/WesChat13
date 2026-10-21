unit Embed;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  IOHandler,
  Math,
  Matrix,
  SysUtils,
  TransformForward,
  TransformBackprop,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunEmbed(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector);

implementation

const
  Scale = Sqrt(ModelDim);         // Optional transformer-style embedding scaling by sqrt(d_model).

function CheckDLL(const LibName: string): Boolean;
var
  DLLHandle: THandle;
begin
  // Attempt to load the library.
  DLLHandle := LoadLibrary(PChar(LibName));

  // If the handle is non-zero, it loaded successfully.
  Result := (DLLHandle <> 0);

  // Clean up if it was successfully loaded.
  if Result then
    FreeLibrary(DLLHandle);
end;

// Check DLL accessibility.
procedure CheckAllDLLs;
begin
  If CheckDLL('cublas64_13.dll') then CublasPresent := True else CublasPresent := False;
  If CheckDLL('cudart64_13.dll') then CudartPresent := True else CudartPresent := False;
  If CheckDLL('WesChatKernel12.dll') then WesChatKernelPresent := True else WesChatKernelPresent := False;
  if not CublasPresent or not CudartPresent or not WesChatkernelPresent then begin
      Writeln('One of the following DLLs is required but not present: cublas64_13.dll, cudart64_13.dll, WesChatKernel12.dll.');
      Pause;
      Halt;
  end;
  if CublasPresent and (cublasCreate_v2(CuHandle) <> 0) then begin
    Writeln('cuBLAS initialization required but failed.');
    Pause;
    Halt;
  end;
end;

// Create the target vector for use in head output.
procedure BuildTargetVector(var Target: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Target[i] := TokenizedCorpus[StartIndex + i + 1];
end;

// Build the input vector .
procedure BuildInputVector(var Input: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Input[i] := TokenizedCorpus[StartIndex + i];
end;

// Create the input matrix and remember which token created each row.
procedure BuildInputMatrix(var X: TSeqMatrix; var InputTokens: TIDimVector; const TokenizedCorpus: TIVector;
  var WModelParams: TWModelParams; const Start, L: Integer);
var
  i, j, id: Integer;
begin
  Assert(Start >= 0);
  Assert(Start + L <= Length(TokenizedCorpus));

  for i := 0 to L - 1 do begin
    id := TokenizedCorpus[Start + i];

    Assert(id >= 0);
    Assert(id < nSymbols);

    InputTokens[i] := id;

    for j := 0 to ModelDim - 1 do
      X[i, j] := WModelParams.Embeddings.Value[id, j];
  end;
end;

// Run the training.
procedure RunEmbed(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector);
var
  i, j, k, Blk, LastBlk: Integer;
  Start, EmbedLoop: Integer;
  Stride: Integer = 64;      // Stride 64 tokens every sequence.

  procedure ReadEmbedIfKeyPressed;
  var
    key: char;
    ModelFileName: string;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X': begin
        Writeln('Exit requested. Stopping execution.');
        Pause;
        Halt;                // Immediately terminate program.
      end;
      'b', 'B': begin
        Writeln('Break requested. Exiting loop.');
        Pause;
        Blk := nBlock;     // Break out of the loop cleanly.
      end;
      'v', 'V': begin
        VeryVerbose := not VeryVerbose;
        Writeln('Very verbose mode: ', VeryVerbose);
        Pause;
      end;                   // Change verbosity.
      'i', 'I': begin
        Writeln;
        ReportInfo;          // Report program info.
        Pause;
      end;
      't', 'T': begin
        Writeln('Training. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim,
          '  Start = ', Start, ' Stride = ', Stride, ' SeqLen = ', SeqLen, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));
        Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of loop. V = toggle Verbose mode. P = Pause.');
        Writeln('  W = WesChat Information. T = Training information. S = Save. Training...');
        Pause;
      end;
      's', 'S': begin
        ChDir(WorkingDir);   // Save model.
        Write('Enter filename: ');
        Readln(ModelFileName);
        if SaveModel(ModelFileName, WModelParams) then
          Writeln('File ', f, ' successfully saved.')
        else
          Writeln('File not saved.');
        Pause;
      end;

    end;
  end;

begin
  CheckAllDLLs;

  nVocab := nSymbols;    // Need nVocab (second name for variable) for Transform.

  if VeryVerbose then
    Writeln('Start Training. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim,
      ' SeqLen = ', SeqLen, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));

  // Seed the weights with random numbers.
  for i := 0 to nSymbols - 1 do             // Random normal distribution.
    for j := 0 to ModelDim - 1 do           // Mean = 0, SD = 0.02.
      WModelParams.Embeddings.Value[i, j] := RandG(0.0, 0.02); // Only time I use this randomizer.

  Writeln('First quarter of first row of embeddings.');
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModelParams.Embeddings.Value[1, k]: 8: 6, ' ');
  Writeln;
  Pause;

  VTPDisplayX('Display Embeddings.Value prior to Transform.', WModelParams.Embeddings.Value, B);

  // Initialize.
  InitializeTransformer(WModelParams, WModelState);
  MAllocCublas(WModelParams, WModelState);
  CopyParamsToDevice(WModelParams);
  CopyInvFreqToDevice(WModelState);

  // SetLength(TokenID, Length(TokenizedCorpus));
  // TokenID := TokenizedCorpus;

  // Stride loop thru Sequence.
  Start := 0;
  EmbedLoop := 0;
  while (Start + SeqLen) < Length(TokenizedCorpus) do begin      // add with WModelState do

    // Display number of loops thru embed loop.
    Inc(EmbedLoop);
    Writeln('&&& SeqLen loop: start ', Start, ' and loop number ', EmbedLoop, ' &&&');
    Writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
    Writeln('  P = Program information. E = Embedding information. Embedding & transforming...');

    if VerboseTransform then Pause;

    // Build the target vector, one ahead, for the loss stage.
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);
    BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);
    cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    // Checking.
    Writeln('Checking tokens');
    for i := 0 to SeqLen - 1 do
      Write(i:4, ' token=', InputTokens[i], ' ');
    Writeln('Values for row zero.');
    k := InputTokens[0];
    for j := 0 to 15 do
      Write(WModelParams.Embeddings.Value[k, j]:8:5, ' ');
    Pause;

    // Build X from TokenizedCorpus[start .. start + SeqLen - 1].
    // Non-c kernel.
    // BuildInputMatrix(WModelState.StateBlock[0].X.Value, InputTokens, TokenizedCorpus, WModelParams, Start, SeqLen);
    // cuda kernel.
    LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, WModelState.StateBlock[0].X.dValue, SeqLen, ModelDim);

    // Display X.Value matrix.
    cudaMemcpy(@WModelState.StateBlock[0].X.Value[0,0], WModelState.StateBlock[0].X.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display X.Value before transform.', WModelState.StateBlock[0].X.Value, G);
    VTPDisplayX('Display X.Value before transform.', WModelState.StateBlock[0].X.Value, B);

    // Optional transformer-style embedding scaling by sqrt(d_model).
    // cudaMemcpy(WModelState.StateBlock[0].X.dValue, @WModelState.StateBlock[0].X.Value[0,0], XSize, cudaMemcpyHostToDevice);
    CuScale(CuHandle, SeqLen * ModelDim, Scale, WModelState.StateBlock[0].X.dValue);

    // Zero gradients.
    for k := 0 to nBlock - 1 do
      ZeroGradients(WModelParams, WModelState, k);

    // Forward pass thru transformer.
    for Blk := 0 to nBlock - 1 do begin
      Writeln('     $$$ Forward Block loop: start ', Blk, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      RunTransformForward(WModelParams, WModelState, Blk);

      if Blk < nBlock - 1 then
        // cblas.
        // CopyXTensor(WModelState.StateBlock[Blk].X7, WModelState.StateBlock[Blk + 1].X);
        // cuda kernel.
        cudaMemcpy(WModelState.StateBlock[Blk + 1].X.dValue, WModelState.StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);

      if PauseIfKeyPressed then
        ReadEmbedIfKeyPressed;
    end;

    LastBlk := nBlock - 1;

    // 3. FORWARD HEAD OUTPUT STAGE.

    with WModelParams do with WModelState do begin
      // 3A. Multiplication/Overwrite. Obtain Probs from X7 and Vocab.
      Writeln('              Transform Gradient Stage 3A');

      // Multiplication: Input X7, Vocab. Output Probs.
      // Equation: Probs = X7 · Embeddingsᵀ. Probs in R^{L x nVocab}. X in R^{L x D}.  Embeddings in R^{nVocab x D}.
      // cblas.
      // MatMulFullNT(@StateBlock[LastBlk].X7.Value[0, 0], @Embeddings.Value[0, 0], @Probs[0, 0], SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);
      // cublas.
      CuMatMulFullNT(CuHandle, StateBlock[LastBlk].X7.dValue, Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

      // Display Probs matrix.
      cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Probs, in transform, before softmax.', Probs, B);

      // Display Embeddings.Value matrix.
      cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Embeddings.Value in transform, before computing Logit.', Embeddings.Value, B);

      // 3B. Softmax. Obtain Probs from Probs.
      Writeln('            Transform Forward Stage 3B');

      // Softmax: Input Logit. Output Logit.
      // Equation: Logit = Softmax(Logit).
      // Use SoftmaxForwardN here. Probs is already in cblas.
        // Non-cuda kernel.
        // for i := 0 to SeqLen - 1 do
        // SoftmaxForwardN(@Probs[i,0], @Probs[i,0], nVocab);
        // cuda kernel.
        LaunchSoftmaxForwardN(dProbs, dProbs, SeqLen, nVocab, Temperature);

      // Display Probs matrix.
      cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Probs, in transform, after softmax.', Probs, B);

      // 3C. Cross-Entropy Loss. Obtain TopGradient from Probs.
      Writeln('            Transform Forward Stage 3D');
      // Gradient: Input Probs. Output TopGradient. Also option of CalculateGradient from KLDivergence.
      // Equation: TopGradient in R^{L x nVocab}. Probs in R^{L x nVocab}.
      // Non-cuda kernel.
      // GradientFromCEProbabilities(WModelState);  // Using CE.
      // GradientFromKLDivergence(WModelState);   // Not using KL.
      // cuda kernel.
      cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);
      LaunchCEGradient(WModelState.dProbs, WModelState.dTopGradient, dTargetTokens, SeqLen, nVocab);

      // Display TopGradient matrix.
      cudaMemcpy(@TopGradient[0, 0], dTopGradient, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display TopGradient, in transform, after Logit calculation.', TopGradient, B);

      // 3D. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.
      Writeln('              Transform Backprop Stage 3E');

      with StateBlock[LastBlk] do begin
        // Equation: X7.Grad = TopGradient · Embeddings.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. Embeddings.Value in R^{nVocab x D}.
        // cblas.
        // MatMulFullNN(@TopGradient[0, 0], @Embeddings.Value[0, 0], @X7.Grad[0, 0], SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);
        {cblas_sgemm(101, 111, 111, SeqLen, ModelDim, nVocab, 1.0, @TopGradient[0, 0], DimVocab,
        @Embeddings.Value[0, 0], ModelDim, 0.0, @X7.Grad[0, 0], ModelDim);}
        // cublas.   Embeddings amd X7 already copied.
        CuMatMulFullNN(CuHandle, dTopGradient, Embeddings.dValue, X7.dGrad, SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);
        Writeln('Finished MatMul X7.Grad loop.');

        // Backprop TopGradient modifies/overwrites Embeddingsᵀ: Input X7ᵀ, TopGradient. Output Embeddingsᵀ.Grad.
        // Equation: Embeddingsᵀ.Grad = X7ᵀ · TopGradient. Embeddingsᵀ.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L}. TopGradient in R^{L x nVocab}.
        // Problem here was I had NT rather than TN.
        // cblas.
        // MatMulFullAccTN(@TopGradient[0,0], @X7.Value[0,0], @Embeddings.Grad[0,0], nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);
        // cublas.
        CuMatMulFullAccTN(CuHandle, dTopGradient, X7.dValue, Embeddings.dGrad, nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);

        Writeln('Finished Embeddings.Grad GEMM.');

        // Backprop Split X7 Grad into X5 and X6: Input X5.Grad, X7.Grad. Output dX.Grad.
        // Equation: X5.Grad = X5.Grad + X7.Grad. All in R^{L x D}.
        // cblas.
        // GradSplit(X7.Grad, X5.Grad, X6.Grad, SeqLen, ModelDim);
        // cublas.
        CuGradSplit(CuHandle, X7.dGrad, WModelState.StateBlock[LastBlk].X5.dGrad, WModelState.StateBlock[LastBlk].X6.dGrad, SeqLen, ModelDim);

        // Display X7.Grad matrix.
        cudaMemcpy(@X7.Grad[0, 0], X7.dGrad, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X7.Grad, in transform, after stage 2D.', X7.Grad, G);
      end;
    end; // End gradient stage.

    // Backprop pass thru transformer.
    for Blk := nBlock - 1 downto 0 do begin
      Writeln('     $$$ Backpropd Block loop: start ', Blk, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      RunTransformBackprop(WModelParams, WModelState, Blk);

      if Blk > 0 then
        // cblas.
        // CopyXTensor(WModelState.StateBlock[Blk].X1, WModelState.StateBlock[Blk - 1].X7);
        // cuda kernel.
        cudaMemcpy(WModelState.StateBlock[Blk].X1.dValue, WModelState.StateBlock[Blk - 1].X7.dValue, XSize, cudaMemcpyDeviceToDevice);

      if PauseIfKeyPressed then
        ReadEmbedIfKeyPressed;
    end;

    // Modify weights and biases.
    for k := 0 to nBlock - 1 do
      Optimization(WModelParams, k);

    // Apply the total embedding gradient (output-side + input-side).
    UpdateEmbeddings(wModelParams, WModelState, InputTokens);

    Start := Start + Stride;
  end; // End sequence loop.

  Writeln('End of training. Press <CR> to continue.');
  Readln;
end;

end.

