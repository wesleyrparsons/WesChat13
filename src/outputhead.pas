unit OutputHead;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  SysUtils, Util,
  { WesChat units }
  Display, Global, Matrix;

procedure RunOutputForward(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure RunOutputBackward(var WModelParams: TWModelParams; var WModelState: TWModelState);

implementation

{ Output head forward pass }
// Compute vocabulary logits and probabilities from the final transformer output.
procedure RunOutputForward(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  Temperature: Single;
begin
  Stage := nBlock * 4 + 2;

  if Training then
    Temperature := TTemperature
  else
    Temperature := ITemperature;

  with WModelParams do with WModelState do begin

    // 3A. Compute vocabulary logits from the final transformer output using tied embeddings.
    if DisplayStage then Writeln('' : Stage, 'Stage 3, Block ', nBlock - 1, ', Model Output');
    if DisplaySubstage then Writeln('' : Stage, '3A. Output Head Forward, compute vocabulary logits');

    // Probs is used first as the logits buffer: Probs = X7 * Embeddings^T.
    // X7 is L x D; Embeddings is nVocab x D; Probs is L x nVocab with physical row stride DimVocab.
    CuMatMulFullNT(CuHandle, StateBlock[nBlock - 1].X7.dValue, Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

    if VerboseTransform then begin
      cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display vocabulary logits before softmax.', Probs, B);
    end;

    if VerboseTransform then begin
      cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Embeddings.Value before computing output probabilities.', Embeddings.Value, B);
    end;

    // 3B. Apply temperature-scaled softmax in place: logits -> probabilities.
    if DisplaySubstage then Writeln('' : Stage, '3B. Output Head Forward, softmax vocabulary logits');
    LaunchSoftmaxForwardStrided(dProbs, dProbs, SeqLen, nVocab, DimVocab, Temperature);

    if VerboseTransform then begin
      cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display output probabilities after softmax.', Probs, B);
    end;

  end;

  if DebugCudaChecks then
    CheckCudaError('OutputHead forward pass.');
end;

{ Output head backward pass }
// Backpropagate cross-entropy through the tied output head.
procedure RunOutputBackward(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  GradScale: Single;
begin
  Stage := nBlock * 4 + 2;

  with WModelParams do with WModelState do begin

    // 3C. Compute the cross-entropy gradient with respect to the vocabulary logits.
    if DisplaySubstage then Writeln('' : Stage, '3C. Output Head Backward, obtain TopGradient from probabilities');

    // Average the loss gradient across sequence positions and include the training-temperature derivative.
    GradScale := 1.0 / (SeqLen * TTemperature);
    LaunchCEGradientStrided(dProbs, dTopGradient, dTargetTokens, SeqLen, nVocab, DimVocab, GradScale);

    if VerboseTransform then begin
      cudaMemcpy(@TopGradient[0, 0], dTopGradient, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display TopGradient in the output head.', TopGradient, B);
    end;

    // 3D. Backpropagate through logits = X7 * Embeddings^T.
    if DisplaySubstage then Writeln('' : Stage, '3D. Output Head Backward, obtain X7 Grad and tied-embedding Grad');

    with StateBlock[nBlock - 1] do begin

      // X7.Grad = TopGradient * Embeddings. TopGradient is L x nVocab; Embeddings is nVocab x D; X7.Grad is L x D.
      CuMatMulFullNN(CuHandle, dTopGradient, Embeddings.dValue, X7.dGrad, SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);

      // Accumulate the output-head contribution into the tied embedding gradient:
      // Embeddings.Grad += TopGradient^T * X7. Embeddings.Grad is nVocab x D.
      CuMatMulFullAccTN(CuHandle, dTopGradient, X7.dValue, Embeddings.dGrad, nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);

      if VerboseTransform then begin
        cudaMemcpy(@X7.Grad[0, 0], X7.dGrad, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X7.Grad after output-head backpropagation.', X7.Grad, G);
      end;

    end;
  end;

  if DebugCudaChecks then
    CheckCudaError('OutputHead backprop pass.');
end;

end.
