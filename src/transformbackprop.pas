unit TransformBackprop;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  SysUtils,
  { WesChat units }
  Display, Global, Matrix, Util;

procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

implementation

{ Transformer backward pass }
// Run one transformer block backward.
procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h, HeadOffset, ValueBWidth: Integer;
begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

      { Feed-forward backward pass }
      Stage := Blk * 4 + 4;

      // Display X6 before starting the feed-forward backward pass.
      if VerboseTransform then begin
        cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X6.Value in transform, before any backpropagation.', X6.Value, G);
      end;

      // 2F. Split the final residual gradient: X7 = X4 + X6.
      if DisplayStage then Writeln('' : Stage, 'Stage  2, Block ', Blk, ', Transform Backprop');
      if DisplaySubstage then Writeln('' : Stage, '2F. Transform Backprop, split X7 gradient and reverse residual dropout');
      CuGradSplit(CuHandle, X7.dGrad, X4.dGrad, X6.dGrad, SeqLen, ModelDim);

      // Reverse residual dropout on the X6 path.
      if Training then
        LaunchDropoutBackward(X6.dGrad, SeqLen * ModelDim, RDropout, RDropoutSeed);

      // 2E. Accumulate the b2 gradient from X6.Grad.
      if DisplaySubstage then Writeln('' : Stage, '2E. Transform Backprop, obtain b2 Grad from X6 Grad');
      // Equation: b2.Grad = sum_rows(X6.Grad). b2.Grad is D; X6.Grad is L x D.
      LaunchAddBiasRowsBackward(X6.dGrad, b2.dGrad, SeqLen, ModelDim);

      // 2D. Backpropagate the W2 projection.
      if DisplaySubstage then Writeln('' : Stage, '2D. Transform Backprop, obtain W2 and Hidden2 gradients');

      if MLPGatingKind = SplitSwiGLU then begin

        ValueBWidth := 2 * GatedDim - ModelDimProj;

        // W2[0..511].Grad = GatedOutput^T * X6.Grad. Read directly from the strided Hidden2 matrix.
        CuMatMulFullTN(CuHandle, PSingle(Hidden2.dValue) + ValueBWidth, X6.dGrad, W2.dGrad,
          GatedDim, ModelDim, SeqLen, ModelDimProj, ModelDim, ModelDim);

        // Calculate the gated output gradient directly into columns 256..767 of Hidden2.dGrad.
        CuMatMulFullNT(CuHandle, X6.dGrad, W2.dValue, PSingle(Hidden2.dGrad) + ValueBWidth,
          SeqLen, GatedDim, ModelDim, ModelDim, ModelDim, ModelDimProj);

        // Reverse dropout on the gated portion only.
        if Training then
          LaunchDropoutBackwardStrided(Hidden2.dGrad, SeqLen, GatedDim, ModelDimProj, ValueBWidth,
            MLPDropOut, MLPDropoutSeed);
      end
      else begin
        // Existing NoGating and SelfGating backprop.
        CuMatMulFullTN(CuHandle, Hidden2.dValue, X6.dGrad, W2.dGrad, ModelDimProj, ModelDim, SeqLen,
          ModelDimProj, ModelDim, ModelDim);
        CuMatMulNT(CuHandle, X6.dGrad, W2.dValue, Hidden2.dGrad, SeqLen, ModelDimProj, ModelDim);

        if Training then
          LaunchDropoutBackward(Hidden2.dGrad, SeqLen * ModelDimProj, MLPDropOut, MLPDropoutSeed);
      end;

      // 2C. Backpropagate MLP activation and gating.
      if DisplaySubstage then Writeln('' : Stage, '2C. Transform Backprop, activation or gating');

      case MLPGatingKind of
        SplitSwiGLU: begin
          // Obtain gate and value gradients. Hidden1.Grad receives gate and Value A gradients.
          // Hidden2.Grad columns 0..255 receive Value B gradients.
          LaunchSwiGLUSplitBackward(Hidden1.dValue, Hidden2.dValue, Hidden2.dGrad, Hidden1.dGrad,
            SeqLen, ModelDimProj, GatedDim);

          // Calculate the gradient of the lower portion of W2. W2[512..767].Grad = ValueB.Grad^T * X5.
          // ValueB.Grad occupies the first 256 columns of Hidden2.Grad, with physical row pitch 768.
          CuMatMulFullTN(CuHandle, Hidden2.dGrad, X5.dValue, PSingle(W2.dGrad) + GatedDim * ModelDim,
            ValueBWidth, ModelDim, SeqLen, ModelDimProj, ModelDim, ModelDim);
        end;

        SelfGating:
          LaunchSelfGatedSiLUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);

        NoGating:
          case ActivationKind of
            ReLU:      LaunchReLUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);
            LeakyReLU: LaunchLeakyReLUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj, ActivationAlpha);
            GELU:      LaunchGELUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);
            SiLU:      LaunchSiLUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);
            ELU:       LaunchELUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj, ActivationAlpha);
            Softplus:  LaunchSoftplusBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);
            Mish:      LaunchMishBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);
          end;
      end;

      // 2B. Accumulate b1 gradient.
      if DisplaySubstage then Writeln('' : Stage, '2B. Transform Backprop, obtain b1 Grad');

      LaunchAddBiasRowsBackward(Hidden1.dGrad, b1.dGrad, SeqLen, ModelDimProj);

      // 2A. Backpropagate Hidden1 = X5 * W1.
      if DisplaySubstage then Writeln('' : Stage, '2A. Transform Backprop, obtain W1 Grad and X5 Grad');

      // W1.Grad = X5^T * Hidden1.Grad.
      CuMatMulFullTN(CuHandle, X5.dValue, Hidden1.dGrad, W1.dGrad, ModelDim, ModelDimProj, SeqLen,
        ModelDim, ModelDimProj, ModelDimProj);

      // X5.Grad += Hidden1.Grad * W1^T.
      CuMatMulAccNT(CuHandle, Hidden1.dGrad, W1.dValue, X5.dGrad, SeqLen, ModelDim, ModelDimProj);

      if MLPGatingKind = SplitSwiGLU then
        // X5.Grad += ValueB.Grad * W2[512..767]. ValueB.Grad occupies the first 256 columns of Hidden2.Grad.
        // Hidden2.Grad has a physical row pitch of 768.
        CuMatMulFullAccNN(CuHandle, Hidden2.dGrad, PSingle(W2.dValue) + GatedDim * ModelDim,
          X5.dGrad, SeqLen, ModelDim, ValueBWidth, ModelDimProj, ModelDim, ModelDim);

    { Attention backward pass }
    Stage := Stage - 2;

    // 1J. Backpropagate X5 = Norm(X4).
    if DisplayStage then Writeln('' : Stage, 'Stage  1, Block ', Blk, ', Transform Backprop');
    if DisplaySubstage then Writeln('' : Stage, '1J. Transform Backprop, Norm backward from X5 to X4');

    // Norm backward produces the X4 path gradient plus Gamma2 and Beta2 gradients.
    case NormKind of
      LayerNorm: LaunchLayerNormBackward(X5.dGrad, dX4FromLN2, Gamma2.dValue, dLNXHat2, dLNInvStd2, Gamma2.dGrad, Beta2.dGrad, SeqLen, ModelDim);
      RMSNorm: LaunchRMSNormBackward(X5.dGrad, dX4FromLN2, Gamma2.dValue, dLNXHat2, dLNInvStd2, Gamma2.dGrad, SeqLen, ModelDim);
    end;
    CuAccumulateGrad(CuHandle, dX4FromLN2, X4.dGrad, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X4.Grad[0, 0], X4.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Grad after Norm backward.', X4.Grad, G);
    end;

    // 1I. Split the attention residual gradient: X4 = X + X3.
    if DisplaySubstage then Writeln('' : Stage, '1I. Transform Backprop, split X4 Grad into X Grad and X3 Grad');
    CuGradSplit(CuHandle, X4.dGrad, X.dGrad, X3.dGrad, SeqLen, ModelDim);

    // 1H. Backpropagate X3 = X2 * W0.
    if DisplaySubstage then Writeln('' : Stage, '1H. Transform Backprop, obtain W0 Grad and X2 Grad from X3 Grad');

    // W0.Grad = X2^T * X3.Grad. X2 and X3.Grad are L x D; W0.Grad is D x D.
    CuMatMulFullTN(CuHandle, X2.dValue, X3.dGrad, W0.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X2.Grad = X3.Grad * W0^T. X2.Grad is L x D.
    CuMatMulNT(CuHandle, X3.dGrad, W0.dValue, X2.dGrad, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X3.Grad[0, 0], X3.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3.Grad before attention backpropagation.', X3.Grad, G);
    end;

    // 1G-E. Backpropagate the per-head attention path.
    if DisplaySubstage then Writeln('' : Stage, '1G-E. Transform Backprop, attention value path, dropout, softmax, mask, Q and K');

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // 1G. ScoresHead2.Grad = X2h.Grad * Vh^T. ScoresHead2.Grad is L x L.
      CuMatMulFullNT(CuHandle, PSingle(X2.dGrad) + HeadOffset, PSingle(V.dValue) + HeadOffset,
        ScoresHead2[h].dGrad, SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen);

      // Vh.Grad = ScoresHead2^T * X2h.Grad. Vh.Grad is L x H.
      CuMatMulFullTN(CuHandle, ScoresHead2[h].dValue, PSingle(X2.dGrad) + HeadOffset,
        PSingle(V.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);

      // 1F-c. Reverse attention dropout.
      if Training then
        LaunchDropoutBackward(ScoresHead2[h].dGrad, SeqLen * SeqLen, ADropout, ADropoutSeed + h);

      // 1F-b. Backpropagate softmax. ScoresHead1.dValue contains the saved pre-dropout softmax probabilities.
      LaunchSoftmaxBackward(ScoresHead1[h].dValue, ScoresHead2[h].dGrad, ScoresHead1[h].dGrad, SeqLen, SeqLen);

      // 1E. Forward attention scores were scaled by 1 / sqrt(HeadDim), so scale their gradient likewise.
      CuScale(CuHandle, SeqLen * SeqLen, InvSqrtHeadDim, ScoresHead1[h].dGrad);

      // 1F-a. Zero gradients in positions suppressed by the autoregressive mask.
      LaunchAutoRegressiveMaskBackward(ScoresHead1[h].dGrad, SeqLen);

      // Qh.Grad = ScoresHead1.Grad * Kh. Qh.Grad is L x H.
      CuMatMulFullNN(CuHandle, ScoresHead1[h].dGrad, PSingle(K.dValue) + HeadOffset,
        PSingle(Q.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);

      // Kh.Grad = ScoresHead1.Grad^T * Qh. Kh.Grad is L x H.
      CuMatMulFullTN(CuHandle, ScoresHead1[h].dGrad, PSingle(Q.dValue) + HeadOffset,
        PSingle(K.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    if VerboseTransform then begin
      cudaMemcpy(@ScoresHead1[0].Grad[0, 0], ScoresHead1[0].dGrad, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display ScoresHead1[0].Grad before Q and K projection backpropagation.', ScoresHead1[0].Grad, G);
    end;

    // 1D. Reverse RoPE on Q.Grad and K.Grad.
    if DisplaySubstage then Writeln('' : Stage, '1D. Transform Backprop, RoPE backward');
    LaunchRoPEBackward(Q.dGrad, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);
    LaunchRoPEBackward(K.dGrad, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);

    // 1C. Backpropagate the Q, K, and V projections from X1.
    if DisplaySubstage then Writeln('' : Stage, '1C. Transform Backprop, obtain Wq/Wk/Wv Grads and X1 path Grads');

    // Wq.Grad = X1^T * Q.Grad.
    CuMatMulFullTN(CuHandle, X1.dValue, Q.dGrad, Wq.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X1q.Grad = Q.Grad * Wq^T.
    CuMatMulNT(CuHandle, Q.dGrad, Wq.dValue, X1q.dGrad, SeqLen, ModelDim, ModelDim);

    // Wk.Grad = X1^T * K.Grad.
    CuMatMulFullTN(CuHandle, X1.dValue, K.dGrad, Wk.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X1k.Grad = K.Grad * Wk^T.
    CuMatMulNT(CuHandle, K.dGrad, Wk.dValue, X1k.dGrad, SeqLen, ModelDim, ModelDim);

    // Wv.Grad = X1^T * V.Grad.
    CuMatMulFullTN(CuHandle, X1.dValue, V.dGrad, Wv.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X1v.Grad = V.Grad * Wv^T.
    CuMatMulNT(CuHandle, V.dGrad, Wv.dValue, X1v.dGrad, SeqLen, ModelDim, ModelDim);

    // 1B. Merge the Q, K, and V paths into X1.Grad.
    if DisplaySubstage then Writeln('' : Stage, '1B. Transform Backprop, sum Q, K, and V paths into X1 Grad');

    // X1.dGrad was zeroed at the beginning of this training window.
    CuAccumulateGrad(CuHandle, X1q.dGrad, X1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, X1k.dGrad, X1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, X1v.dGrad, X1.dGrad, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X1.Grad[0, 0], X1.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Grad after summing the Q, K, and V paths.', X1.Grad, G);
    end;

    // 1A. Backpropagate X1 = Norm(X).
    if DisplaySubstage then begin
      Writeln('' : Stage, '1A. Transform Backprop, Norm backward from X1 to X');
      PauseNNL;
    end;

    // Norm backward produces the X path gradient plus Gamma1 and Beta1 gradients.
    case NormKind of
      LayerNorm: LaunchLayerNormBackward(X1.dGrad, dXFromLN1, Gamma1.dValue, dLNXHat1, dLNInvStd1, Gamma1.dGrad, Beta1.dGrad, SeqLen, ModelDim);
      RMSNorm: LaunchRMSNormBackward(X1.dGrad, dXFromLN1, Gamma1.dValue, dLNXHat1, dLNInvStd1, Gamma1.dGrad, SeqLen, ModelDim);
    end;
    CuAccumulateGrad(CuHandle, dXFromLN1, X.dGrad, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X.Grad[0, 0], X.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Grad at the end of transform backpropagation.', X.Grad, G);
    end;

  end;
end;

end.
