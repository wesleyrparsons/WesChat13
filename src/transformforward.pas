unit TransformForward;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  { RTL and platform units }
  SysUtils,
  { WesChat units }
  Display, Global, Matrix, Util;

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

implementation

{// Accumulation for dimension repurposing.
procedure AccumulateX7Similarity(CuHandle: TcublasHandle; dX7: PSingle; var Stats: TX7SimilarityStats; SeqLen, ModelDim: Integer);
begin
  // CrossTemp = X7' * X7.
  CuMatMulFullTN(CuHandle, dX7, dX7, Stats.dCrossTemp, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

  // Cross += CrossTemp.
  CuMatAdd(CuHandle, Stats.dCrossTemp, Stats.dCross, Stats.dCross, ModelDim, ModelDim);

  // SumTemp[d] = sum over all SeqLen rows of X7[:,d].
  cudaMemset(Stats.dSumTemp, 0, ModelDim * SizeOf(Single));
  LaunchAddBiasRowsBackward(dX7, Stats.dSumTemp, SeqLen, ModelDim);

  // Sum += SumTemp.
  CuMatAdd(CuHandle, Stats.dSumTemp, Stats.dSum, Stats.dSum, 1, ModelDim);

  Inc(Stats.SampleCount, SeqLen);
end;}

{ Transformer forward pass }
// Run one transformer block forward.
procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h, HeadOffset, ValueBWidth: Integer;
  StepSeed: UInt64;
begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

    { Dropout seeds }
    // Set deterministic dropout seeds for this training step and block.
    if Training then begin
      StepSeed := GlobalSeed + UInt64(GlobalStep) * 100000 + UInt64(Blk) * 1000;
      ADropoutSeed := StepSeed + 1;
      MLPDropoutSeed := StepSeed + 100;
      RDropoutSeed := StepSeed + 200;
    end;

    // Display the input X matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X.Value[0, 0], X.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Value in transform, before any action.', X.Value, G);
    end;

    { Attention forward pass }

    Stage := Blk * 4 + 2;

    // 1A. Norm: X -> X1.
    if DisplayStage then Writeln('' : Stage, 'Stage  1, Block ', Blk, ', Transform Forward');
    if DisplaySubstage then Writeln('' : Stage, '1A. Transform Forward, Norm X');
    // Equation: X1 = Norm(X). X and X1 are L x D; Gamma1 and Beta1 are D.
    case NormKind of
      LayerNorm: LaunchLayerNormForward(X.dValue, X1.dValue, Gamma1.dValue, Beta1.dValue, dLNXHat1, dLNInvStd1, SeqLen, ModelDim);
      RMSNorm: LaunchRMSNormForward(X.dValue, X1.dValue, Gamma1.dValue, dLNXHat1, dLNInvStd1, SeqLen, ModelDim);
    end;

    // Display X1 after Norm.
    if VerboseTransform then begin
      cudaMemcpy(@X1.Value[0, 0], X1.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Value after Norm.', X1.Value, B);
    end;

    // 1B-C. Preserve X for the residual path and project X1 into Q, K, and V.
    if DisplaySubstage then Writeln('' : Stage, '1B-C. Transform Forward, project X1 into Q, K, and V');

    // Q = X1 * Wq. Q and X1 are L x D; Wq is D x D.
    CuMatMulNN(CuHandle, X1.dValue, Wq.dValue, Q.dValue, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@Q.Value[0, 0], Q.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Q in transform.', Q.Value, G);
    end;

    // K = X1 * Wk. K and X1 are L x D; Wk is D x D.
    CuMatMulNN(CuHandle, X1.dValue, Wk.dValue, K.dValue, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@K.Value[0, 0], K.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display K in transform.', K.Value, E);
    end;

    // V = X1 * Wv. V and X1 are L x D; Wv is D x D.
    CuMatMulNN(CuHandle, X1.dValue, Wv.dValue, V.dValue, SeqLen, ModelDim, ModelDim);

    // 1D. Apply RoPE independently within each attention head to Q and K.
    if DisplaySubstage then Writeln('' : Stage, '1D. Transform Forward, RoPE Q and K');
    LaunchRoPEForward(Q.dValue, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);
    LaunchRoPEForward(K.dValue, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);

    // 1E-G. Compute attention scores, apply causal masking and softmax, apply attention dropout, and multiply by V.
    if DisplaySubstage then Writeln('' : Stage, '1E-G. Transform Forward, attention scores, mask, softmax, attention dropout, and V projection');

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // 1E. ScoresHead1 = Qh * Kh^T / sqrt(HeadDim). Qh and Kh are L x H; ScoresHead1 is L x L.
      CuMatMulFullScaledNT(CuHandle, PSingle(Q.dValue) + HeadOffset, PSingle(K.dValue) + HeadOffset, ScoresHead1[h].dValue,
        SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);

      // 1F-a. Apply the autoregressive mask in place to ScoresHead1.
      LaunchAutoRegressiveMask(ScoresHead1[h].dValue, SeqLen);

      // 1F-b. Softmax: ScoresHead1 -> ScoresHead2.
      LaunchSoftmaxForward(ScoresHead1[h].dValue, ScoresHead2[h].dValue, SeqLen, SeqLen, 1.0);

      // 1F-c. During training, preserve the pre-dropout softmax probabilities in ScoresHead1, then apply attention dropout to ScoresHead2.
      if Training then begin
        cudaMemcpy(ScoresHead1[h].dValue, ScoresHead2[h].dValue, ScoresSize, cudaMemcpyDeviceToDevice);
        LaunchDropout(ScoresHead2[h].dValue, SeqLen * SeqLen, ADropOut, ADropoutSeed + h);
      end;

      // 1G. X2h = ScoresHead2 * Vh. ScoresHead2 is L x L; Vh is L x H; X2h is L x H.
      CuMatMulFullNN(CuHandle, ScoresHead2[h].dValue, PSingle(V.dValue) + HeadOffset, PSingle(X2.dValue) + HeadOffset,
        SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // Display attention probabilities for head 0 and the concatenated X2 matrix.
    if VerboseTransform then begin
      cudaMemcpy(@ScoresHead2[0].Value[0, 0], ScoresHead2[0].dValue, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display ScoresHead2[0] after softmax and attention dropout.', ScoresHead2[0].Value, G);
      cudaMemcpy(@X2.Value[0, 0], X2.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X2 after attention and head concatenation.', X2.Value, B);
    end;

    // 1H. Project the concatenated attention output through W0: X2 -> X3.
    if DisplaySubstage then Writeln('' : Stage, '1H. Transform Forward, obtain X3 by weighting X2 by W0');
    // Equation: X3 = X2 * W0. X2 and X3 are L x D; W0 is D x D.
    CuMatMulNN(CuHandle, X2.dValue, W0.dValue, X3.dValue, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X3.Value[0, 0], X3.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3 in transform.', X3.Value, B);
    end;

    // 1I. Attention residual merge: X4 = X + X3.
    if DisplaySubstage then Writeln('' : Stage, '1I. Transform Forward, obtain X4 from X and X3');
    CuMatAdd(CuHandle, X.dValue, X3.dValue, X4.dValue, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X4.Value[0, 0], X4.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Value after adding the attention residual.', X4.Value, G);
    end;

    // 1J. Norm: X4 -> X5.
    if DisplaySubstage then Writeln('' : Stage, '1J. Transform Forward, obtain X5 from Norm X4');
    // Equation: X5 = Norm(X4). X4 and X5 are L x D; Gamma2 and Beta2 are D.
    case NormKind of
      LayerNorm: LaunchLayerNormForward(X4.dValue, X5.dValue, Gamma2.dValue, Beta2.dValue, dLNXHat2, dLNInvStd2, SeqLen, ModelDim);
      RMSNorm: LaunchRMSNormForward(X4.dValue, X5.dValue, Gamma2.dValue, dLNXHat2, dLNInvStd2, SeqLen, ModelDim);
    end;

    if VerboseTransform then begin
      cudaMemcpy(@X5.Value[0, 0], X5.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X5.Value before the feed-forward network.', X5.Value, G);
    end;

    { Feed-forward forward pass }
    Stage := Stage + 2;

    // 2A. Expand X5 through W1: X5 -> Hidden1.
    if DisplayStage then Writeln('' : Stage, 'Stage  2, Block ', Blk, ', Transform Forward');
    if DisplaySubstage then Writeln('' : Stage, '2A. Transform Forward, obtain Hidden1 from X5 and W1');
    // Equation: Hidden1 = X5 * W1. X5 is L x D; W1 is D x DB; Hidden1 is L x DB.
    CuMatMulNN(CuHandle, X5.dValue, W1.dValue, Hidden1.dValue, SeqLen, ModelDimProj, ModelDim);

    // 2B. Add b1 to every row of Hidden1.
    if DisplaySubstage then Writeln('' : Stage, '2B. Transform Forward, add b1 to Hidden1');
    LaunchAddBiasRows(Hidden1.dValue, b1.dValue, SeqLen, ModelDimProj);

    if VerboseTransform then begin
      cudaMemcpy(@Hidden1.Value[0, 0], Hidden1.dValue, HiddenSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Hidden1.Value after adding b1 and before ReLU.', Hidden1.Value, G);
    end;

    // 2C-D. Activation/gating, MLP dropout, and W2 projection. C1 = Gating. C2 = Activation. C3 = Dropout.
    if DisplaySubstage then
      Writeln('' : Stage, '2C-D. Transform Forward, MLP gating and contraction');

    case MLPGatingKind of
      SplitSwiGLU: begin
        // 2C1. Gating. Logical dimensions: Hidden1: 768 columns (512 gate + 256 value A). Hidden2: 768 columns (256 value B + 512 gated output).
        // W2: first 512 rows for contraction, last 256 rows for value B projection.
        ValueBWidth := 2 * GatedDim - ModelDimProj;

        // Calculate value B = X5 * transpose(W2[512..767]). Store in Hidden2 columns 0..255, pitch 768.
        CuMatMulFullNT(CuHandle, X5.dValue, PSingle(W2.dValue) + GatedDim * ModelDim,
          Hidden2.dValue, SeqLen, ValueBWidth, ModelDim, ModelDim, ModelDim, ModelDimProj);

        // 2C2. Activation. Gate and value projections -> gated output. Output goes into Hidden2 columns 256..767.
        LaunchSwiGLUSplit(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj, GatedDim);

        // 2C3. Dropout. Apply dropout directly to the gated output.
        // Hidden2 columns 0..255 contain Value B. Hidden2 columns 256..767 contain the gated output.
        if Training then
          LaunchDropoutStrided(Hidden2.dValue, SeqLen, GatedDim,
            ModelDimProj, ValueBWidth, MLPDropOut, MLPDropoutSeed);

        // 2D. Contract the gated output through W2.
        // Hidden2 has a physical row pitch of 768. W2[0..511] is the down-projection matrix.
        CuMatMulFullNN(CuHandle, PSingle(Hidden2.dValue) + ValueBWidth, W2.dValue, X6.dValue,
          SeqLen, ModelDim, GatedDim, ModelDimProj, ModelDim, ModelDim);
      end;

      SelfGating: begin
        // 2C2. Activation. Hidden2 = Hidden1 * SiLU(Hidden1).
        LaunchSiLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);

        LaunchHadamard(Hidden1.dValue, Hidden2.dValue, Hidden2.dValue, SeqLen * ModelDimProj);

        // 2C3. Dropout. Apply dropout directly to the gated output.
        if Training then
          LaunchDropout(Hidden2.dValue, SeqLen * ModelDimProj, MLPDropOut, MLPDropoutSeed);

        // 2D. Contract the gated output through W2.
        CuMatMulNN(CuHandle, Hidden2.dValue, W2.dValue, X6.dValue, SeqLen, ModelDim, ModelDimProj);
      end;

      NoGating: begin
        // 2C2. Activation.
        case ActivationKind of
          ReLU:      LaunchReLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);
          LeakyReLU: LaunchLeakyReLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj, ActivationAlpha);
          GELU:      LaunchGELUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);
          SiLU:      LaunchSiLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);
          ELU:       LaunchELUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj, ActivationAlpha);
          Softplus:  LaunchSoftplusForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);
          Mish:      LaunchMishForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);
        end;

        // 2C3. Dropout.
        if Training then
          LaunchDropout(Hidden2.dValue, SeqLen * ModelDimProj, MLPDropOut, MLPDropoutSeed);

        // 2D. Contract the gated output through W2.
        CuMatMulNN(CuHandle, Hidden2.dValue, W2.dValue, X6.dValue, SeqLen, ModelDim, ModelDimProj);
      end;
    end;

    // 2E. Add b2 to every row of X6.
    if DisplaySubstage then
      Writeln('' : Stage, '2E. Transform Forward, add b2 to X6');

    LaunchAddBiasRows(X6.dValue, b2.dValue, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X6 in transform after contraction and b2.', X6.Value, B);
    end;

    // 2F. Apply residual dropout to X6 during training, then merge with X4: X7 = X4 + X6.
    if DisplaySubstage then Writeln('' : Stage, '2F. Transform Forward, residual dropout and obtain X7 from X4 and X6');

    if Training then
      LaunchDropout(X6.dValue, SeqLen * ModelDim, RDropout, RDropoutSeed);

    CuMatAdd(CuHandle, X4.dValue, X6.dValue, X7.dValue, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X7.Value[0, 0], X7.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X7.Value after the feed-forward residual merge.', X7.Value, B);
    end;

    // For this block, accumulate the matrix of X7 correlations. For dim repurposing
    // AccumulateX7Similarity(CuHandle, X7.dValue, WModelState.X7Stats[Blk], SeqLen, ModelDim);
  end;
end;

end.
