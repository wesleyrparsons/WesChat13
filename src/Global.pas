unit Global;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.3, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  Classes;

{ Program constants }
const
  Version: shortstring = '1.3';
  ModelMagic: array[0..7] of Char = ('W','E','S','2','M','O','D','L');
  RecentCount = 10;                               // Number of values used for rolling means in training.
  DisplayLength = 100;                            // Number of corpus bytes or tokens normally displayed.
  WES3ModelOptionsVersion = 1;                    // Version for additional model options.

{ Model dimensions }
const
  MaxEpochs = 1000000;                            // Maximum number of epochs over the tokenized corpus.
  ModelDim = 192;                                 // Embedding and transformer dimension.
  Scale = Sqrt(ModelDim);                         // Transformer embedding scale, sqrt(ModelDim).
  Proj = 4;                                       // MLP projection multiplier in 1/3s. 12 is 12/3 = 4. 8 is 8/3.
  ModelDimProj = ModelDim * Proj;                 // Projected MLP dimension.
  GatedDim = ModelDim * 8 div 3;                  // Projected MLP dimension when gated.
  SeqLen = 256;                                   // Sequence length.
  nHead = 8;                                      // Number of attention heads.
  HeadDim = ModelDim div nHead;                   // Dimension of one attention head.
  nBlock = 6;                                     // Number of transformer blocks.
  InvSqrtHeadDim: Single = 1 / Sqrt(HeadDim);     // Attention scaling factor.
  MaxSymbols = 50400;                             // Maximum Wes/UD symbol count during BPE construction.
  DimVocab = 50400;                               // Physical model vocabulary capacity; must be >= nVocab.

{ Activation function parameters }
const
  LeakyReLUAlpha = 0.01;
  ELUAlpha = 1.0;

{ GPT-2 token constants }
const
  GPT2BaseVocabSize = 50257;                      // Official GPT-2 vocabulary count; IDs 0..50256.
  GPT2EOS = 50256;                                // Official GPT-2 end-of-text token.
  GPT2PAD = 50257;                                // WesChat extension.
  GPT2BOS = 50258;                                // WesChat extension.
  GPT2UNK = 50259;                                // WesChat extension; normally not needed.
  GPT2ModelVocabSize = 50260;                     // WesChat GPT-2 vocabulary count; IDs 0..50259.

{ Wes and UD token boundaries }
const
  FirstTagToken = 260;                            // First reserved UD tag token.
  UDTagBoundary = 400;                            // First token after the reserved UD range.
  FirstMergedToken = 400;                         // First ID available to learned BPE symbols in the current layout.
  TokBOS = 256;
  TokEOS = 257;
  TokPAD = 258;
  TokNUL = 259;

{ UD parts of speech }
const
  TokNoun  = 260;                                 // |noun
  TokVerb  = 261;                                 // |verb
  TokAdj   = 262;                                 // |adj
  TokAdv   = 263;                                 // |adv
  TokPrep  = 264;                                 // |prep
  TokDet   = 265;                                 // |det
  TokPron  = 266;                                 // |pron
  TokAux   = 267;                                 // |aux
  TokSConj = 268;                                 // |sconj
  TokCConj = 269;                                 // |cconj
  TokPart  = 270;                                 // |part
  TokIntj  = 271;                                 // |intj
  TokNum   = 272;                                 // |num
  TokPropn = 273;                                 // |propn
  TokX     = 274;                                 // |x
  TokSym   = 275;                                 // |sym
  TokPunct = 276;                                 // |punct

{ UD number }
const
  TokSing = 277;                                  // |sg
  TokPlur = 278;                                  // |pl

{ UD person }
const
  TokPerson1 = 279;                               // |1p
  TokPerson2 = 280;                               // |2p
  TokPerson3 = 281;                               // |3p

{ UD case }
const
  TokNom = 282;                                   // |nom
  TokAcc = 283;                                   // |acc
  TokGen = 284;                                   // |gen
  TokDat = 285;                                   // |dat
  TokLoc = 286;                                   // |loc
  TokIns = 287;                                   // |ins
  TokVoc = 288;                                   // |voc

{ UD gender }
const
  TokMasc   = 289;                                // |masc
  TokFem    = 290;                                // |fem
  TokNeut   = 291;                                // |neut
  TokCommon = 292;                                // |common

{ UD tense }
const
  TokPast = 293;                                  // |past
  TokPres = 294;                                  // |pres
  TokFut  = 295;                                  // |fut

{ UD mood }
const
  TokMoodInd  = 296;                              // |ind
  TokMoodImp  = 297;                              // |imp
  TokMoodSub  = 298;                              // |sub
  TokMoodCond = 299;                              // |cond
  TokMoodOpt  = 300;                              // |opt

{ UD verb form }
const
  TokVerbFin  = 301;                              // |fin
  TokVerbInf  = 302;                              // |inf
  TokVerbGer  = 303;                              // |ger
  TokVerbPart = 304;                              // |participle
  TokVerbConv = 305;                              // |conv

{ UD voice }
const
  TokVoiceAct  = 306;                             // |act
  TokVoicePass = 307;                             // |pass
  TokVoiceMid  = 308;                             // |mid

{ UD aspect }
const
  TokAspectImp   = 309;                           // |impf
  TokAspectPerf  = 310;                           // |perf
  TokAspectProg  = 311;                           // |prog
  TokAspectProsp = 312;                           // |prosp

{ UD degree }
const
  TokDegreePos = 313;                             // |pos
  TokDegreeCmp = 314;                             // |cmp
  TokDegreeSup = 315;                             // |sup
  TokDegreeAbs = 316;                             // |abs

{ UD definiteness }
const
  TokDefiniteDef = 317;                           // |def
  TokDefiniteInd = 318;                           // |indef

{ UD pronoun type }
const
  TokPronArt = 319;                               // |art
  TokPronDem = 320;                               // |dem
  TokPronInt = 321;                               // |int
  TokPronPrs = 322;                               // |prs
  TokPronRel = 323;                               // |rel
  TokPronInd = 324;                               // |indpron
  TokPronNeg = 325;                               // |negpron
  TokPronTot = 326;                               // |tot

{ UD possessive and reflexive }
const
  TokPoss = 327;                                  // |poss
  TokRefl = 328;                                  // |refl

{ UD polarity }
const
  TokPolarityNeg = 329;                           // |neg
  TokPolarityPos = 330;                           // |positive

{ UD numeral type }
const
  TokNumCard = 331;                               // |card
  TokNumOrd  = 332;                               // |ord
  TokNumFrac = 333;                               // |frac
  TokNumMult = 334;                               // |mult
  TokNumSets = 335;                               // |sets
  TokNumDist = 336;                               // |dist

{ UD numeral form }
const
  TokNumDigit = 337;                              // |digit
  TokNumWord  = 338;                              // |numword
  TokNumRoman = 339;                              // |roman

{ UD miscellaneous lexical properties }
const
  TokAbbr    = 340;                               // |abbr
  TokForeign = 341;                               // |foreign
  TokTypo    = 342;                               // |typo

{ UD animacy }
const
  TokAnim     = 343;                              // |anim
  TokInan     = 344;                              // |inan
  TokHuman    = 345;                              // |human
  TokNonHuman = 346;                              // |nonhuman

{ Basic program types }
type
  TcublasHandle = Pointer;
  TTokenizerKind = (WesTokenizer, UDTokenizer, GPT2Tokenizer);
  TLearningStyle = (FlatLearning, FastLearning, SlowLearning, RolledOffLearning);
  TNormKind = (LayerNorm, RMSNorm);
  TActivationKind = (ReLU, LeakyReLU, GELU, SiLU, ELU, Softplus, Mish);
  TMLPGatingKind = (NoGating, SelfGating, SplitSwiGLU);
  TPart = (B, E, F, G);

{ Corpus and utility vector types }
type
  TBooleanVector = array of Boolean;
  TIVector = array of Integer;                    // Dynamic integer token vector.
  TBVector = array of Byte;                       // Dynamic UTF-8 byte corpus.
  TRBSVector = array of RawByteString;            // Dynamic raw-byte-string vector.
  TSVector = array of String;                     // Dynamic string vector.
  TFVector = array of Single;                     // Dynamic Single vector, used by RoPE.
  TSymbolTable = TRBSVector;                      // Symbol strings indexed by token ID.

{ Fixed model vector and matrix types }
type
  TSeqVector = array[0..ModelDim - 1] of Single;                              // D
  TSeqVectorProj = array[0..ModelDimProj - 1] of Single;                       // DB
  TDimVector = array[0..SeqLen - 1] of Single;                                 // L
  TIDimVector = array[0..SeqLen - 1] of Integer;                               // L
  THeadVector = array[0..HeadDim - 1] of Single;                               // H
  TVocabVector = array[0..DimVocab - 1] of Single;                             // DV
  TFSVector = array[0..SeqLen - 1] of Single;                                  // L
  TSeqMatrix = array[0..SeqLen - 1] of TSeqVector;                             // L x D
  TWeightMatrix = array[0..ModelDim - 1] of TSeqVector;                        // D x D
  TWeightProjMatrix = array[0..ModelDim - 1] of TSeqVectorProj;                // D x DB
  TWeightProjMatrixT = array[0..ModelDimProj - 1] of TSeqVector;               // DB x D
  THiddenMatrix = array[0..SeqLen - 1] of TSeqVectorProj;                      // L x DB
  TScoresMatrix = array[0..SeqLen - 1] of TDimVector;                          // L x L
  TSeqVocabMatrix = array[0..SeqLen - 1] of TVocabVector;                      // L x DV
  TEmbeddingsMatrix = array[0..DimVocab - 1] of TSeqVector;                    // DV x D
  // Dimension repurposing.
  {type
    TX7SimilarityStats = record
      dCross: PSingle;        // ModelDim x ModelDim
      dSum: PSingle;          // ModelDim
      dCrossTemp: PSingle;    // ModelDim x ModelDim
      dSumTemp: PSingle;      // ModelDim
      SampleCount: Int64;
    end;
    TX7SimilarityArray = array of TX7SimilarityStats;}

{ Model tensor types }
type
  TSeqTensor = record
    Value, Grad: TSeqMatrix;
    dValue, dGrad: PSingle;
  end;

  TSeqVectorTensor = record
    Value, Grad: TSeqVector;
    dValue, dGrad: PSingle;
  end;

  THiddenTensor = record
    Value, Grad: THiddenMatrix;
    dValue, dGrad: PSingle;
  end;

  TSeqVectorProjTensor = record
    Value, Grad: TSeqVectorProj;
    dValue, dGrad: PSingle;
  end;

  TWeightTensor = record
    Value, Grad: TWeightMatrix;
    dValue, dGrad: PSingle;
  end;

  TWeightProjTensor = record
    Value, Grad: TWeightProjMatrix;
    dValue, dGrad: PSingle;
  end;

  TWeightProjTensorT = record
    Value, Grad: TWeightProjMatrixT;
    dValue, dGrad: PSingle;
  end;

  TScoresHeadTensor = record
    Value, Grad: TScoresMatrix;
    dValue, dGrad: PSingle;
  end;

  TEmbeddingsTensor = record
    Value, Grad: TEmbeddingsMatrix;
    dValue, dGrad: PSingle;
  end;

{ AdamW tensor types }
type
  TAdamWeightTensor = record
    M, V: TWeightMatrix;
    dM, dV: PSingle;
  end;

  TAdamWeightProjTensor = record
    M, V: TWeightProjMatrix;
    dM, dV: PSingle;
  end;

  TAdamWeightProjTensorT = record
    M, V: TWeightProjMatrixT;
    dM, dV: PSingle;
  end;

  TAdamSeqVectorTensor = record
    M, V: TSeqVector;
    dM, dV: PSingle;
  end;

  TAdamSeqVectorProjTensor = record
    M, V: TSeqVectorProj;
    dM, dV: PSingle;
  end;

  TAdamEmbeddingsTensor = record
    M, V: TEmbeddingsMatrix;
    dM, dV: PSingle;
  end;

{ Training state types }
type
  TAdaptiveLRState = record
    Initialized: Boolean;
    PrevLoss: Double;
    PrevParamRMS: Double;
    PrevUpdateRatio: Double;
    PrevMRMS: Double;
    PrevSqrtVRMS: Double;
    PrevMaxGammaRMS: Double;
    ConsecutiveWorse: Integer;
    ConsecutiveFlat: Integer;
    LastLRChangeEpoch: Integer;
  end;

  TTrainingCheckpoint = record
    GlobalStep: Int64;
    CompletedEpochs: Integer;
    LearningStyle: TLearningStyle;
    LearningRate: Double;
    OverrideLearningRate: Double;
    BaseLearningRate: Double;
    FloorLearningRate: Double;
    RollOff: Double;
    WeightDecay: Double;
    ClipLimit: Double;
    TTemperature: Double;
    ITemperature: Double;
    ADropOut: Double;
    RDropOut: Double;
    MLPDropOut: Double;
    ShuffleWindows: Boolean;
    Stride: Integer;
    StartStride: Integer;
    GlobalSeed: QWord;
    AdamWStep: Int64;
    AdamBeta1: Double;
    AdamBeta2: Double;
    AdamEpsilon: Double;
    AdaptiveLR: Boolean;
    AdaptiveLRState: TAdaptiveLRState;
    MinLoss: Double;
    MinLossEpoch: Integer;
    BestSavedLoss: Double;
    LastBestSaveEpoch: Integer;
  end;

{ Trainable model parameter types }
type
  TParamBlock = array[0..nBlock - 1] of record
    Wq, Wk, Wv, W0:                TWeightTensor;
    W1:                            TWeightProjTensor;
    W2:                            TWeightProjTensorT;
    b1:                            TSeqVectorProjTensor;
    b2:                            TSeqVectorTensor;
    Gamma1, Beta1, Gamma2, Beta2: TSeqVectorTensor;
  end;

  TWModelParams = record
    Embeddings: TEmbeddingsTensor;
    ParamBlock: TParamBlock;
  end;

{ Non-trainable model state types }
type
  TStateBlock = array[0..nBlock - 1] of record
    X, X1, X2, X3, X4, X5, X6, X7: TSeqTensor;               // Activations at the transformer stages.
    X1q, X1v, X1k: TSeqTensor;                               // Inputs used to form Q, K, and V.
    Q, K, V: TSeqTensor;                                     // Attention query, key, and value matrices.
    ScoresHead1, ScoresHead2: array[0..nHead - 1] of TScoresHeadTensor;
    Hidden1, Hidden2: THiddenTensor;                         // MLP activations.

    LNInvStd1: TFSVector;                                    // LayerNorm 1 inverse-standard-deviation cache.
    dLNInvStd1: PSingle;
    LNXhat1: TSeqMatrix;                                     // LayerNorm 1 normalized-value cache.
    dLNXhat1: PSingle;
    LNInvStd2: TFSVector;                                    // LayerNorm 2 inverse-standard-deviation cache.
    dLNInvStd2: PSingle;
    LNXhat2: TSeqMatrix;                                     // LayerNorm 2 normalized-value cache.
    dLNXhat2: PSingle;

    ADropoutSeed: UInt64;
    MLPDropoutSeed: UInt64;
    RDropoutSeed: UInt64;

    dX4FromLN2: PSingle;                                     // Temporary gradient from LayerNorm 2.
    dXFromLN1: PSingle;                                      // Temporary gradient from LayerNorm 1.
  end;

  TWModelState = record
    StateBlock: TStateBlock;
    // X7Stats: TX7SimilarityArray;                          // For dim repurposing.
    InvFreq: TFVector;                                       // RoPE inverse frequencies.
    dInvFreq: PSingle;
    Probs, TopGradient: TSeqVocabMatrix;                     // Output probabilities/logits and top gradient.
    dProbs, dTopGradient: PSingle;
    dRowLoss: PSingle;                                       // Cross-entropy row-loss storage.
    // dWeakX7Vector: PSingle;                               // For testing dimrepurposing.
    // dX7Backup: PSingle;
  end;

{ AdamW model state types }
type
  TAdamParamBlock = array[0..nBlock - 1] of record
    Wq, Wk, Wv, W0: TAdamWeightTensor;
    W1: TAdamWeightProjTensor;
    W2: TAdamWeightProjTensorT;
    b1: TAdamSeqVectorProjTensor;
    b2: TAdamSeqVectorTensor;
    Gamma1, Beta1, Gamma2, Beta2: TAdamSeqVectorTensor;
  end;

  TWAdamWState = record
    Embeddings: TAdamEmbeddingsTensor;
    ParamBlock: TAdamParamBlock;
  end;

{ Runtime control and verbosity }
var
  DoNotPause: Boolean = False;                    // Disable explicit pauses.
  PauseIfKeyPressed: Boolean = True;              // Allow keyboard-triggered pauses.
  StopTraining: Boolean;                          // Set when training should stop for user input.
  TrainSuccess: Boolean = False;                  // Training completed successfully.
  YesToAll: Boolean = True;                       // Say Yes to the option.

  VerboseTokenize: Boolean = False;               // Display tokenization and symbolization details.
  VeryVerboseTokenize: Boolean = False;           // Display additional tokenization detail.
  VerboseTransform: Boolean = False;              // Display transformer tensors and intermediate work.
  VerboseInfer: Boolean = False;                  // Display detailed inference work.
  DetailInfer: Boolean = False;                   // Display selected inference detail.

  DisplayCorpus: Boolean = False;                 // Display corpus data.
  DisplayWindow: Boolean = False;                 // Display the current SeqLen training window.
  DisplayTokenWork: Boolean = False;              // Display tokenization work.
  DisplayMergeWork: Boolean = False;              // Display BPE merge work.
  DisplayCorpusVerification: Boolean = False;     // Verify reconstructed corpus.
  DisplayTokenVerification: Boolean = False;      // Verify reconstructed tokens.
  DisplayEachByteRead: Boolean = False;           // Display individual bytes while reading.

  DisplayStage: Boolean = False;                  // Display training/transform stage progress.
  DisplaySubstage: Boolean = False;               // Display training/transform substage progress.
  DisplayEpoch: Boolean = True;                   // Display epoch progress.

  SaveFiles: Boolean = True;                      // Permit normal file output.
  SaveTokenizationFiles: Boolean = True;          // Permit tokenizer-generated files.

{ BPE and tokenization settings }
var
  MaxMerges: Integer = 60000;                     // Maximum number of BPE merges.
  MaxPairCount: Integer = 800000;                 // Maximum number of BPE pair-count entries.
  TokenizerKind: TTokenizerKind;                  // Active Wes, UD, or GPT-2 tokenizer.
  BOS: Integer = 256;                             // Beginning-of-sequence token.
  EOS: Integer = 257;                             // End-of-sequence token.
  PAD: Integer = 258;                             // Padding token.
  UNK: Integer = 259;                             // Unknown token.

  UDPipeFileName: string = 'ud\udpipe.exe';
  UDModelFileName: string = 'ud\english-ewt-ud-2.4-190531.udpipe';
  Vocab: TStringList;                             // GPT-2 vocabulary.
  RepurposeDimsOn: Boolean = False;               // Repurpose redundant dimensions. This is turned off.


{ WesChat installation folders }
  var
    WesChatRoot: string = '';       // Root of the entire WesChat installation.
    SharedCorpusDir: string = '';   // Shared input corpora.
    DataDir: string = '';           // GPT-2 vocabulary, merges, UD data.
    ToolsDir: string = '';          // External utilities.

{ Work folders and file names }
var
  ExistingWorkRoot: string = '';                  // Root containing predefined work folders.
  WorkRoot: string = '';                          // Active work folder.
  WorkingDir: string = '';                        // Active working directory.
  WorkingName: string = '';                       // Base name used for current work.

  CorpusDir: string = '';                         // Corpus file folder.
  SymbolDir: string = '';                         // Symbol-table folder.
  MergeDir: string = '';                          // Merge-table folder.
  TokenDir: string = '';                          // Token-list folder.
  ModelDir: string = '';                          // Model folder.
  LogDir: string = '';                            // Log folder.
  RunDir: string = '';                            // Run-file folder.
  ListDir: string = '';                           // File-list folder.
  ScratchDir: string = '';                        // Scratch folder.

  CurrentBaseName: string = 'weschat';            // Base name used to construct default output names.
  CorpusFileNames: array of string;               // Corpus files contributing to the current work.
  CorpusFileName: string = '';
  TokenFileName: string = '';
  SymbolFileName: string = '';
  VocabFileName: string = '';
  MergeFileName: string = '';
  ModelFileName: string = '';
  BestModelFileName: string = '';
  RunFileName: string = '';
  ListFile: string = '';
  LogFileName: string = '';
  TrainLogFileName: string = '';

{ Corpus, symbol, and token state }
var
  nCorpus: Integer;                               // Corpus length in bytes.
  CorpusID: UInt64;                               // Corpus identifier.
  CorpusFileInfo: string;                         // Descriptive information about the corpus.
  MultipleFileName: string;                       // Name associated with a combined multi-file corpus.

  SymbolTable: TSymbolTable;                      // Current symbol table.
  nSymbols: Integer;                              // Number of symbols produced or loaded.
  nVocab: Integer;                                // Active model vocabulary size.
  MergeCount: Integer;                            // Number of merges in the current symbolization run.
  nMerges: Integer;                               // Total number of learned merges.
  FromSymbolTable: Boolean = False;               // Current tokenization is using an existing symbol table.

  nTokenizedCorpus: Integer;                      // Length of padded tokenized corpus.
  RawTokenCount: Integer;                         // Number of tokens before padding.
  PaddedTokenCount: Integer;                      // Number of tokens after padding.
  TokenID: TIVector;                              // General dynamic token-ID vector.

  InputTokens: TIDimVector;                       // Current model input tokens.
  dInputTokens: PInteger;                         // Device input tokens.
  TargetTokens: TIDimVector;                      // Targets shifted one token ahead.
  dTargetTokens: PInteger;                        // Device target tokens.

{ Training settings and state }
var
  Training: Boolean = False;                      // True while training; enables training behavior such as dropout.
  NormKind: TNormKind = LayerNorm;                // Layer norming kind.
  MLPGatingKind: TMLPGatingKind = NoGating;       // Gating kind. Default is none.
  ProjText: string = '4';                         // Text version of Proj.
  ActivationKind: TActivationKind = SiLU;         // Activation kind.
  ActivationAlpha: Single = 0.01;                 // Activation parameter.
  LearningStyle: TLearningStyle = SlowLearning;   // Learning-rate schedule style.
  ShuffleWindows: Boolean = True;                 // Shuffle training windows each epoch.

  BaseLearningRate: Double = 0.000100;            // Base learning rate.
  FloorLearningRate: Double = 0.000005;           // Minimum scheduled learning rate.
  OverrideLearningRate: Double = -1.00000;        // Negative means no manual learning-rate override.
  RollOff: Double = 0.999900;                     // Multiplicative learning-rate rolloff.
  WeightDecay: Double = 0.000100;                 // AdamW weight decay.
  DecayScale: Double;                             // 1.0 - LearningRate * WeightDecay.
  LearningRate: Double;                           // Current derived learning rate.

  TTemperature: Single = 1.00000;                 // Training softmax temperature.
  ITemperature: Single = 1.00000;                 // Inference softmax temperature.
  ClipLimit: Single = 0.40000;                    // Gradient clipping limit.

  ADropOut: Single = 0.01;                        // Attention dropout probability.
  MLPDropOut: Single = 0.01;                      // MLP dropout probability.
  RDropout: Single = 0.01;                        // Residual dropout probability.
  GlobalSeed: UInt64 = 123456789;                 // Base random seed.

  Stride: Integer = 128;                          // Training-window stride.
  StartStride: Integer = 17;                      // Starting offset; chosen coprime with Stride.
  GlobalStep: Int64;                              // Incremented once per training window.
  CompletedEpochs: Integer;                       // Number of completed training epochs.
  Stage: Byte;                                    // Current training/transform display stage.

  AdaptiveLR: Boolean = True;                     // Enable adaptive learning-rate logic.
  AdaptiveLRState: TAdaptiveLRState;
  MinLoss: Double;
  MinLossEpoch: Integer;
  BestSavedLoss: Double;
  LastBestSaveEpoch: Integer;

  TrainLogFile: Text;                             // Training log stream.
  TrainLogOpen: Boolean = False;                  // True while the training log is open.

{ AdamW state }
var
  AdamBeta1: Single = 0.90000;
  AdamBeta2: Single = 0.99900;
  AdamEpsilon: Single = 1.0e-8;
  AdamWStep: Int64 = 0;
  AdamWStateLoaded: Boolean = False;

{ CUDA and library state }
var
  CuHandle: TcublasHandle;                        // cuBLAS handle.
  CudaAllocated: Boolean = False;                 // True after CUDA model storage is allocated.
  DebugCudaChecks: Boolean = False;               // Enable expensive CUDA error checks.
  CublasPresent: Boolean;                         // cuBLAS DLL is available.                                Ned these?
  CudartPresent: Boolean;                         // CUDA runtime DLL is available.
  WesChatKernelPresent: Boolean;                  // WesChat CUDA kernel DLL is available.
  ParamsNeedCopyToDevice: Boolean = True;         // Host parameters must be copied to the GPU before use.

{ Model and data presence state }
var
  NewModel: Boolean = True;                       // Initialize parameters for a new model.
  CorpusPresent: Boolean = False;
  SymbolTablePresent: Boolean = False;
  MergeTablePresent: Boolean = False;
  TokenizedCorpusPresent: Boolean = False;
  ModelPresent: Boolean = False;
  QueryPresent: Boolean = False;

{ Timing state }
var
  Mt0, Mt1, t0, t1, StopTime: TDateTime;

implementation

begin
end.
