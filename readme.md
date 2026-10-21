# WesChat

**WesChat** is a homebrew transformer-based language model written in **Free Pascal** and accelerated with **NVIDIA CUDA** and **cuBLAS**.

The project is built largely from the ground up as a way to understand how modern language models actually work at the level of tokens, matrices, gradients, GPU memory, and training algorithms.

WesChat is an educational and experimental implementation that avoids high-level machine-learning frameworks and instead implements the major components directly.

---

## Background

I started programming around 1970 using Fortran and BASIC in a time-sharing environment. My first compiler on a personal computer was **Turbo Pascal**, and Pascal has remained my language of choice ever since.

Most modern LLM development is done in Python using frameworks such as PyTorch or TensorFlow. WesChat takes a different approach: the tokenizer, transformer, training loop, inference code, optimization, serialization, CUDA interface, and many GPU kernels are implemented directly in Pascal and CUDA.

This makes it possible to see and control what is happening at each stage rather than relying on a framework to hide the implementation details.

My interest in coding an LLM was influenced by Loglan. a constructed language I have been involved with since college.

---

## Status

WesChat is **actively under development**. There is no release candidate yet.

The current implementation includes:

* Transformer training and inference
* NVIDIA CUDA acceleration
* cuBLAS matrix operations
* Custom CUDA kernels
* AdamW optimization
* Weight tying
* Dropout
* Selectable normalization, including LayerNorm and RMSNorm
* Selectable MLP activation functions, including ReLU, SiLU, and GeLU
* MLP gating, including self-gating and split SwiGLU
* Rotary positional embeddings
* Model checkpointing and resume support
* Multiple tokenizer formats
* Universal Dependencies grammatical tagging

A flow chart of the program is available in `flowchart.pas`.

---

## Tokenization

WesChat currently supports three tokenizer modes.

### WesTokenizer

The native WesChat tokenizer is a **byte-level BPE tokenizer**.

Features include:

* Initial vocabulary based on byte values
* Learned BPE symbols
* Deterministic longest-prefix token matching
* Greedy longest-symbol decoding
* Symbol-table generation from a corpus
* Saving and loading symbol tables
* Tokenization of individual files or lists of corpus files

---

### UDTokenizer

WesChat also supports a tokenizer based on the native Wes tokenizer with reserved **Universal Dependencies (UD)** grammatical tokens.

A normal corpus can be processed and prefixed with UD tags to learn both the text and explicit grammatical information.

During inference:

1. The user's ordinary text query is prefixed with UD tags.
2. The UD-tagged result is tokenized.
3. The model receives both text and grammatical tokens as context.
4. Generated UD tags remain in the model's autoregressive context.
5. The tags are stripped from the text shown to the user.

---

### GPT2Tokenizer

WesChat also supports GPT-2-compatible tokenization using the standard GPT-2 vocabulary and merge table.

This includes:

* `vocab.json`
* `merges.txt`
* GPT-2 byte encoding
* GPT-2 BPE merging
* GPT-style token decoding

---

## Corpus Processing

WesChat can:

* Read a single corpus
* Read a list of corpus files
* Build a symbol table
* Load an existing symbol table
* Tokenize a corpus
* Save tokenized data
* Generate a UD-tagged corpus.
* Track corpus, symbol, token, and model files within a work directory

---

## Model Architecture

WesChat implements a decoder-style autoregressive transformer.

The architecture is configurable, but current experiments typically use models in the range of:

* 6 transformer blocks
* 8 attention heads
* Model dimensions of 192
* Sequence lengths of 256
* Float32 parameters and activations

### Transformer Block

Each block includes:

**Pre-Normalization**

Normalization is performed before the attention and MLP sublayers.

WesChat supports selectable normalization:

* LayerNorm
* RMSNorm

LayerNorm uses learned gamma and beta parameters. RMSNorm uses gamma without beta and reuses the same transformer-block architecture and normalization cache buffers.

**Multi-Head Self-Attention**

* Query, key, and value projections
* Multiple attention heads
* Causal masking
* Softmax attention
* Attention output projection
* Residual connection

**Rotary Positional Embeddings**

RoPE is used to provide positional information to attention.

**MLP**

The feed-forward section includes:

* Input projection
* Selectable activation function
* MLP gating
* Output projection
* Residual connection

Current activation experiments include:

* ReLU
* Leaky ReLU
* GELU
* SiLU
* ELU
* Softplus
* Mish

The activation is selected without changing the surrounding MLP architecture. Activations that require an alpha parameter can store that value with the model configuration.

Current gating functions include:
* No gating
* Self-gating
* Split SwiGLU

**Dropout**

Dropout can be applied to attention, residual, and MLP paths.

---

## Output Head

The output head produces logits over the vocabulary followed by softmax probabilities.

WesChat uses **weight tying**, so the token embedding matrix is also used by the output head.

Training uses cross-entropy loss.

Inference supports:

* Temperature
* Top-K sampling
* Repetition discouragement
* Adjustable generation length
* Diagnostic probability reporting

---

## Training

Training operates on sliding windows over the tokenized corpus.

Important training features include:

* Configurable sequence length
* Configurable stride
* Corpus shuffling
* Forward propagation
* Full transformer backpropagation
* Gradient clipping
* Dropout
* AdamW optimization
* Learning-rate schedules
* Model checkpointing
* Resume training
* Best-model saving

Training statistics include loss and perplexity as well as optional parameter and gradient diagnostics.

---

## AdamW

WesChat uses the **AdamW** optimizer.

Optimizer state includes:

* First moment estimates
* Second moment estimates
* AdamW step count
* Bias correction
* Decoupled weight decay

AdamW state is saved with the model so training can resume without restarting the optimizer history.

---

## CUDA Acceleration

A major goal of WesChat is to keep transformer computation on the GPU.

CUDA functionality includes custom kernels for operations such as:

* Embedding lookup
* LayerNorm forward and backward passes
* RMSNorm forward and backward passes
* Softmax
* Cross-entropy gradients
* Dropout
* Dropout backward pass
* Activation forward and backward passes
* Bias addition and bias gradients
* Rotary positional embeddings
* Gradient clipping
* Embedding-gradient accumulation

---

## cuBLAS

WesChat uses **cuBLAS** for major matrix and vector operations.

Examples include:

* Matrix multiplication
* Transposed matrix multiplication
* Gradient accumulation
* Vector scaling
* Vector copying
* SAXPY-style updates

Pascal wrapper routines provide a relatively direct interface between WesChat and cuBLAS.

---

## GPU Training Design

The general goal is a largely GPU-resident training loop:

```text
Token IDs
   |
   v
Embedding lookup
   |
   v
Transformer blocks
   |
   v
Output head
   |
   v
Softmax / loss
   |
   v
Backpropagation
   |
   v
AdamW parameter update
```

CPU-to-GPU and GPU-to-CPU transfers are minimized where practical.

---

## Model Files

WesChat saves the information required to reload and resume a model, including model parameters and training state.

Depending on model version, saved information may include:

* Transformer parameters
* Embeddings
* Model dimensions
* Vocabulary information
* Normalization kind
* Activation kind
* Activation alpha
* Training progress
* Optimizer state
* Learning-rate state
* Dropout configuration
* Stride
* Random seed state

WES3 model files use a fixed-size metadata header with reserved space for compatible additions. Normalization and activation settings are stored in that header so a loaded model uses the same mathematical architecture with which it was trained.

Older WES3 models that predate these fields are treated as using the earlier defaults. The program also includes compatibility handling for older WesChat model formats.

---

## Work Directories

A WesChat project is organized into a work directory with subdirectories for data such as:

```text
corpus
lists
logs
merges
models
scratch
symbols
tokens
```

This keeps each experiment's corpus, vocabulary, tokenized data, and trained models together.

---

## Example Experimental Configuration

A typical recent configuration is:

| Parameter                 |             Example |
| ------------------------- | ------------------: |
| Transformer blocks        |                   6 |
| Attention heads           |                   8 |
| Model dimension           |                 192 |
| Sequence length           |                 256 |
| MLP projection multiplier |                   4 |
| Normalization             |LayerNorm or RMSNorm |
| MLP activation            |          Selectable |
| Precision                 |             Float32 |
| Optimizer                 |               AdamW |
| Output                    | Weight-tied softmax |
| Positional encoding       |                RoPE |

These values are experimental and are not fixed architectural limits.

---

## Why Pascal?

Part of the purpose of WesChat is to demonstrate that transformer models are not inherently tied to Python.

At their core, the important operations are:

* Arrays
* Matrix multiplication
* Vector operations
* Probability calculations
* Gradient propagation
* Parameter updates

Free Pascal provides direct memory access, strong typing, compiled performance, and straightforward integration with CUDA libraries.

Using Pascal also makes the project an interesting bridge between traditional compiled programming and current machine-learning techniques.

The matrix operations were originally coded by me in Pascal. I then transitioned to the library in libopenblas.dll. When I obtained an NVidia GPU, I moved to NVidia's CUDA library.

---

## Goals

The goals of WesChat are to:

* Understand transformer internals at a low level
* Build an LLM without Python machine-learning frameworks
* Implement forward and backward propagation directly
* Experiment with CUDA and cuBLAS
* Explore tokenization techniques
* Experiment with explicit grammatical information using Universal Dependencies
* Compare different model and training configurations
* Maintain direct control over memory and numerical operations
* Learn by building the complete system

---

## Current Areas of Development

Active areas include:

* UD-aware tokenization and inference
* Implementation of a gated feed-forward network
* Training stability
* CUDA performance
* Model serialization
* Checkpoint compatibility
* Normalization experiments
* Activation-function experiments
* Inference quality
* Sampling methods
* Tokenization experiments
* Larger corpus experiments

---

## Closing

WesChat is a hands-on exploration of how transformer language models work, implemented in a language that predates most modern machine-learning tooling.

Rather than treating the transformer as a black box, WesChat exposes the machinery directly: tokens, embeddings, matrices, attention scores, gradients, optimizer state, CUDA kernels, and generated probabilities.

If you are interested in language models at the level of **matrices, memory, math, and code**, WesChat may prove interesting.ast interesting.
