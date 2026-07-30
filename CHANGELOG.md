# Changelog

## 1.4.1

- Shorter block-input conditional in the annotation card

## 1.4.0

- Dedicated block-input navigation step highlighting `X₀ → Xᵦ`
- Block-input annotation explains the first-block and previous-block cases

## 1.3.2

- Layer-aware hidden-state notation: embedding output `X₀`, generic block input
  `Xᵦ`, normalized input `Xᵦ′`, and block output `Xᵦ₊₁`

## 1.3.1

- Conventional attention notation: `S` for raw scores and `A` for the
  causal-softmax attention weights

## 1.3.0

- One persistent, source-aware header above the visualization
- Local filename, Ollama tag, or remote filename selected automatically
- Middle truncation of long source names on narrow terminals
- Model-family text removed from annotation-card titles to avoid repetition

## 1.2.1

- Model-family identity on every annotation-card title
- DeepSeek-R1 Distill cards continue to identify the separate GGUF backbone

## 1.2.0

- DeepSeek-R1 Distill identity detection from GGUF metadata
- Qwen2, Qwen3, or Llama backbone-driven rendering for DeepSeek distills
- Model/backbone labels in block titles and annotation cards
- Explicit rejection of native DeepSeek architectures

## 1.1.0

- In-memory HTTP range reading for public remote/Hugging Face GGUF links
- Automatic conversion of Hugging Face `/blob/` links to `/resolve/`
- Remote byte-transfer and model-size reporting
- Responsive below-diagram annotation cards for narrow terminals
- Narrow-layout footnote suggesting the side-by-side wide-terminal view

## 1.0.0

- Standalone metadata and tensor-descriptor GGUF reader
- Local Ollama manifest and model-blob resolution
- Dense Llama, original Gemma, Qwen2/Qwen2.5, and Qwen3 support
- MHA, GQA, and MQA matrix layouts
- Qwen per-head Q/K normalization visualization
- Tied and untied language-model head visualization
- Ordinal dimension geometry with equal sequence/vocabulary heights
- Lower-border end labels for every panel
- Shape-aware annotation cards with canonical GGUF tensor sources
- Interactive and static terminal modes
