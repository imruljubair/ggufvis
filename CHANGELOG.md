# Changelog

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
