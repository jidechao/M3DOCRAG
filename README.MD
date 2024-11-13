# M3DOCRAG: Multi-modal Multi-page Document RAG System

An implementation of ["M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding"](https://arxiv.org/abs/2411.04952) by Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal (UNC Chapel Hill & Bloomberg).

## Paper Abstract

Document visual question answering (DocVQA) pipelines that answer questions from documents have broad applications. Existing methods focus on handling single-page documents with multi-modal language models (MLMs), or rely on text-based retrieval-augmented generation (RAG) that uses text extraction tools such as optical character recognition (OCR). However, there are difficulties in applying these methods in real-world scenarios: (a) questions often require information across different pages or documents, where MLMs cannot handle many long documents; (b) documents often have important information in visual elements such as figures, but text extraction tools ignore them.

This implementation provides:
- Multi-modal RAG framework for document understanding
- Support for both closed-domain and open-domain settings
- Efficient handling of multi-page documents
- Preservation of visual information in documents

## Features

- 🔍 Multi-modal document retrieval using ColPali
- 🤖 Visual question answering using Qwen2-VL
- 📄 Support for multi-page PDF documents
- 💾 Efficient FAISS indexing for fast retrieval
- 🎯 Optimized for multi-GPU environments
- 💡 Interactive command-line interface

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install torch transformers pdf2image Pillow faiss-cpu tqdm
pip install git+https://github.com/vidore/colpali-engine.git
```

3. Install system dependencies for PDF handling:
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# Download and install poppler from: http://blog.alivate.com.au/poppler-windows/
```

## Models Setup

The system uses the following models:
- Retrieval: `vidore/colpaligemma-3b-mix-448-base` with `vidore/colpali` adapter
- QA: `Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4` (quantized for memory efficiency)

## Usage

1. Start the interactive shell:
```bash
python m3-doc-rag.py
```

2. Initialize the system:
```bash
(M3DOCRAG) init
```

3. Add PDF documents:
```bash
(M3DOCRAG) add path/to/document.pdf
```

4. Build the search index:
```bash
(M3DOCRAG) build
```

5. Ask questions:
```bash
(M3DOCRAG) ask "What is the commercial franchising program?"
```

6. List loaded documents:
```bash
(M3DOCRAG) list
```

7. Exit the system:
```bash
(M3DOCRAG) exit
```

## System Requirements

- CUDA-capable GPU with at least 16GB VRAM (recommended)
- 16GB+ RAM
- Python 3.8+
- Storage space for models and document index

## GPU Memory Configuration

The system is configured to use multiple GPUs efficiently:
- GPU 0: ColPali retrieval model
- GPU 1: Qwen2-VL QA model

Memory optimization features:
- Quantized QA model (GPTQ Int4)
- Batch size optimization
- Aggressive cache clearing
- Memory-efficient attention

## Architecture

1. **Document Processing**
   - PDF to image conversion
   - Page-level processing
   - Multi-modal content handling

2. **Retrieval System**
   - ColPali-based visual-text embeddings
   - FAISS indexing for efficient search
   - Approximate/exact index options

3. **Question Answering**
   - Qwen2-VL visual language model
   - Memory-efficient processing
   - Batch-wise page handling

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cho2024m3docrag,
      title={M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding}, 
      author={Jaemin Cho and Debanjan Mahata and Ozan Irsoy and Yujie He and Mohit Bansal},
      journal={arXiv preprint arXiv:2411.04952},
      year={2024}
}
```

## Code Implementation Credits

This is an unofficial implementation of the M3DOCRAG paper. The original paper was authored by researchers from UNC Chapel Hill and Bloomberg. This implementation uses:
- ColPali for multi-modal retrieval
- Qwen2-VL for visual question answering
- FAISS for efficient similarity search
- pdf2image for PDF processing

## Acknowledgments

- Original Paper Authors:
  - Jaemin Cho (UNC Chapel Hill)
  - Debanjan Mahata (Bloomberg)
  - Ozan Irsoy (Bloomberg)
  - Yujie He (Bloomberg)
  - Mohit Bansal (UNC Chapel Hill)
- Open-source communities behind the various libraries used in this implementation
