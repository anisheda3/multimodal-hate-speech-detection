# Multimodal Hate and Abusive Content Detection

A multimodal system for detecting abusive, offensive, and non-abusive content in
text and image-based memes using parameter-efficient fine-tuning and gated
cross-modal fusion.

## Overview

Online hate speech frequently appears in multimodal formats such as memes, where
meaning emerges from the interaction between text and visual context. Text-only
models often fail to capture this interaction.

This project implements a multimodal hate and abusive content detection system
that jointly models textual and visual signals. The system is designed with a
strong emphasis on:
- Robust inference under extreme class imbalance
- Parameter-efficient fine-tuning using LoRA
- Explicit analysis of modality contribution and failure modes

The model performs 3-class classification:
**Abusive / Offensive / Non-abusive**

## System Architecture

The system follows a modular multimodal pipeline:

1. **Text Pipeline**
   - Text normalization and cleaning
   - Transformer-based encoder (MiniLM / RoBERTa variants)
   - LoRA-based parameter-efficient adaptation

2. **Image Pipeline**
   - OCR-based text extraction (EasyOCR)
   - Vision encoder using CLIP ViT
   - Projection into a shared embedding space

3. **Multimodal Fusion**
   - Gated fusion mechanism dynamically weights text and image features
   - Supports modality dominance analysis at inference time

4. **Prediction Heads**
   - 3-class classifier (Abusive / Offensive / Non-abusive)
   - Auxiliary binary head for hierarchical consistency

## Quick Start (Inference Only)

A pretrained checkpoint is provided. No training is required.

### Requirements
- Python 3.10+
- Torch (CPU or GPU)

### Install Dependencies

**Colab (recommended)**

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U transformers peft gradio easyocr pandas numpy pillow

## Quick Start (Local Setup)

### Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate

**Install dependencies**
```bash
pip install -r requirements.txt

**Run demo (CLI)**
```bash
python user_test_fused_lora.py



## Key Results and Insights

- Strong high-precision behavior for abusive content detection
- Effective parameter efficiency:
  - ~98.5% reduction in trainable parameters using LoRA
  - ~2× reduction in training time and memory
- Text modality dominates predictions (~73%) due to explicit linguistic cues
- Visual modality contributes meaningfully in ambiguous meme-based cases

### Known Limitations
- Extreme class imbalance (0.16% Hate class) leads to failure in rare-class recall
- Model tends to over-predict the Offensive class
- Visual features remain underutilized relative to text


## Datasets Used

- Hate Speech Detection Curated
- Hate Speech and Offensive Language Dataset
- Jigsaw Toxic Comment Classification
- Memotion Dataset 7K (multimodal)
- Suspicious Communication on Social Platforms

Due to size and licensing constraints, datasets are not included in this repository.


## Future Work

- Advanced handling of extreme class imbalance (anomaly detection, synthetic generation)
- Cross-attention mechanisms for deeper text-image interaction
- Multilingual extension using XLM-R
- Explainability via attention and prototype-based methods
- Human-in-the-loop moderation workflows

