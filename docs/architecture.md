\# System Architecture



This project implements a multimodal hate and abusive content detection system

that combines textual and visual signals for robust inference.



\## High-Level Flow



1\. Text input and/or image input is received

2\. OCR extracts text from images when present

3\. Text is encoded using a transformer model with LoRA adaptation

4\. Images are encoded using a frozen CLIP vision encoder

5\. Text and image embeddings are aligned and fused using a gated mechanism

6\. A 3-class classifier produces final predictions



\## Design Goals



\- Parameter-efficient fine-tuning

\- Transparent modality contribution analysis

\- Robust behavior under extreme class imbalance

\- Inference-focused deployment



