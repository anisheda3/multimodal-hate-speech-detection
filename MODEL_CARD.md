\# Model Card — Multimodal Hate and Abusive Content Detection



\## Model Overview

This model performs multimodal classification of online content into three classes:

\- Abusive

\- Offensive

\- Non-abusive



It jointly processes textual and visual signals, making it suitable for meme-based

and multimodal moderation scenarios.



\## Architecture Summary

\- Text encoder: Transformer-based language model with LoRA adaptation

\- Image encoder: Frozen CLIP ViT vision backbone

\- Fusion: Gated multimodal fusion with dynamic modality weighting

\- Output: 3-class classification with auxiliary binary consistency head



\## Training Data

The model was trained on a combination of publicly available hate speech datasets:

\- Hate Speech Detection Curated

\- Hate Speech and Offensive Language Dataset

\- Jigsaw Toxic Comment Classification

\- Memotion Dataset 7K (multimodal)

\- Suspicious Communication on Social Platforms



Datasets are not included in this repository due to licensing and size constraints.



\## Intended Use

\- First-pass automated content moderation

\- Decision support for human moderators

\- Research and experimentation in multimodal hate speech detection



\## Limitations

\- Extreme class imbalance leads to poor recall for rare hate categories

\- Model tends to over-predict offensive content

\- English-only; not suitable for multilingual or code-switched text

\- Contextual nuances such as sarcasm may be misclassified



\## Ethical Considerations

This model is intended to assist, not replace, human judgment.

Predictions should be reviewed by humans in high-stakes moderation scenarios.

Bias audits and continuous monitoring are recommended before deployment.



