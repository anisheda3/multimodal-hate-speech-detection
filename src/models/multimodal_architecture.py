# multimodal_architecture.py
# Multimodal hate/abusive content detection: RoBERTa + ResNet50 + fusion + contrastive + multi-task heads
# Paths are aligned to your preprocessing outputs.

import os
import math
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# ------------------------------
# Config
# ------------------------------
@dataclass
class Config:
    # Paths
    base_dir: str = r"C:\Users\G ABHINAV REDDY\Downloads\processed_data"
    processed_dir_name: str = "processed_data"
    text_splits_dirname: str = "splits"

    # Files
    memotion_csv: str = "memotion_7k_multimodal.csv"
    text_train_csv: str = os.path.join("train", "text_train.csv")
    text_val_csv: str = os.path.join("val", "text_val.csv")
    text_test_csv: str = os.path.join("test", "text_test.csv")

    # Model
    text_model_name: str = "roberta-base"  # or "bert-base-uncased"
    max_len: int = 128
    proj_dim: int = 768  # projection dimension for both text and image embeddings
    fusion: str = "concat"  # "concat" or "gated"
    freeze_text: bool = False
    freeze_vision: bool = False
    contrastive_temperature: float = 0.07

    # Optional meta features (Memotion): sarcasm + humour flags
    use_meta_flags: bool = True  # if True, adds 2 dims to fused representation
    meta_dim: int = 2

    # Training
    epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    num_workers: int = 4
    early_stop_patience: int = 3
    grad_clip: float = 1.0

    # Loss weights
    w_hs3: float = 1.0
    w_ab2: float = 1.0
    w_contrastive: float = 0.1  # tune based on multimodal ratio

    # Optional class weights for CE (set to None to disable)
    hs3_class_weights: Optional[List[float]] = None  # e.g., [2.0, 1.0, 0.5] for [Hate, Offensive, Neither]
    ab2_class_weights: Optional[List[float]] = None  # e.g., [1.0, 3.0] for [Non-abusive, Abusive]

    # Image
    image_size: int = 224

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dirname: str = "checkpoints"
    best_model_name: str = "best_model.pt"

    # Logging
    print_every: int = 100


# ------------------------------
# Utils
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    pooled = masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return pooled


def exists(p):
    return p is not None and p != "" and isinstance(p, str)


def to_device_if_tensor(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x


# ------------------------------
# Datasets
# ------------------------------
class MultiModalCSVDataset(Dataset):
    """
    Unified dataset that reads either:
    - Text-only splits (text_train/val/test.csv from processed_data/splits)
    - Memotion multimodal CSV (memotion_7k_multimodal.csv)
    Produces:
      - tokenized text
      - optional image tensor (only for samples that have images)
      - labels:
           hs3_label: {0=Hate, 1=Offensive, 2=Neither, -100=ignore}
           ab2_label: {0/1, -100=ignore}
      - has_image flag
      - optional meta flags: [sarcasm_flag, humour_flag] if use_meta_flags=True
    """
    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer,
        max_len: int,
        image_transform=None,
        is_memotion: bool = False,
        split: Optional[str] = None,  # for memotion random split
        use_meta_flags: bool = True
    ):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_transform = image_transform
        self.is_memotion = is_memotion
        self.use_meta_flags = use_meta_flags

        df = pd.read_csv(csv_path)
        df = df.fillna("")
        self.full_df = df

        # Optional: split memotion into train/val/test if requested
        if is_memotion and split in {"train", "val", "test"}:
            rng = np.random.RandomState(123)  # deterministic
            idx = np.arange(len(df))
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            indices = {
                "train": idx[:n_train],
                "val": idx[n_train:n_train + n_val],
                "test": idx[n_train + n_val:]
            }[split]
            self.df = df.iloc[indices].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        # Column presence flags
        self.has_image_col = "image_path" in self.df.columns
        self.has_off_cat = "offensive_category" in self.df.columns
        self.has_original_class = "original_class" in self.df.columns
        self.has_binary_label = "label" in self.df.columns

        # Memotion meta columns
        self.has_sarcasm = "sarcasm" in self.df.columns
        self.has_humour = "humour" in self.df.columns

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _parse_sarcasm(val: Any) -> int:
        s = str(val).strip().lower()
        if s in {"sarcasm", "sarcastic", "yes", "true", "1"}:
            return 1
        if "sarcas" in s:
            return 1
        return 0

    @staticmethod
    def _parse_humour(val: Any) -> int:
        s = str(val).strip().lower()
        if s in {"", "none", "not_funny", "not funny", "no_humour", "no_humor"}:
            return 0
        if any(k in s for k in ["funny", "hilar", "humor", "humour", "very_funny", "hilarious"]):
            return 1
        return 0

    def _map_memotion_labels(self, offensive_category: str):
        """
        hs3 mapping (3-class):
          hateful_offensive -> 0 (Hate)
          offensive/very_offensive -> 1 (Offensive)
          slight/not_offensive -> 2 (Neither)

        ab2 mapping (binary abusive):
          1 only if offensive/very_offensive/hateful_offensive
          0 for slight/not_offensive
        """
        cat = str(offensive_category).strip().lower()
        # HateSpeech3
        if cat == "hateful_offensive":
            hs3 = 0  # Hate
        elif cat in ("offensive", "very_offensive"):
            hs3 = 1  # Offensive
        elif cat in ("slight", "not_offensive"):
            hs3 = 2  # Neither
        else:
            hs3 = -100  # ignore if unknown

        # Abusive2 (aligned with your preprocessing)
        ab2 = 1 if cat in ("offensive", "very_offensive", "hateful_offensive") else 0
        return hs3, ab2

    def _map_text_labels(self, row: pd.Series):
        # HateSpeech3 from original_class if present (Davidson dataset)
        if self.has_original_class and str(row.get("original_class", "")).strip() != "":
            try:
                oc = int(float(row["original_class"]))
                if oc in (0, 1, 2):
                    hs3 = oc  # 0=Hate, 1=Offensive, 2=Neither
                else:
                    hs3 = -100
            except:
                hs3 = -100
        else:
            hs3 = -100

        # Abusive2 from binary 'label' if present
        if self.has_binary_label:
            try:
                ab2 = int(float(row["label"]))
                ab2 = 1 if ab2 == 1 else 0
            except:
                ab2 = -100
        else:
            ab2 = -100

        return hs3, ab2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row.get("text", ""))

        # Tokenize
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Image (only load if column is present and path exists)
        img_tensor = None
        has_image = 0
        if self.has_image_col:
            img_path = str(row.get("image_path", "")).strip()
            if exists(img_path) and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    if self.image_transform:
                        img_tensor = self.image_transform(img)
                    else:
                        # default resize-normalize if no transform provided
                        tfm = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                        ])
                        img_tensor = tfm(img)
                    has_image = 1
                except:
                    img_tensor = None
                    has_image = 0

        # Labels
        if self.is_memotion and self.has_off_cat:
            hs3, ab2 = self._map_memotion_labels(row.get("offensive_category", ""))
        else:
            hs3, ab2 = self._map_text_labels(row)

        # Meta flags
        if self.use_meta_flags and self.is_memotion:
            sarcasm_flag = self._parse_sarcasm(row.get("sarcasm", "")) if self.has_sarcasm else 0
            humour_flag = self._parse_humour(row.get("humour", "")) if self.has_humour else 0
        else:
            sarcasm_flag = 0
            humour_flag = 0
        meta = torch.tensor([sarcasm_flag, humour_flag], dtype=torch.float32)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img_tensor,  # may be None
            "has_image": has_image,
            "hs3_label": hs3 if hs3 is not None else -100,
            "ab2_label": ab2 if ab2 is not None else -100,
            "meta": meta,  # [2] float tensor
            "text": text,
        }
        return sample


def collate_fn(batch: List[Dict[str, Any]]):
    # Text tensors
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

    # Labels
    hs3_labels = torch.tensor([b["hs3_label"] for b in batch], dtype=torch.long)
    ab2_labels = torch.tensor([b["ab2_label"] for b in batch], dtype=torch.long)

    # Meta flags
    meta = torch.stack([b["meta"] for b in batch], dim=0)  # [B, 2]

    # Images: stack only present ones and track their indices
    has_image_list = [int(b["has_image"]) for b in batch]
    has_image = torch.tensor(has_image_list, dtype=torch.long)
    image_indices = torch.tensor([i for i, b in enumerate(batch) if isinstance(b["image"], torch.Tensor)], dtype=torch.long)
    if image_indices.numel() > 0:
        images_present = torch.stack([batch[i]["image"] for i in image_indices.tolist()], dim=0)
    else:
        images_present = None
        image_indices = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images_present,          # [N_img, 3, H, W] or None
        "image_indices": image_indices,    # [N_img] or None
        "has_image": has_image,            # [B]
        "hs3_labels": hs3_labels,          # [B]
        "ab2_labels": ab2_labels,          # [B]
        "meta": meta,                      # [B, 2]
        "texts": [b["text"] for b in batch],
    }


# ------------------------------
# Model
# ------------------------------
class MultiModalHateAbuseModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(cfg.text_model_name, output_attentions=True)
        text_hidden = self.text_encoder.config.hidden_size
        if cfg.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.text_proj = nn.Linear(text_hidden, cfg.proj_dim)

        # Image encoder (ResNet-50)
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            resnet = models.resnet50(pretrained=True)
        if cfg.freeze_vision:
            for p in resnet.parameters():
                p.requires_grad = False
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove FC, keep GAP
        self.image_proj = nn.Linear(2048, cfg.proj_dim)

        # Fusion
        if cfg.fusion == "concat":
            fused_dim = cfg.proj_dim * 2
            self.gate = None
        elif cfg.fusion == "gated":
            fused_dim = cfg.proj_dim
            self.gate = nn.Sequential(
                nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
                nn.ReLU(),
                nn.Linear(cfg.proj_dim, 2),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError("cfg.fusion must be 'concat' or 'gated'")

        # Add meta dims (sarcasm + humour) if enabled
        head_in_dim = fused_dim + (cfg.meta_dim if cfg.use_meta_flags else 0)

        # Heads
        self.hate3_head = nn.Linear(head_in_dim, 3)  # Hate/Offensive/Neither
        self.abuse2_head = nn.Linear(head_in_dim, 2)  # Abusive/Non-abusive

        # Temperature for contrastive
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / cfg.contrastive_temperature))

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, L, H]
        pooled = mean_pooling(last_hidden, attention_mask)  # [B, H]
        t_emb = self.text_proj(pooled)  # [B, D]
        t_emb = F.normalize(t_emb, dim=-1)
        return t_emb, out.attentions  # list of layer attentions

    def encode_image(self, images):
        # images: [N_img, 3, H, W]
        feats = self.vision_backbone(images)  # [N_img, 2048, 1, 1]
        feats = feats.flatten(1)  # [N_img, 2048]
        v_emb = self.image_proj(feats)  # [N_img, D]
        v_emb = F.normalize(v_emb, dim=-1)
        return v_emb

    def fuse(self, t_emb, v_full):
        # t_emb, v_full: [B, D] (v_full may be None -> zeros)
        if v_full is None:
            v_full = torch.zeros_like(t_emb)

        if self.cfg.fusion == "concat":
            fused = torch.cat([t_emb, v_full], dim=-1)  # [B, 2D]
            return fused, None
        else:
            w = self.gate(torch.cat([t_emb, v_full], dim=-1))  # [B, 2]
            w_t = w[:, 0].unsqueeze(-1)
            w_v = w[:, 1].unsqueeze(-1)
            fused = w_t * t_emb + w_v * v_full
            return fused, w

    def forward(self, input_ids, attention_mask, images=None, image_indices=None, meta=None):
        """
        images: [N_img, 3, H, W] or None
        image_indices: [N_img] positions in the batch that have images
        meta: [B, 2] or None
        """
        B = input_ids.size(0)
        t_emb, attentions = self.encode_text(input_ids, attention_mask)

        v_emb_present = None
        v_full = None
        if images is not None and image_indices is not None and image_indices.numel() > 0:
            v_emb_present = self.encode_image(images)  # [N_img, D]
            # scatter into full batch
            v_full = torch.zeros_like(t_emb)
            v_full[image_indices] = v_emb_present

        fused, gates = self.fuse(t_emb, v_full)

        # Append meta flags if enabled
        if self.cfg.use_meta_flags and meta is not None and meta.size(1) == self.cfg.meta_dim:
            fused = torch.cat([fused, meta], dim=-1)

        hate3_logits = self.hate3_head(fused)
        abuse2_logits = self.abuse2_head(fused)

        return {
            "hate3_logits": hate3_logits,
            "abuse2_logits": abuse2_logits,
            "text_emb": t_emb,                 # [B, D]
            "img_emb_present": v_emb_present,  # [N_img, D] or None
            "image_indices": image_indices,    # [N_img] or None
            "fused": fused,
            "attentions": attentions,
            "gates": gates
        }

    def contrastive_loss(self, text_emb, img_emb_present, image_indices):
        # Align text-image pairs using image_indices order
        if img_emb_present is None or image_indices is None or image_indices.numel() < 2:
            return torch.tensor(0.0, device=text_emb.device)

        t = text_emb[image_indices]  # [N_img, D]
        v = img_emb_present         # [N_img, D]
        if t.size(0) < 2:
            return torch.tensor(0.0, device=text_emb.device)

        # cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_t2v = logit_scale * t @ v.t()  # [N, N]
        logits_v2t = logit_scale * v @ t.t()
        targets = torch.arange(t.shape[0], device=t.device)

        loss_t = F.cross_entropy(logits_t2v, targets)
        loss_v = F.cross_entropy(logits_v2t, targets)
        return 0.5 * (loss_t + loss_v)


# ------------------------------
# Trainer
# ------------------------------
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Paths
        self.processed_dir = os.path.join(cfg.base_dir, cfg.processed_dir_name)
        self.splits_dir = os.path.join(self.processed_dir, cfg.text_splits_dirname)
        self.checkpoint_dir = os.path.join(self.processed_dir, cfg.checkpoint_dirname)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, use_fast=True)

        # Image transforms
        self.train_tfms = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.eval_tfms = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Datasets
        # Text splits
        train_text_csv = os.path.join(self.splits_dir, cfg.text_train_csv)
        val_text_csv = os.path.join(self.splits_dir, cfg.text_val_csv)
        test_text_csv = os.path.join(self.splits_dir, cfg.text_test_csv)

        assert os.path.exists(train_text_csv), f"Missing: {train_text_csv}"
        assert os.path.exists(val_text_csv), f"Missing: {val_text_csv}"
        assert os.path.exists(test_text_csv), f"Missing: {test_text_csv}"

        self.train_text = MultiModalCSVDataset(
            csv_path=train_text_csv,
            tokenizer=self.tokenizer,
            max_len=cfg.max_len,
            image_transform=None,
            is_memotion=False,
            use_meta_flags=cfg.use_meta_flags
        )
        self.val_text = MultiModalCSVDataset(
            csv_path=val_text_csv,
            tokenizer=self.tokenizer,
            max_len=cfg.max_len,
            image_transform=None,
            is_memotion=False,
            use_meta_flags=cfg.use_meta_flags
        )
        self.test_text = MultiModalCSVDataset(
            csv_path=test_text_csv,
            tokenizer=self.tokenizer,
            max_len=cfg.max_len,
            image_transform=None,
            is_memotion=False,
            use_meta_flags=cfg.use_meta_flags
        )

        # Memotion multimodal dataset
        memotion_csv = os.path.join(self.processed_dir, cfg.memotion_csv)
        assert os.path.exists(memotion_csv), f"Missing: {memotion_csv}"

        self.train_memo = MultiModalCSVDataset(
            csv_path=memotion_csv,
            tokenizer=self.tokenizer,
            max_len=cfg.max_len,
            image_transform=self.train_tfms,
            is_memotion=True,
            split="train",
            use_meta_flags=cfg.use_meta_flags
        )
        self.val_memo = MultiModalCSVDataset(
            csv_path=memotion_csv,
            tokenizer=self.tokenizer,
            max_len=cfg.max_len,
            image_transform=self.eval_tfms,
            is_memotion=True,
            split="val",
            use_meta_flags=cfg.use_meta_flags
        )
        self.test_memo = MultiModalCSVDataset(
            csv_path=memotion_csv,
            tokenizer=self.tokenizer,
            max_len=cfg.max_len,
            image_transform=self.eval_tfms,
            is_memotion=True,
            split="test",
            use_meta_flags=cfg.use_meta_flags
        )

        # Combine datasets
        self.train_ds = ConcatDataset([self.train_text, self.train_memo])
        self.val_ds = ConcatDataset([self.val_text, self.val_memo])
        self.test_ds = ConcatDataset([self.test_text, self.test_memo])

        # Loaders
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Model
        self.model = MultiModalHateAbuseModel(cfg).to(self.device)

        # Losses with ignore_index for missing labels
        hs3_weight = torch.tensor(cfg.hs3_class_weights, dtype=torch.float32, device=self.device) if cfg.hs3_class_weights else None
        ab2_weight = torch.tensor(cfg.ab2_class_weights, dtype=torch.float32, device=self.device) if cfg.ab2_class_weights else None

        self.ce_hs3 = nn.CrossEntropyLoss(ignore_index=-100, weight=hs3_weight)
        self.ce_ab2 = nn.CrossEntropyLoss(ignore_index=-100, weight=ab2_weight)

        # Optim + scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = cfg.epochs * len(self.train_loader)
        warmup_steps = int(cfg.warmup_ratio * total_steps) if total_steps > 0 else 0
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Best model tracking
        self.best_val = -1.0
        self.patience = 0
        self.best_state = None
        self.best_model_path = os.path.join(self.checkpoint_dir, cfg.best_model_name)

    def _step_forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        images = to_device_if_tensor(batch["images"], self.device) if batch["images"] is not None else None
        image_indices = to_device_if_tensor(batch["image_indices"], self.device) if batch["image_indices"] is not None else None
        meta = batch["meta"].to(self.device) if batch["meta"] is not None else None
        hs3_labels = batch["hs3_labels"].to(self.device)
        ab2_labels = batch["ab2_labels"].to(self.device)

        out = self.model(input_ids, attention_mask, images, image_indices, meta)

        hs3_loss = self.ce_hs3(out["hate3_logits"], hs3_labels)
        ab2_loss = self.ce_ab2(out["abuse2_logits"], ab2_labels)
        cl_loss = self.model.contrastive_loss(out["text_emb"], out["img_emb_present"], out["image_indices"])

        loss = self.cfg.w_hs3 * hs3_loss + self.cfg.w_ab2 * ab2_loss + self.cfg.w_contrastive * cl_loss
        return loss, hs3_loss, ab2_loss, cl_loss, out

    def train_epoch(self, epoch: int):
        self.model.train()
        running = {"loss": 0.0, "hs3": 0.0, "ab2": 0.0, "cl": 0.0}
        n_steps = 0

        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss, hs3_loss, ab2_loss, cl_loss, _ = self._step_forward(batch)
            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            running["loss"] += loss.item()
            running["hs3"] += hs3_loss.item()
            running["ab2"] += ab2_loss.item()
            running["cl"] += cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0
            n_steps += 1

            if (step + 1) % self.cfg.print_every == 0:
                print(f"Epoch {epoch} | Step {step+1}/{len(self.train_loader)} "
                      f"| Loss {running['loss']/n_steps:.4f} "
                      f"| HS3 {running['hs3']/n_steps:.4f} "
                      f"| AB2 {running['ab2']/n_steps:.4f} "
                      f"| CL {running['cl']/n_steps:.4f}")

        return {k: v / max(n_steps, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self, loader, split_name="val"):
        self.model.eval()
        all_hs3_y, all_hs3_p = [], []
        all_ab2_y, all_ab2_p = [], []

        running = {"loss": 0.0, "hs3": 0.0, "ab2": 0.0, "cl": 0.0}
        n_steps = 0

        for batch in loader:
            loss, hs3_loss, ab2_loss, cl_loss, out = self._step_forward(batch)
            running["loss"] += loss.item()
            running["hs3"] += hs3_loss.item()
            running["ab2"] += ab2_loss.item()
            running["cl"] += cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0
            n_steps += 1

            # Collect metrics for available labels only
            hs3_labels = batch["hs3_labels"].to(self.device)
            ab2_labels = batch["ab2_labels"].to(self.device)
            hs3_mask = hs3_labels != -100
            ab2_mask = ab2_labels != -100

            if hs3_mask.any():
                preds = out["hate3_logits"][hs3_mask].argmax(dim=-1).detach().cpu().numpy()
                gold = hs3_labels[hs3_mask].detach().cpu().numpy()
                all_hs3_p.append(preds)
                all_hs3_y.append(gold)

            if ab2_mask.any():
                preds = out["abuse2_logits"][ab2_mask].argmax(dim=-1).detach().cpu().numpy()
                gold = ab2_labels[ab2_mask].detach().cpu().numpy()
                all_ab2_p.append(preds)
                all_ab2_y.append(gold)

        # Aggregate
        def agg(y_list, p_list):
            if len(y_list) == 0:
                return None
            y = np.concatenate(y_list)
            p = np.concatenate(p_list)
            return {
                "acc": accuracy_score(y, p),
                "f1_macro": f1_score(y, p, average="macro"),
                "precision_macro": precision_score(y, p, average="macro", zero_division=0),
                "recall_macro": recall_score(y, p, average="macro", zero_division=0),
                "y": y, "p": p
            }

        hs3_metrics = agg(all_hs3_y, all_hs3_p)
        ab2_metrics = agg(all_ab2_y, all_ab2_p)

        metrics = {
            "loss": running["loss"] / max(n_steps, 1),
            "hs3": hs3_metrics,
            "ab2": ab2_metrics,
        }

        def print_metrics(name, m):
            if m is None:
                print(f"[{split_name}] {name}: no labels available")
            else:
                print(f"[{split_name}] {name}: "
                      f"acc={m['acc']:.4f} f1_macro={m['f1_macro']:.4f} "
                      f"prec={m['precision_macro']:.4f} rec={m['recall_macro']:.4f}")

        print_metrics("HateSpeech3", hs3_metrics)
        print_metrics("Abusive2", ab2_metrics)

        # Confusion matrices
        if hs3_metrics is not None:
            print(f"[{split_name}] HS3 Confusion Matrix:\n{confusion_matrix(hs3_metrics['y'], hs3_metrics['p'])}")
        if ab2_metrics is not None:
            print(f"[{split_name}] AB2 Confusion Matrix:\n{confusion_matrix(ab2_metrics['y'], ab2_metrics['p'])}")

        return metrics

    def fit(self):
        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\n===== Epoch {epoch}/{self.cfg.epochs} =====")
            tr = self.train_epoch(epoch)
            print(f"Train: loss={tr['loss']:.4f} hs3={tr['hs3']:.4f} ab2={tr['ab2']:.4f} cl={tr['cl']:.4f}")

            val_metrics = self.evaluate(self.val_loader, split_name="val")

            # Early stopping: average F1 across available heads (or negative loss if not available)
            f1s = []
            if val_metrics["hs3"] is not None:
                f1s.append(val_metrics["hs3"]["f1_macro"])
            if val_metrics["ab2"] is not None:
                f1s.append(val_metrics["ab2"]["f1_macro"])
            curr = np.mean(f1s) if f1s else -val_metrics["loss"]

            if curr > self.best_val:
                self.best_val = curr
                self.patience = 0
                self.best_state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                    "cfg": self.cfg.__dict__,
                }
                print("✓ New best model (in-memory).")
                if self.cfg.save_checkpoints:
                    torch.save(self.best_state, self.best_model_path)
                    print(f"✓ Checkpoint saved to: {self.best_model_path}")
            else:
                self.patience += 1
                print(f"No improvement. Patience {self.patience}/{self.cfg.early_stop_patience}")
                if self.patience >= self.cfg.early_stop_patience:
                    print("Early stopping triggered.")
                    break

        # Load best and evaluate on test
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model"])
        elif self.cfg.save_checkpoints and os.path.exists(self.best_model_path):
            state = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(state["model"])

        print("\n===== Final Test Evaluation =====")
        self.evaluate(self.test_loader, split_name="test")


# ------------------------------
# Optional: Interpretability hooks
# ------------------------------
def extract_text_attention(tokens: List[str], attentions):
    """
    Extract average CLS attention across layers/heads for quick visualization.
    tokens: tokenizer.convert_ids_to_tokens(input_ids[0])
    attentions: list of (B, H, L, L)
    Returns: per-token weights (L,)
    """
    # average layers and heads, take CLS row -> distribution over tokens
    att = torch.stack(attentions, dim=0)  # [L, B, H, T, T]
    att = att.mean(dim=(0, 2))  # [B, T, T]
    cls_att = att[0, 0]  # CLS attending to others
    weights = cls_att / (cls_att.sum() + 1e-8)
    return weights.detach().cpu().numpy()


# ------------------------------
# Entrypoint
# ------------------------------
def main():
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()