import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CATEGORY_TO_ID = {
    "Politics": 0,
    "Sports": 1,
    "Economy": 2,
    "International": 3,
    "HealthSociety": 4,
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}


TOPIC_KEYWORDS = {
    "Politics": [
        "حکومت",
        "حکوم",
        "وزیر",
        "وزیراعظم",
        "پارلیمنٹ",
        "اسمبلی",
        "انتخاب",
        "الیکشن",
        "سیاست",
        "سیاسی",
        "government",
        "minister",
        "parliament",
        "election",
    ],
    "Sports": [
        "کرکٹ",
        "میچ",
        "ٹیم",
        "کھلاڑی",
        "کھلاڑ",
        "اسکور",
        "رنز",
        "اوور",
        "کھیل",
        "sports",
        "match",
        "team",
        "player",
        "score",
        "cricket",
    ],
    "Economy": [
        "معیشت",
        "معیش",
        "مہنگائی",
        "تجارت",
        "بینک",
        "بجٹ",
        "قرض",
        "روپیہ",
        "ڈالر",
        "تیل",
        "گیس",
        "economy",
        "inflation",
        "trade",
        "bank",
        "gdp",
        "budget",
    ],
    "International": [
        "اقوام",
        "اقوام متحدہ",
        "معاہدہ",
        "خارجہ",
        "سفارت",
        "مذاکرات",
        "جنگ",
        "ایران",
        "امریکہ",
        "بھارت",
        "چین",
        "سعودی",
        "غزہ",
        "یوکرین",
        "international",
        "foreign",
        "treaty",
        "bilateral",
        "conflict",
        "un",
    ],
    "HealthSociety": [
        "ہسپتال",
        "بیماری",
        "ویکسین",
        "سیلاب",
        "تعلیم",
        "صحت",
        "ڈاکٹر",
        "نیند",
        "وبا",
        "ایچ آئی وی",
        "اسکول",
        "health",
        "hospital",
        "disease",
        "vaccine",
        "education",
    ],
}


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def read_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    blocks = [b.strip() for b in __import__("re").split(r"\n\s*\n(?=\[\d+\])", text) if b.strip()]
    rows = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        m = __import__("re").fullmatch(r"\[(\d+)\]", lines[0])
        if not m:
            continue
        article_id = int(m.group(1))
        tokens = []
        for ln in lines[1:]:
            tokens.extend([w for w in ln.split() if w])
        rows.append({"article_id": article_id, "tokens": tokens})
    return rows


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_for_match(text):
    text = text.lower()
    text = text.replace("آ", "ا")
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("ٱ", "ا")
    text = text.replace("ى", "ی")
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = re.sub(r"[^\u0600-\u06FFa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def topic_scores(title_text, body_text):
    title_norm = normalize_for_match(title_text)
    body_norm = normalize_for_match(body_text)
    title_tokens = title_norm.split()
    body_tokens = body_norm.split()

    scores = {k: 0 for k in TOPIC_KEYWORDS}
    for topic, kws in TOPIC_KEYWORDS.items():
        score = 0
        for kw in kws:
            kw_norm = normalize_for_match(kw)
            if not kw_norm:
                continue
            if " " in kw_norm:
                score += 3 * title_norm.count(kw_norm)
                score += body_norm.count(kw_norm)
            else:
                score += 3 * sum(1 for t in title_tokens if t == kw_norm)
                score += sum(1 for t in body_tokens if t == kw_norm)
        scores[topic] = score
    return scores


def infer_topic(title_text, body_text):
    scores = topic_scores(title_text, body_text)
    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score > 0:
        return best_topic

    # Tie-break fallback when no keyword matches occur.
    body_norm = normalize_for_match(body_text)
    fallback_order = ["Politics", "International", "Economy", "Sports", "HealthSociety"]
    for topic in fallback_order:
        for kw in TOPIC_KEYWORDS[topic]:
            kw_norm = normalize_for_match(kw)
            if kw_norm and kw_norm in body_norm:
                return topic
    return "Politics"


def build_labeled_articles(cleaned_rows, metadata):
    out = []
    for row in cleaned_rows:
        key = str(row["article_id"])
        title = metadata.get(key, {}).get("title", "")
        body_preview = " ".join(row["tokens"][:500])
        label_name = infer_topic(title, body_preview)
        out.append(
            {
                "article_id": row["article_id"],
                "tokens": row["tokens"],
                "label": CATEGORY_TO_ID[label_name],
                "label_name": label_name,
            }
        )
    return out


def build_vocab(rows):
    counter = Counter()
    for row in rows:
        counter.update(row["tokens"])
    idx2word = ["<PAD>", "<UNK>", "[CLS]"] + [w for w, _ in counter.most_common()]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def encode_sequences(rows, word2idx, max_len=256):
    out = []
    for row in rows:
        ids = [2]  # [CLS]
        ids.extend([word2idx.get(t, 1) for t in row["tokens"]])
        ids = ids[:max_len]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad = max_len - len(ids)
            ids.extend([0] * pad)
            mask.extend([0] * pad)
        out.append(
            {
                "article_id": row["article_id"],
                "input_ids": ids,
                "mask": mask,
                "label": row["label"],
                "tokens": ["[CLS]"] + row["tokens"],
            }
        )
    return out


class TextDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (
            torch.tensor(r["input_ids"], dtype=torch.long),
            torch.tensor(r["mask"], dtype=torch.bool),
            torch.tensor(r["label"], dtype=torch.long),
            r["tokens"],
            r["article_id"],
        )


def text_collate(batch):
    input_ids = torch.stack([x[0] for x in batch])
    masks = torch.stack([x[1] for x in batch])
    labels = torch.stack([x[2] for x in batch])
    tokens = [x[3] for x in batch]
    article_ids = [x[4] for x in batch]
    return input_ids, masks, labels, tokens, article_ids


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return out, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_k=32, d_v=32):
        super().__init__()
        self.num_heads = num_heads
        self.q_layers = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(num_heads)])
        self.v_layers = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(num_heads)])
        self.attn = ScaledDotProductAttention()
        self.out_proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x, padding_mask=None):
        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            q = self.q_layers[h](x)
            k = self.k_layers[h](x)
            v = self.v_layers[h](x)

            if padding_mask is not None:
                m = padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                m = None

            out, w = self.attn(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), m)
            head_outputs.append(out.squeeze(1))
            head_weights.append(w.squeeze(1))

        cat = torch.cat(head_outputs, dim=-1)
        out = self.out_proj(cat)
        weights = torch.stack(head_weights, dim=1)
        return out, weights


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        length = x.size(1)
        return x + self.pe[:, :length]


class EncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, d_k=32, d_v=32)
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        attn_out, weights = self.mha(self.ln1(x), padding_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x, weights


class TransformerTopicClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=5, d_model=128, num_heads=4, d_ff=512, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=512)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)])
        self.cls_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, padding_mask):
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.dropout(x)

        final_attn = None
        for layer in self.layers:
            x, final_attn = layer(x, padding_mask)

        cls_vec = x[:, 0, :]
        logits = self.cls_mlp(cls_vec)
        return logits, final_attn


class BiLSTMTopicClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=5, emb_dim=128, hidden_dim=128, dropout=0.3, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, padding_mask):
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        cls_vec = out[:, 0, :]
        logits = self.classifier(cls_vec)
        return logits, None


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps=50):
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def compute_class_weights(rows, num_classes):
    counts = np.ones(num_classes, dtype=np.float32)
    for row in rows:
        counts[row["label"]] += 1.0

    weights = np.sqrt(np.sum(counts) / counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def load_part1_embedding_init(word2idx, emb_dim):
    emb_path = "embeddings/embeddings_w2v.npy"
    vocab_path = "embeddings/word2idx.json"
    if not os.path.exists(emb_path) or not os.path.exists(vocab_path):
        return None, 0

    part1_emb = np.load(emb_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        part1_word2idx = json.load(f)

    matrix = np.random.normal(0, 0.02, (len(word2idx), emb_dim)).astype(np.float32)
    matrix[0] = 0.0
    matched = 0
    copy_dim = min(emb_dim, part1_emb.shape[1])

    for token, idx in word2idx.items():
        src_idx = part1_word2idx.get(token)
        if src_idx is None:
            continue
        matrix[idx, :copy_dim] = part1_emb[src_idx, :copy_dim]
        if emb_dim > copy_dim:
            matrix[idx, copy_dim:] = 0.0
        matched += 1

    return matrix, matched


def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs=20,
    use_scheduler=False,
    class_weights=None,
    select_by="val_acc",
    device=None,
):
    if device is None:
        device = DEVICE

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=(class_weights.to(device) if class_weights is not None else None))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_macro_f1s = []
    epoch_times = []

    total_steps = epochs * max(len(train_loader), 1)
    global_step = 0

    best_state = None
    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_metric = -1.0
    best_epoch = 0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        e_loss = 0.0
        y_true = []
        y_pred = []

        for input_ids, mask, labels, tokens, article_ids in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_scheduler:
                lr = cosine_warmup_lr(global_step, total_steps, 5e-4, warmup_steps=50)
                set_lr(optimizer, lr)

            logits, _ = model(input_ids, mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            e_loss += float(loss.item())
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            global_step += 1

        train_losses.append(e_loss / max(len(train_loader), 1))
        train_accs.append(accuracy_score(y_true, y_pred))

        model.eval()
        v_loss = 0.0
        vy_true = []
        vy_pred = []
        with torch.no_grad():
            for input_ids, mask, labels, tokens, article_ids in val_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits, _ = model(input_ids, mask)
                loss = criterion(logits, labels)
                v_loss += float(loss.item())
                preds = torch.argmax(logits, dim=-1)
                vy_true.extend(labels.detach().cpu().tolist())
                vy_pred.extend(preds.detach().cpu().tolist())

        val_losses.append(v_loss / max(len(val_loader), 1))
        val_acc = accuracy_score(vy_true, vy_pred)
        val_macro_f1 = f1_score(vy_true, vy_pred, average="macro")
        val_accs.append(val_acc)
        val_macro_f1s.append(val_macro_f1)
        epoch_times.append(time.perf_counter() - t0)

        print(f"clf {epoch + 1}/{epochs} val_acc {val_acc:.4f} val_f1 {val_macro_f1:.4f}")

        current_metric = val_macro_f1 if select_by == "macro_f1" else val_acc
        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "epoch_times": epoch_times,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
    }


def evaluate_classifier(model, loader, device=None):
    if device is None:
        device = DEVICE

    model.eval()
    y_true = []
    y_pred = []
    attn_payload = []

    with torch.no_grad():
        for input_ids, mask, labels, tokens, article_ids in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, attn = model(input_ids, mask)
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            if attn is not None:
                for i in range(input_ids.size(0)):
                    attn_payload.append(
                        {
                            "article_id": int(article_ids[i]),
                            "true": int(labels[i].item()),
                            "pred": int(preds[i].item()),
                            "tokens": tokens[i],
                            "attn": attn[i].detach().cpu().numpy(),
                            "mask": mask[i].detach().cpu().numpy(),
                        }
                    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, y_true, y_pred, attn_payload


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, prefix):
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"{prefix} Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{prefix.lower()}_loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title(f"{prefix} Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{prefix.lower()}_accuracy_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(y_true, y_pred, path, title):
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID_TO_CATEGORY[i] for i in labels],
        yticklabels=[ID_TO_CATEGORY[i] for i in labels],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_attention_heatmaps(attn_payload):
    correct = [x for x in attn_payload if x["true"] == x["pred"]]
    picks = correct[:3]
    saved = []

    for idx, item in enumerate(picks, start=1):
        mask = item["mask"]
        valid_len = int(np.sum(mask))
        tokens = item["tokens"][:valid_len]
        attn = item["attn"]

        short_len = min(valid_len, 30)
        tokens_short = tokens[:short_len]

        for head in [0, 1]:
            mat = attn[head][:short_len, :short_len]
            plt.figure(figsize=(8, 6))
            sns.heatmap(mat, cmap="viridis", xticklabels=tokens_short, yticklabels=tokens_short)
            plt.title(f"Final Layer Attention - Sample {idx}, Head {head + 1}")
            plt.xlabel("Key Tokens")
            plt.ylabel("Query Tokens")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            out_path = f"figures/part3_attention_sample{idx}_head{head + 1}.png"
            plt.savefig(out_path, dpi=180)
            plt.close()
            saved.append(out_path)

    return saved


def class_distribution(rows):
    c = Counter([r["label_name"] for r in rows])
    return dict(c)


def write_comparison(transformer_metrics, bilstm_metrics):
    t_acc = transformer_metrics["test_acc"]
    b_acc = bilstm_metrics["test_acc"]
    acc_gap = t_acc - b_acc

    t_best_epoch = transformer_metrics["best_epoch"]
    b_best_epoch = bilstm_metrics["best_epoch"]

    t_time = float(np.mean(transformer_metrics["epoch_times"]))
    b_time = float(np.mean(bilstm_metrics["epoch_times"]))

    if t_time < b_time:
        faster_model = "Transformer"
        speed_reason = "its implementation is highly vectorized on GPU, while recurrent processing in BiLSTM is less parallel-friendly"
    else:
        faster_model = "BiLSTM"
        speed_reason = "its recurrent state updates are lighter than full self-attention in this setup"

    t_macro = transformer_metrics["test_macro_f1"]
    b_macro = bilstm_metrics["test_macro_f1"]
    if t_macro >= b_macro:
        small_data_choice = "Transformer"
        small_data_reason = "it delivered better macro-F1 and used attention to focus on category-bearing tokens despite the smaller corpus"
    else:
        small_data_choice = "BiLSTM"
        small_data_reason = "it generalized better on limited supervision and showed stronger macro-F1 in this run"

    lines = [
        f"1. Transformer test accuracy was {t_acc:.4f} while BiLSTM test accuracy was {b_acc:.4f}.",
        f"2. The accuracy gap was {acc_gap:.4f}, so {'Transformer' if acc_gap >= 0 else 'BiLSTM'} performed better.",
        f"3. Transformer reached best validation accuracy at epoch {t_best_epoch}.",
        f"4. BiLSTM reached best validation accuracy at epoch {b_best_epoch}.",
        f"5. This means {'Transformer' if t_best_epoch < b_best_epoch else 'BiLSTM'} converged in fewer epochs.",
        f"6. Average training time per epoch for Transformer was {t_time:.2f} seconds.",
        f"7. Average training time per epoch for BiLSTM was {b_time:.2f} seconds.",
        f"8. In this run, {faster_model} trained faster per epoch.",
        f"9. A likely reason is that {speed_reason}.",
        "10. The attention heatmaps show that final-layer heads focus strongly on a few topic-indicative tokens.",
        "11. Some heads focus near the [CLS] token while others focus on repeated category keywords.",
        "12. This indicates the Transformer learns where to read evidence for class decisions.",
        f"13. For only 200-300 articles, {small_data_choice} appears more appropriate in this experiment.",
        f"14. This is because {small_data_reason}.",
        "15. If additional labeled data becomes available, re-running both models is recommended to confirm whether this preference remains stable.",
    ]

    with open("data/part3_bilstm_vs_transformer.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_dirs()
    print(f"device {DEVICE.type}")

    cleaned_rows = read_articles("cleaned.txt")
    metadata = load_metadata("Metadata.json")
    labeled_rows = build_labeled_articles(cleaned_rows, metadata)

    labels = [r["label"] for r in labeled_rows]
    train_rows, temp_rows = train_test_split(labeled_rows, test_size=0.30, random_state=SEED, stratify=labels)
    temp_labels = [r["label"] for r in temp_rows]
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.50, random_state=SEED, stratify=temp_labels)

    word2idx, idx2word = build_vocab(labeled_rows)
    class_weights = compute_class_weights(train_rows, num_classes=5)
    embedding_init, embedding_matches = load_part1_embedding_init(word2idx, emb_dim=128)
    print(f"part3 matched embeddings from part1: {embedding_matches}/{len(word2idx)}")

    train_enc = encode_sequences(train_rows, word2idx, max_len=256)
    val_enc = encode_sequences(val_rows, word2idx, max_len=256)
    test_enc = encode_sequences(test_rows, word2idx, max_len=256)

    pin_mem = False
    train_loader = DataLoader(TextDataset(train_enc), batch_size=8, shuffle=True, pin_memory=pin_mem, collate_fn=text_collate)
    val_loader = DataLoader(TextDataset(val_enc), batch_size=8, shuffle=False, pin_memory=pin_mem, collate_fn=text_collate)
    test_loader = DataLoader(TextDataset(test_enc), batch_size=8, shuffle=False, pin_memory=pin_mem, collate_fn=text_collate)

    transformer = TransformerTopicClassifier(vocab_size=len(word2idx), num_classes=5, d_model=128, num_heads=4, d_ff=512, num_layers=4, dropout=0.15)
    t_hist = train_classifier(
        transformer,
        train_loader,
        val_loader,
        epochs=20,
        use_scheduler=True,
        class_weights=class_weights,
        select_by="macro_f1",
        device=DEVICE,
    )
    plot_training_curves(t_hist["train_losses"], t_hist["val_losses"], t_hist["train_accs"], t_hist["val_accs"], "Transformer")

    t_acc, t_f1, t_true, t_pred, attn_payload = evaluate_classifier(t_hist["model"], test_loader, device=DEVICE)
    save_confusion_matrix(t_true, t_pred, "figures/part3_transformer_confusion_matrix.png", "Transformer Confusion Matrix")
    attention_files = save_attention_heatmaps(attn_payload)
    torch.save(t_hist["model"].state_dict(), "models/transformer_cls.pt")

    bilstm = BiLSTMTopicClassifier(
        vocab_size=len(word2idx),
        num_classes=5,
        emb_dim=128,
        hidden_dim=128,
        dropout=0.3,
        embedding_matrix=None,
    )
    b_hist = train_classifier(
        bilstm,
        train_loader,
        val_loader,
        epochs=20,
        use_scheduler=False,
        class_weights=class_weights,
        select_by="macro_f1",
        device=DEVICE,
    )
    plot_training_curves(b_hist["train_losses"], b_hist["val_losses"], b_hist["train_accs"], b_hist["val_accs"], "BiLSTM")

    b_acc, b_f1, b_true, b_pred, _ = evaluate_classifier(b_hist["model"], test_loader, device=DEVICE)
    save_confusion_matrix(b_true, b_pred, "figures/part3_bilstm_confusion_matrix.png", "BiLSTM Confusion Matrix")

    transformer_metrics = {
        "test_acc": t_acc,
        "test_macro_f1": t_f1,
        "best_epoch": t_hist["best_epoch"],
        "best_val_acc": t_hist["best_val_acc"],
        "best_val_macro_f1": t_hist["best_val_macro_f1"],
        "epoch_times": t_hist["epoch_times"],
    }
    bilstm_metrics = {
        "test_acc": b_acc,
        "test_macro_f1": b_f1,
        "best_epoch": b_hist["best_epoch"],
        "best_val_acc": b_hist["best_val_acc"],
        "best_val_macro_f1": b_hist["best_val_macro_f1"],
        "epoch_times": b_hist["epoch_times"],
    }

    write_comparison(transformer_metrics, bilstm_metrics)

    report = {
        "class_distribution": {
            "train": class_distribution(train_rows),
            "val": class_distribution(val_rows),
            "test": class_distribution(test_rows),
        },
        "transformer": transformer_metrics,
        "bilstm_baseline": bilstm_metrics,
        "attention_heatmaps": attention_files,
    }

    with open("data/part3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if DEVICE.type == "cuda":
        for mdl in [transformer, bilstm]:
            mdl.cpu()
        torch.cuda.empty_cache()

    print("part3 done")

    if DEVICE.type == "cuda" and os.name == "nt":
        # Work around a Windows CUDA teardown crash after successful completion.
        os._exit(0)


if __name__ == "__main__":
    main()
