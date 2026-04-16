import copy
import json
import os
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "POST", "NUM", "PUNC", "AUX", "UNK"]
NER_TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

PUNC_SET = {"۔", "؟", "!", "،", "؛", ":", ".", "?", ",", "(", ")", "[", "]", "-"}


TOPIC_KEYWORDS = {
    "Politics": ["حکومت", "وزیر", "پارلیمنٹ", "انتخاب", "الیکشن", "government", "minister", "parliament"],
    "Sports": ["کرکٹ", "میچ", "ٹیم", "کھلاڑی", "اسکور", "sports", "match", "team", "cricket"],
    "Economy": ["معیشت", "مہنگائی", "تجارت", "بینک", "بجٹ", "economy", "inflation", "trade", "bank"],
    "International": ["اقوام", "معاہدہ", "خارجہ", "international", "foreign", "treaty", "conflict"],
    "HealthSociety": ["ہسپتال", "بیماری", "ویکسین", "سیلاب", "تعلیم", "health", "hospital", "disease", "education"],
}


PRONOUNS = {
    "میں",
    "ہم",
    "تم",
    "آپ",
    "وہ",
    "یہ",
    "اس",
    "ان",
    "جس",
    "جن",
    "خود",
    "مجھے",
    "تمہیں",
    "ہمیں",
}

DETERMINERS = {"یہ", "وہ", "ایک", "کچھ", "ہر", "تمام", "اس", "ان", "اسی", "ایسے", "وہی", "کسی"}
CONJUNCTIONS = {"اور", "یا", "لیکن", "مگر", "بلکہ", "اگر", "تو", "جب", "چونکہ", "کیونکہ", "ورنہ", "تاہم"}
POSTPOSITIONS = {"میں", "پر", "سے", "کو", "نے", "کا", "کی", "کے", "تک", "بعد", "قبل", "ساتھ", "لیے", "خلاف"}
AUX_WORDS = {"ہے", "ہیں", "تھا", "تھی", "تھے", "ہوں", "گا", "گی", "گے", "ہوگا", "ہوگی", "ہوئے", "کرے"}
ADV_WORDS = {"اب", "پھر", "جلد", "زیادہ", "کم", "بہت", "کبھی", "ہمیشہ", "بعد", "پہلے", "وہاں", "یہاں"}


PER_GAZ = {
    "عمران",
    "نواز",
    "شہباز",
    "بلاول",
    "مریم",
    "آصف",
    "پرویز",
    "حمزہ",
    "فواد",
    "احسن",
    "عاصم",
    "قمر",
    "شاہد",
    "سلمان",
    "عثمان",
    "ندیم",
    "شعیب",
    "بابر",
    "رضوان",
    "حسن",
    "محمد",
    "علی",
    "احمد",
    "فیصل",
    "کامران",
    "ندیم",
    "ارشد",
    "وقار",
    "وسیم",
    "انضمام",
    "مصباح",
    "یونس",
    "سرفراز",
    "شاہین",
    "حارث",
    "ساجد",
    "سعید",
    "عبداللہ",
    "طارق",
    "عدنان",
    "سلیم",
    "جہانگیر",
    "منصور",
    "حیدر",
    "نعمان",
    "شازیہ",
    "عائشہ",
    "فاطمہ",
    "مہرین",
    "مونا",
    "زینب",
    "نادیہ",
    "صبا",
    "حنا",
    "ثنا",
    "کلثوم",
    "رخسانہ",
    "نرگس",
    "صائمہ",
    "روبینہ",
    "مشال",
    "اقرا",
}

LOC_GAZ = {
    "پاکستان",
    "اسلام",
    "آباد",
    "کراچی",
    "لاہور",
    "راولپنڈی",
    "پشاور",
    "کوئٹہ",
    "ملتان",
    "فیصل",
    "آباد",
    "سیالکوٹ",
    "حیدرآباد",
    "گوجرانوالہ",
    "بہاولپور",
    "سکھر",
    "میرپور",
    "گلگت",
    "سکردو",
    "مظفرآباد",
    "کوہاٹ",
    "مردان",
    "چارسدہ",
    "خضدار",
    "تربت",
    "گوادر",
    "کوہاٹ",
    "بنوں",
    "ڈیرا",
    "چمن",
    "نوشہرہ",
    "بھارت",
    "دہلی",
    "ممبئی",
    "کابل",
    "ایران",
    "تہران",
    "ترکی",
    "چین",
    "بیجنگ",
    "امریکہ",
    "لندن",
    "برطانیہ",
    "روس",
    "ماسکو",
    "فرانس",
    "پیرس",
    "جرمنی",
    "برلن",
    "دوحہ",
    "دبئی",
    "ریاض",
    "جدہ",
    "مکہ",
    "مدینہ",
    "نیویارک",
    "واشنگٹن",
    "یوکرین",
    "فلسطین",
    "غزہ",
}

ORG_GAZ = {
    "حکومت",
    "پارلیمنٹ",
    "سینیٹ",
    "اسمبلی",
    "سپریم",
    "عدالت",
    "فوج",
    "نیوی",
    "ائرفورس",
    "پولیس",
    "پی",
    "ٹی",
    "آئی",
    "پیپلز",
    "مسلم",
    "لیگ",
    "ن",
    "پیپلزپارٹی",
    "الیکشن",
    "کمیشن",
    "اسٹیٹ",
    "بینک",
    "ایف",
    "بی",
    "آر",
    "پی",
    "ایس",
    "ایل",
    "آئی",
    "سی",
    "سی",
    "بی",
    "سی",
    "یو",
    "این",
    "ناتو",
    "آئی",
    "ایم",
    "ایف",
    "ورلڈ",
    "بینک",
}

MISC_GAZ = {
    "پیر",
    "منگل",
    "بدھ",
    "جمعرات",
    "جمعہ",
    "ہفتہ",
    "اتوار",
    "جنوری",
    "فروری",
    "مارچ",
    "اپریل",
    "مئی",
    "جون",
    "جولائی",
    "اگست",
    "ستمبر",
    "اکتوبر",
    "نومبر",
    "دسمبر",
}


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def read_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n(?=\[\d+\])", text) if b.strip()]
    rows = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        m = re.fullmatch(r"\[(\d+)\]", lines[0])
        if not m:
            continue
        article_id = int(m.group(1))
        body_lines = lines[1:]
        rows.append({"article_id": article_id, "lines": body_lines, "body": " ".join(body_lines)})
    return rows


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_topic(text):
    text_l = text.lower()
    best = "Politics"
    best_score = -1
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_l:
                score += 1
        if score > best_score:
            best_score = score
            best = topic
    return best


def build_article_topics(metadata, cleaned_rows):
    cleaned_map = {r["article_id"]: r["body"] for r in cleaned_rows}
    out = {}
    for k, v in metadata.items():
        article_id = int(k)
        title = v.get("title", "")
        body = cleaned_map.get(article_id, "")
        out[article_id] = infer_topic(f"{title} {body[:1500]}")
    return out


def sample_500_sentences(cleaned_rows, article_topics):
    all_rows = []
    for row in cleaned_rows:
        topic = article_topics.get(row["article_id"], "Politics")
        for line in row["lines"]:
            tokens = [t for t in line.split() if t]
            if len(tokens) < 3:
                continue
            all_rows.append(
                {
                    "article_id": row["article_id"],
                    "topic": topic,
                    "sentence": line,
                    "tokens": tokens,
                }
            )

    by_topic = defaultdict(list)
    for row in all_rows:
        by_topic[row["topic"]].append(row)

    topic_counts = sorted([(k, len(v)) for k, v in by_topic.items()], key=lambda x: x[1], reverse=True)
    top3 = [k for k, _ in topic_counts[:3]]

    selected = []
    used_ids = set()

    for topic in top3:
        rows = by_topic[topic]
        random.shuffle(rows)
        picked = rows[:100]
        for row in picked:
            key = (row["article_id"], row["sentence"])
            if key not in used_ids:
                selected.append(row)
                used_ids.add(key)

    leftovers = [r for r in all_rows if (r["article_id"], r["sentence"]) not in used_ids]
    random.shuffle(leftovers)
    needed = 500 - len(selected)
    selected.extend(leftovers[:needed])

    random.shuffle(selected)
    return selected[:500], top3


def build_pos_lexicons(sample_rows):
    counter = Counter()
    for row in sample_rows:
        counter.update(row["tokens"])

    verb_lex = {
        "ہے",
        "ہیں",
        "تھا",
        "تھی",
        "کیا",
        "کرتا",
        "کرتی",
        "ہوا",
        "ہوئی",
        "ہوئے",
        "کہا",
        "دیا",
        "لیا",
        "بنایا",
        "چاہتا",
        "چاہتی",
        "لگا",
        "دیکھا",
        "آیا",
        "گیا",
    }

    adj_lex = {
        "بڑا",
        "بڑی",
        "نیا",
        "نئی",
        "اچھا",
        "اچھی",
        "اہم",
        "قومی",
        "عالمی",
        "سیاسی",
        "اقتصادی",
        "طبی",
    }

    noun_lex = set()

    for token, _ in counter.most_common():
        if len(verb_lex) < 220 and (token.endswith("نا") or token.endswith("تا") or token.endswith("تی") or token.endswith("تے")):
            verb_lex.add(token)
        if len(adj_lex) < 220 and (token.endswith("ی") or token.endswith("ا")):
            adj_lex.add(token)

    if len(verb_lex) < 220:
        for token, _ in counter.most_common():
            if token in verb_lex:
                continue
            if token in PRONOUNS or token in DETERMINERS or token in CONJUNCTIONS or token in POSTPOSITIONS:
                continue
            if token in PUNC_SET:
                continue
            verb_lex.add(token)
            if len(verb_lex) >= 220:
                break

    if len(adj_lex) < 220:
        for token, _ in counter.most_common():
            if token in adj_lex:
                continue
            if token in PRONOUNS or token in DETERMINERS or token in CONJUNCTIONS or token in POSTPOSITIONS:
                continue
            if token in PUNC_SET:
                continue
            adj_lex.add(token)
            if len(adj_lex) >= 220:
                break

    for token, _ in counter.most_common():
        if token in verb_lex or token in adj_lex:
            continue
        if token in PRONOUNS or token in DETERMINERS or token in CONJUNCTIONS or token in POSTPOSITIONS:
            continue
        if len(noun_lex) < 220:
            noun_lex.add(token)

    return noun_lex, verb_lex, adj_lex


def pos_tag_token(token, noun_lex, verb_lex, adj_lex):
    if token in PUNC_SET:
        return "PUNC"
    if token == "<NUM>" or re.fullmatch(r"[0-9۰-۹]+", token):
        return "NUM"
    if token in PRONOUNS:
        return "PRON"
    if token in DETERMINERS:
        return "DET"
    if token in CONJUNCTIONS:
        return "CONJ"
    if token in POSTPOSITIONS:
        return "POST"
    if token in ADV_WORDS or token.endswith("طور"):
        return "ADV"
    if token in AUX_WORDS:
        return "AUX"
    if token in verb_lex or token.endswith("نا") or token.endswith("تا") or token.endswith("تی") or token.endswith("تے"):
        return "VERB"
    if token in adj_lex:
        return "ADJ"
    if token in noun_lex or len(token) > 2:
        return "NOUN"
    return "UNK"


def ner_tag_sentence(tokens):
    tags = []
    for t in tokens:
        if t in PER_GAZ:
            tags.append("B-PER")
        elif t in LOC_GAZ:
            tags.append("B-LOC")
        elif t in ORG_GAZ:
            tags.append("B-ORG")
        elif t in MISC_GAZ:
            tags.append("B-MISC")
        else:
            tags.append("O")

    for i in range(1, len(tags)):
        if tags[i].startswith("B-") and tags[i - 1].startswith("B-"):
            prev_type = tags[i - 1][2:]
            this_type = tags[i][2:]
            if prev_type == this_type:
                tags[i] = "I-" + this_type
        if tags[i].startswith("B-") and tags[i - 1].startswith("I-"):
            prev_type = tags[i - 1][2:]
            this_type = tags[i][2:]
            if prev_type == this_type:
                tags[i] = "I-" + this_type
    return tags


def annotate_rows(sample_rows, noun_lex, verb_lex, adj_lex):
    out = []
    for row in sample_rows:
        tokens = row["tokens"]
        pos_tags = [pos_tag_token(t, noun_lex, verb_lex, adj_lex) for t in tokens]
        ner_tags = ner_tag_sentence(tokens)
        out.append(
            {
                "article_id": row["article_id"],
                "topic": row["topic"],
                "sentence": row["sentence"],
                "tokens": tokens,
                "pos_tags": pos_tags,
                "ner_tags": ner_tags,
            }
        )
    return out


def stratified_split(rows):
    topics = [r["topic"] for r in rows]
    train_rows, temp_rows = train_test_split(rows, test_size=0.30, random_state=SEED, stratify=topics)
    temp_topics = [r["topic"] for r in temp_rows]
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.50, random_state=SEED, stratify=temp_topics)
    return train_rows, val_rows, test_rows


def write_conll(path, rows, tag_key):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            for tok, tag in zip(row["tokens"], row[tag_key]):
                f.write(f"{tok} {tag}\n")
            f.write("\n")


def label_distribution(rows, key):
    c = Counter()
    for row in rows:
        c.update(row[key])
    return dict(c)


def build_vocab(rows):
    counter = Counter()
    for row in rows:
        counter.update(row["tokens"])
    idx2word = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common()]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def encode_rows(rows, word2idx, label2idx, label_key):
    out = []
    for row in rows:
        w_ids = [word2idx.get(t, 1) for t in row["tokens"]]
        y_ids = [label2idx[t] for t in row[label_key]]
        out.append(
            {
                "tokens": row["tokens"],
                "labels": y_ids,
                "word_ids": w_ids,
                "sentence": row["sentence"],
                "topic": row["topic"],
            }
        )
    return out


class SeqDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["word_ids"], r["labels"], r["tokens"], r["sentence"]


def collate_batch(batch, pad_label_idx):
    lengths = [len(x[0]) for x in batch]
    max_len = max(lengths)

    w_batch = []
    y_batch = []
    mask_batch = []
    tok_batch = []
    sent_batch = []

    for word_ids, labels, tokens, sentence in batch:
        pad_len = max_len - len(word_ids)
        w_batch.append(word_ids + [0] * pad_len)
        y_batch.append(labels + [pad_label_idx] * pad_len)
        mask_batch.append([1] * len(word_ids) + [0] * pad_len)
        tok_batch.append(tokens)
        sent_batch.append(sentence)

    return (
        torch.tensor(w_batch, dtype=torch.long),
        torch.tensor(y_batch, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(mask_batch, dtype=torch.bool),
        tok_batch,
        sent_batch,
    )


def load_pretrained_embeddings(dataset_word2idx):
    pre_word2idx = json.load(open("embeddings/word2idx.json", "r", encoding="utf-8"))
    pre_emb = np.load("embeddings/embeddings_w2v.npy")
    dim = pre_emb.shape[1]
    mat = np.random.normal(0, 0.05, (len(dataset_word2idx), dim)).astype(np.float32)
    mat[0] = 0.0
    matched = 0
    for word, idx in dataset_word2idx.items():
        if word in pre_word2idx:
            mat[idx] = pre_emb[pre_word2idx[word]]
            matched += 1
    return mat, matched


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, embedding_matrix=None, freeze=False, dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_labels)

    def forward(self, words, lengths):
        x = self.embedding(words)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=words.size(1))
        logits = self.classifier(out)
        return logits


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)

    def forward(self, emissions, tags, mask):
        log_z = self.compute_log_partition(emissions, mask)
        gold = self.score_sentence(emissions, tags, mask)
        return torch.mean(log_z - gold)

    def compute_log_partition(self, emissions, mask):
        batch, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            score_t = score.unsqueeze(2) + trans + emit
            score_t = torch.logsumexp(score_t, dim=1)
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, score_t, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def score_sentence(self, emissions, tags, mask):
        batch, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[torch.arange(batch), 0, tags[:, 0]]

        for t in range(1, seq_len):
            prev_tag = tags[:, t - 1]
            curr_tag = tags[:, t]
            trans_score = self.transitions[prev_tag, curr_tag]
            emit_score = emissions[torch.arange(batch), t, curr_tag]
            m = mask[:, t]
            score = score + (trans_score + emit_score) * m

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch), lengths]
        score = score + self.end_transitions[last_tags]
        return score

    def decode(self, emissions, mask):
        batch, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for t in range(1, seq_len):
            score_t = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_tag = torch.max(score_t, dim=1)
            best_score = best_score + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_tag)

        score = score + self.end_transitions
        best_last_score, best_last_tag = torch.max(score, dim=1)

        paths = []
        for b in range(batch):
            length = int(mask[b].long().sum().item())
            tag = int(best_last_tag[b].item())
            path = [tag]
            for step in reversed(history[: length - 1]):
                tag = int(step[b, tag].item())
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths


class NERTagger(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_labels,
        embedding_matrix=None,
        freeze=False,
        dropout=0.5,
        bidirectional=True,
        use_crf=True,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.encoder = BiLSTMTagger(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            embedding_matrix=embedding_matrix,
            freeze=freeze,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        if use_crf:
            self.crf = CRF(num_labels)

    def forward(self, words, lengths):
        return self.encoder(words, lengths)


def plot_curves(train_losses, val_losses, title, path):
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def evaluate_pos(model, loader, idx2label, pad_label_idx, device=None):
    if device is None:
        device = DEVICE

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for words, labels, lengths, mask, tok_batch, sent_batch in loader:
            words = words.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(words, lengths)
            preds = torch.argmax(logits, dim=-1)

            for i in range(words.size(0)):
                length = int(lengths[i].item())
                true_seq = labels[i, :length].detach().cpu().tolist()
                pred_seq = preds[i, :length].detach().cpu().tolist()
                for t, p in zip(true_seq, pred_seq):
                    if t == pad_label_idx:
                        continue
                    y_true.append(idx2label[t])
                    y_pred.append(idx2label[p])

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, y_true, y_pred


def train_pos_model(
    train_loader,
    val_loader,
    vocab_size,
    emb_dim,
    hidden_dim,
    num_labels,
    idx2label,
    pad_label_idx,
    embedding_matrix,
    freeze,
    dropout=0.5,
    bidirectional=True,
    max_epochs=20,
    device=None,
):
    if device is None:
        device = DEVICE

    model = BiLSTMTagger(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        embedding_matrix=embedding_matrix,
        freeze=freeze,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_val_f1 = -1.0
    patience = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for words, labels, lengths, mask, tok_batch, sent_batch in train_loader:
            words = words.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(words, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        train_losses.append(epoch_loss / max(steps, 1))

        model.eval()
        val_epoch_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for words, labels, lengths, mask, tok_batch, sent_batch in val_loader:
                words = words.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                logits = model(words, lengths)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_epoch_loss += float(loss.item())
                val_steps += 1
        val_losses.append(val_epoch_loss / max(val_steps, 1))

        _, val_f1, _, _ = evaluate_pos(model, val_loader, idx2label, pad_label_idx, device=device)
        print(f"pos {epoch + 1}/{max_epochs} f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_val_f1


def evaluate_ner(model, loader, idx2label, pad_label_idx, use_crf, device=None):
    if device is None:
        device = DEVICE

    model.eval()
    true_seq_all = []
    pred_seq_all = []
    token_records = []

    with torch.no_grad():
        for words, labels, lengths, mask, tok_batch, sent_batch in loader:
            words = words.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            emissions = model(words, lengths)
            if use_crf:
                pred_paths = model.crf.decode(emissions, mask)

                all_o = True
                for path in pred_paths:
                    for p in path:
                        if p != 0:
                            all_o = False
                            break
                    if not all_o:
                        break

                if all_o:
                    pred_ids = torch.argmax(emissions, dim=-1)
                    pred_paths = []
                    for i in range(words.size(0)):
                        pred_paths.append(pred_ids[i, : int(lengths[i].item())].tolist())
            else:
                pred_paths = []
                pred_ids = torch.argmax(emissions, dim=-1)
                for i in range(words.size(0)):
                    pred_paths.append(pred_ids[i, : int(lengths[i].item())].tolist())

            for i in range(words.size(0)):
                length = int(lengths[i].item())
                true_ids = labels[i, :length].detach().cpu().tolist()
                pred_ids = pred_paths[i]
                true_labels = [idx2label[t] for t in true_ids if t != pad_label_idx]
                pred_labels = [idx2label[p] for p in pred_ids[: len(true_labels)]]
                true_seq_all.append(true_labels)
                pred_seq_all.append(pred_labels)

                token_records.append(
                    {
                        "tokens": tok_batch[i],
                        "sentence": sent_batch[i],
                        "true": true_labels,
                        "pred": pred_labels,
                    }
                )

    report = seqeval_report(true_seq_all, pred_seq_all, output_dict=True, zero_division=0)
    overall_f1 = seqeval_f1(true_seq_all, pred_seq_all)
    return report, overall_f1, token_records


def train_ner_model(
    train_loader,
    val_loader,
    vocab_size,
    emb_dim,
    hidden_dim,
    num_labels,
    idx2label,
    pad_label_idx,
    embedding_matrix,
    freeze,
    use_crf=True,
    dropout=0.5,
    bidirectional=True,
    max_epochs=20,
    device=None,
):
    if device is None:
        device = DEVICE

    model = NERTagger(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        embedding_matrix=embedding_matrix,
        freeze=freeze,
        dropout=dropout,
        bidirectional=bidirectional,
        use_crf=use_crf,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_val_f1 = -1.0
    patience = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for words, labels, lengths, mask, tok_batch, sent_batch in train_loader:
            words = words.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            emissions = model(words, lengths)
            if use_crf:
                labels_for_crf = labels.clone()
                labels_for_crf[labels_for_crf == pad_label_idx] = 0
                crf_loss = model.crf(emissions, labels_for_crf, mask)
                ce_loss = criterion(emissions.view(-1, emissions.size(-1)), labels.view(-1))
                loss = crf_loss + 0.5 * ce_loss
            else:
                loss = criterion(emissions.view(-1, emissions.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        train_losses.append(epoch_loss / max(steps, 1))

        model.eval()
        val_epoch_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for words, labels, lengths, mask, tok_batch, sent_batch in val_loader:
                words = words.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                emissions = model(words, lengths)
                if use_crf:
                    labels_for_crf = labels.clone()
                    labels_for_crf[labels_for_crf == pad_label_idx] = 0
                    crf_loss = model.crf(emissions, labels_for_crf, mask)
                    ce_loss = criterion(emissions.view(-1, emissions.size(-1)), labels.view(-1))
                    loss = crf_loss + 0.5 * ce_loss
                else:
                    loss = criterion(emissions.view(-1, emissions.size(-1)), labels.view(-1))

                val_epoch_loss += float(loss.item())
                val_steps += 1
        val_losses.append(val_epoch_loss / max(val_steps, 1))

        _, val_f1, _ = evaluate_ner(model, val_loader, idx2label, pad_label_idx, use_crf, device=device)
        print(f"ner {epoch + 1}/{max_epochs} f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_val_f1


def extract_pos_confusions(y_true, y_pred):
    tags = POS_TAGS
    cm = confusion_matrix(y_true, y_pred, labels=tags)
    pair_counts = []
    for i, t1 in enumerate(tags):
        for j, t2 in enumerate(tags):
            if i == j:
                continue
            if cm[i, j] > 0:
                pair_counts.append((int(cm[i, j]), t1, t2))
    pair_counts.sort(reverse=True)
    return cm, pair_counts[:3]


def collect_confusion_examples(test_rows, pos_model, word2idx, pos_label2idx, pos_idx2label, device=None):
    if device is None:
        device = DEVICE

    examples = defaultdict(list)
    ds = SeqDataset(
        encode_rows(test_rows, word2idx, pos_label2idx, "pos_tags")
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch(b, pos_label2idx["UNK"]))

    pos_model.eval()
    with torch.no_grad():
        for words, labels, lengths, mask, tok_batch, sent_batch in loader:
            words = words.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = pos_model(words, lengths)
            preds = torch.argmax(logits, dim=-1)
            for i in range(words.size(0)):
                length = int(lengths[i].item())
                for j in range(length):
                    t = pos_idx2label[int(labels[i, j].item())]
                    p = pos_idx2label[int(preds[i, j].item())]
                    if t != p and len(examples[(t, p)]) < 4:
                        examples[(t, p)].append(sent_batch[i])
    return examples


def ner_error_analysis(token_records):
    fps = []
    fns = []
    for item in token_records:
        for tok, t, p in zip(item["tokens"], item["true"], item["pred"]):
            if p != "O" and t == "O" and len(fps) < 5:
                fps.append({"token": tok, "true": t, "pred": p, "sentence": item["sentence"]})
            if t != "O" and p == "O" and len(fns) < 5:
                fns.append({"token": tok, "true": t, "pred": p, "sentence": item["sentence"]})
            if len(fps) >= 5 and len(fns) >= 5:
                return fps, fns
    return fps, fns


def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    ensure_dirs()
    print(f"device {DEVICE.type}")

    cleaned_rows = read_articles("cleaned.txt")
    metadata = load_metadata("Metadata.json")
    article_topics = build_article_topics(metadata, cleaned_rows)

    sample_rows, top3_topics = sample_500_sentences(cleaned_rows, article_topics)
    noun_lex, verb_lex, adj_lex = build_pos_lexicons(sample_rows)
    annotated = annotate_rows(sample_rows, noun_lex, verb_lex, adj_lex)

    train_rows, val_rows, test_rows = stratified_split(annotated)

    write_conll("data/pos_train.conll", train_rows, "pos_tags")
    write_conll("data/pos_val.conll", val_rows, "pos_tags")
    write_conll("data/pos_test.conll", test_rows, "pos_tags")
    write_conll("data/ner_train.conll", train_rows, "ner_tags")
    write_conll("data/ner_val.conll", val_rows, "ner_tags")
    write_conll("data/ner_test.conll", test_rows, "ner_tags")

    # Build vocab and embeddings.
    word2idx, idx2word = build_vocab(train_rows)
    emb_matrix, matched_count = load_pretrained_embeddings(word2idx)
    emb_dim = emb_matrix.shape[1]

    pos_label2idx = {t: i for i, t in enumerate(POS_TAGS)}
    pos_idx2label = {i: t for t, i in pos_label2idx.items()}
    pos_pad_idx = pos_label2idx["UNK"]

    ner_label2idx = {t: i for i, t in enumerate(NER_TAGS)}
    ner_idx2label = {i: t for t, i in ner_label2idx.items()}
    ner_pad_idx = ner_label2idx["O"]

    pos_train_enc = encode_rows(train_rows, word2idx, pos_label2idx, "pos_tags")
    pos_val_enc = encode_rows(val_rows, word2idx, pos_label2idx, "pos_tags")
    pos_test_enc = encode_rows(test_rows, word2idx, pos_label2idx, "pos_tags")

    ner_train_enc = encode_rows(train_rows, word2idx, ner_label2idx, "ner_tags")
    ner_val_enc = encode_rows(val_rows, word2idx, ner_label2idx, "ner_tags")
    ner_test_enc = encode_rows(test_rows, word2idx, ner_label2idx, "ner_tags")

    pin_mem = False
    pos_train_loader = DataLoader(SeqDataset(pos_train_enc), batch_size=32, shuffle=True, pin_memory=pin_mem, collate_fn=lambda b: collate_batch(b, pos_pad_idx))
    pos_val_loader = DataLoader(SeqDataset(pos_val_enc), batch_size=32, shuffle=False, pin_memory=pin_mem, collate_fn=lambda b: collate_batch(b, pos_pad_idx))
    pos_test_loader = DataLoader(SeqDataset(pos_test_enc), batch_size=32, shuffle=False, pin_memory=pin_mem, collate_fn=lambda b: collate_batch(b, pos_pad_idx))

    ner_train_loader = DataLoader(SeqDataset(ner_train_enc), batch_size=32, shuffle=True, pin_memory=pin_mem, collate_fn=lambda b: collate_batch(b, ner_pad_idx))
    ner_val_loader = DataLoader(SeqDataset(ner_val_enc), batch_size=32, shuffle=False, pin_memory=pin_mem, collate_fn=lambda b: collate_batch(b, ner_pad_idx))
    ner_test_loader = DataLoader(SeqDataset(ner_test_enc), batch_size=32, shuffle=False, pin_memory=pin_mem, collate_fn=lambda b: collate_batch(b, ner_pad_idx))

    # POS frozen
    pos_frozen_model, pos_frozen_train_losses, pos_frozen_val_losses, pos_frozen_val_f1 = train_pos_model(
        pos_train_loader,
        pos_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(POS_TAGS),
        idx2label=pos_idx2label,
        pad_label_idx=pos_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=True,
        dropout=0.5,
        bidirectional=True,
        max_epochs=20,
        device=DEVICE,
    )
    plot_curves(pos_frozen_train_losses, pos_frozen_val_losses, "POS Loss (Frozen Embeddings)", "figures/part2_pos_loss_frozen.png")

    # POS fine-tuned
    pos_ft_model, pos_ft_train_losses, pos_ft_val_losses, pos_ft_val_f1 = train_pos_model(
        pos_train_loader,
        pos_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(POS_TAGS),
        idx2label=pos_idx2label,
        pad_label_idx=pos_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=False,
        dropout=0.5,
        bidirectional=True,
        max_epochs=20,
        device=DEVICE,
    )
    plot_curves(pos_ft_train_losses, pos_ft_val_losses, "POS Loss (Fine-tuned Embeddings)", "figures/part2_pos_loss_finetuned.png")

    pos_acc, pos_macro_f1, pos_true, pos_pred = evaluate_pos(pos_ft_model, pos_test_loader, pos_idx2label, pos_pad_idx, device=DEVICE)

    cm, top_confused = extract_pos_confusions(pos_true, pos_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=POS_TAGS, yticklabels=POS_TAGS)
    plt.title("POS Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("figures/part2_pos_confusion_matrix.png", dpi=180)
    plt.close()

    confusion_examples = collect_confusion_examples(test_rows, pos_ft_model, word2idx, pos_label2idx, pos_idx2label, device=DEVICE)

    torch.save(pos_ft_model.state_dict(), "models/bilstm_pos.pt")

    # NER with CRF (frozen and fine-tuned)
    ner_crf_frozen_model, ner_crf_frozen_train_losses, ner_crf_frozen_val_losses, ner_crf_frozen_val_f1 = train_ner_model(
        ner_train_loader,
        ner_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(NER_TAGS),
        idx2label=ner_idx2label,
        pad_label_idx=ner_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=True,
        use_crf=True,
        dropout=0.5,
        bidirectional=True,
        max_epochs=20,
        device=DEVICE,
    )
    plot_curves(ner_crf_frozen_train_losses, ner_crf_frozen_val_losses, "NER Loss (CRF, Frozen)", "figures/part2_ner_loss_crf_frozen.png")

    ner_crf_ft_model, ner_crf_ft_train_losses, ner_crf_ft_val_losses, ner_crf_ft_val_f1 = train_ner_model(
        ner_train_loader,
        ner_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(NER_TAGS),
        idx2label=ner_idx2label,
        pad_label_idx=ner_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=False,
        use_crf=True,
        dropout=0.5,
        bidirectional=True,
        max_epochs=20,
        device=DEVICE,
    )
    plot_curves(ner_crf_ft_train_losses, ner_crf_ft_val_losses, "NER Loss (CRF, Fine-tuned)", "figures/part2_ner_loss_crf_finetuned.png")

    ner_crf_report, ner_crf_test_f1, ner_crf_records = evaluate_ner(
        ner_crf_ft_model,
        ner_test_loader,
        ner_idx2label,
        ner_pad_idx,
        use_crf=True,
        device=DEVICE,
    )

    # NER without CRF
    ner_softmax_model, ner_softmax_train_losses, ner_softmax_val_losses, ner_softmax_val_f1 = train_ner_model(
        ner_train_loader,
        ner_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(NER_TAGS),
        idx2label=ner_idx2label,
        pad_label_idx=ner_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=False,
        use_crf=False,
        dropout=0.5,
        bidirectional=True,
        max_epochs=20,
        device=DEVICE,
    )
    plot_curves(ner_softmax_train_losses, ner_softmax_val_losses, "NER Loss (Softmax)", "figures/part2_ner_loss_softmax.png")

    ner_softmax_report, ner_softmax_test_f1, _ = evaluate_ner(
        ner_softmax_model,
        ner_test_loader,
        ner_idx2label,
        ner_pad_idx,
        use_crf=False,
        device=DEVICE,
    )

    torch.save(ner_crf_ft_model.state_dict(), "models/bilstm_ner.pt")

    fps, fns = ner_error_analysis(ner_crf_records)

    # Ablations A1-A4 on NER
    ablation_results = {}

    # A1: unidirectional LSTM
    a1_model, _, _, _ = train_ner_model(
        ner_train_loader,
        ner_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(NER_TAGS),
        idx2label=ner_idx2label,
        pad_label_idx=ner_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=False,
        use_crf=True,
        dropout=0.5,
        bidirectional=False,
        max_epochs=12,
        device=DEVICE,
    )
    _, a1_f1, _ = evaluate_ner(a1_model, ner_test_loader, ner_idx2label, ner_pad_idx, use_crf=True, device=DEVICE)
    ablation_results["A1_unidirectional_lstm"] = a1_f1

    # A2: no dropout
    a2_model, _, _, _ = train_ner_model(
        ner_train_loader,
        ner_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(NER_TAGS),
        idx2label=ner_idx2label,
        pad_label_idx=ner_pad_idx,
        embedding_matrix=emb_matrix,
        freeze=False,
        use_crf=True,
        dropout=0.0,
        bidirectional=True,
        max_epochs=12,
        device=DEVICE,
    )
    _, a2_f1, _ = evaluate_ner(a2_model, ner_test_loader, ner_idx2label, ner_pad_idx, use_crf=True, device=DEVICE)
    ablation_results["A2_no_dropout"] = a2_f1

    # A3: random embedding initialization
    a3_model, _, _, _ = train_ner_model(
        ner_train_loader,
        ner_val_loader,
        vocab_size=len(word2idx),
        emb_dim=emb_dim,
        hidden_dim=128,
        num_labels=len(NER_TAGS),
        idx2label=ner_idx2label,
        pad_label_idx=ner_pad_idx,
        embedding_matrix=None,
        freeze=False,
        use_crf=True,
        dropout=0.5,
        bidirectional=True,
        max_epochs=12,
        device=DEVICE,
    )
    _, a3_f1, _ = evaluate_ner(a3_model, ner_test_loader, ner_idx2label, ner_pad_idx, use_crf=True, device=DEVICE)
    ablation_results["A3_random_embeddings"] = a3_f1

    # A4: softmax output instead of CRF
    ablation_results["A4_softmax_instead_of_crf"] = ner_softmax_test_f1

    pos_dist = {
        "train": label_distribution(train_rows, "pos_tags"),
        "val": label_distribution(val_rows, "pos_tags"),
        "test": label_distribution(test_rows, "pos_tags"),
    }
    ner_dist = {
        "train": label_distribution(train_rows, "ner_tags"),
        "val": label_distribution(val_rows, "ner_tags"),
        "test": label_distribution(test_rows, "ner_tags"),
    }

    confused_pairs_payload = []
    for count, true_tag, pred_tag in top_confused:
        ex = confusion_examples.get((true_tag, pred_tag), [])[:2]
        confused_pairs_payload.append(
            {
                "pair": f"{true_tag}->{pred_tag}",
                "count": count,
                "examples": ex,
            }
        )

    report = {
        "selected_topics_for_sampling": top3_topics,
        "sentence_count": len(sample_rows),
        "lexicon_sizes": {
            "noun": len(noun_lex),
            "verb": len(verb_lex),
            "adj": len(adj_lex),
            "per_gazetteer": len(PER_GAZ),
            "loc_gazetteer": len(LOC_GAZ),
            "org_gazetteer": len(ORG_GAZ),
        },
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "label_distribution": {
            "pos": pos_dist,
            "ner": ner_dist,
        },
        "embedding_init": {
            "matched_tokens_with_part1_embeddings": matched_count,
            "vocab_size": len(word2idx),
        },
        "pos_results": {
            "frozen_val_f1": pos_frozen_val_f1,
            "finetuned_val_f1": pos_ft_val_f1,
            "test_accuracy": pos_acc,
            "test_macro_f1": pos_macro_f1,
            "top_confused_pairs": confused_pairs_payload,
        },
        "ner_results": {
            "crf_frozen_val_f1": ner_crf_frozen_val_f1,
            "crf_finetuned_val_f1": ner_crf_ft_val_f1,
            "test_overall_f1_crf": ner_crf_test_f1,
            "test_overall_f1_softmax": ner_softmax_test_f1,
            "entity_report_crf": ner_crf_report,
            "entity_report_softmax": ner_softmax_report,
            "false_positives": fps,
            "false_negatives": fns,
        },
        "ablations": ablation_results,
    }

    with open("data/part2_report.json", "w", encoding="utf-8") as f:
        json.dump(to_builtin(report), f, ensure_ascii=False, indent=2)

    if DEVICE.type == "cuda":
        for mdl in [
            pos_frozen_model,
            pos_ft_model,
            ner_crf_frozen_model,
            ner_crf_ft_model,
            ner_softmax_model,
            a1_model,
            a2_model,
            a3_model,
        ]:
            mdl.cpu()
        torch.cuda.empty_cache()

    print("part2 done")

    if DEVICE.type == "cuda" and os.name == "nt":
        # Work around a Windows CUDA teardown crash after successful completion.
        os._exit(0)


if __name__ == "__main__":
    main()
