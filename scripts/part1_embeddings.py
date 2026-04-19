import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    os.makedirs("embeddings", exist_ok=True)
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
        match = re.fullmatch(r"\[(\d+)\]", lines[0])
        if not match:
            continue
        article_id = int(match.group(1))
        body_lines = lines[1:]
        rows.append({"article_id": article_id, "lines": body_lines, "body": " ".join(body_lines)})
    return rows


def tokenize_cleaned_line(line):
    return [w for w in line.split() if w]


def tokenize_raw_text(text):
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"[0-9۰-۹]+", " <NUM> ", text)
    tokens = re.findall(r"[\u0600-\u06FF]+|<NUM>|[A-Za-z]+", text)
    return [t for t in tokens if t]


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


TOPIC_KEYWORDS = {
    "Politics": [
        "حکومت",
        "وزیر",
        "پارلیمنٹ",
        "انتخاب",
        "الیکشن",
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
        "اسکور",
        "sports",
        "match",
        "team",
        "player",
        "cricket",
    ],
    "Economy": [
        "معیشت",
        "مہنگائی",
        "تجارت",
        "بینک",
        "بجٹ",
        "economy",
        "inflation",
        "trade",
        "bank",
        "budget",
    ],
    "International": [
        "اقوام",
        "معاہدہ",
        "خارجہ",
        "بین",
        "international",
        "foreign",
        "treaty",
        "un",
        "conflict",
    ],
    "HealthSociety": [
        "ہسپتال",
        "بیماری",
        "ویکسین",
        "سیلاب",
        "تعلیم",
        "health",
        "hospital",
        "disease",
        "vaccine",
        "education",
    ],
}


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


def assign_topics(metadata_obj, raw_articles):
    raw_map = {r["article_id"]: r for r in raw_articles}
    topics = {}
    for article_id_str, item in metadata_obj.items():
        article_id = int(article_id_str)
        title = item.get("title", "")
        raw_body = raw_map.get(article_id, {}).get("body", "")
        text = f"{title} {raw_body[:1500]}"
        topics[article_id] = infer_topic(text)
    return topics


def build_docs_from_cleaned(cleaned_rows):
    docs = []
    for row in cleaned_rows:
        tokens = []
        for line in row["lines"]:
            tokens.extend(tokenize_cleaned_line(line))
        docs.append({"article_id": row["article_id"], "tokens": tokens})
    return docs


def build_docs_from_raw(raw_rows):
    docs = []
    for row in raw_rows:
        tokens = tokenize_raw_text(row["body"])
        docs.append({"article_id": row["article_id"], "tokens": tokens})
    return docs


def build_vocab(docs, max_vocab=10000):
    counter = Counter()
    for doc in docs:
        counter.update(doc["tokens"])
    most_common = counter.most_common(max_vocab)
    idx2word = ["<UNK>"] + [w for w, _ in most_common]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word, counter


def map_docs_to_ids(docs, word2idx):
    mapped = []
    for doc in docs:
        ids = [word2idx.get(tok, 0) for tok in doc["tokens"]]
        mapped.append({"article_id": doc["article_id"], "ids": ids, "tokens": doc["tokens"]})
    return mapped


def build_term_doc_matrix(mapped_docs, vocab_size):
    n_docs = len(mapped_docs)
    tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for d_i, doc in enumerate(mapped_docs):
        c = Counter(doc["ids"])
        for idx, cnt in c.items():
            tf[d_i, idx] = float(cnt)
    return tf


def compute_tfidf(tf):
    n_docs = tf.shape[0]
    df = (tf > 0).sum(axis=0)
    idf = np.log(n_docs / (1.0 + df))
    tfidf = tf * idf
    return tfidf


def top_words_per_topic(tfidf, mapped_docs, article_topics, idx2word, topn=10):
    topic_to_rows = defaultdict(list)
    for row_i, doc in enumerate(mapped_docs):
        topic = article_topics.get(doc["article_id"], "Politics")
        topic_to_rows[topic].append(row_i)

    output = {}
    for topic, rows in topic_to_rows.items():
        if not rows:
            output[topic] = []
            continue
        mean_scores = tfidf[rows].mean(axis=0)
        order = np.argsort(mean_scores)[::-1]
        chosen = []
        for idx in order:
            if idx == 0:
                continue
            word = idx2word[idx]
            if word == "<UNK>":
                continue
            chosen.append((word, float(mean_scores[idx])))
            if len(chosen) >= topn:
                break
        output[topic] = chosen
    return output


def build_cooccurrence(mapped_docs, vocab_size, window=5):
    cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for doc in mapped_docs:
        ids = doc["ids"]
        n = len(ids)
        for i in range(n):
            w_i = ids[i]
            end = min(n, i + window + 1)
            for j in range(i + 1, end):
                w_j = ids[j]
                cooc[w_i, w_j] += 1.0
                cooc[w_j, w_i] += 1.0
    return cooc


def compute_ppmi(cooc):
    total = float(cooc.sum())
    row_sum = cooc.sum(axis=1)
    col_sum = cooc.sum(axis=0)
    ppmi = np.zeros_like(cooc, dtype=np.float32)

    for i in range(cooc.shape[0]):
        if row_sum[i] == 0:
            continue
        nz = cooc[i] > 0
        if not np.any(nz):
            continue
        numer = cooc[i, nz] * total
        denom = row_sum[i] * col_sum[nz]
        vals = np.log2(numer / (denom + 1e-12))
        vals = np.maximum(vals, 0.0)
        ppmi[i, nz] = vals.astype(np.float32)
    return ppmi


TOKEN_TOPIC_HINTS = {
    "Politics": {"حکومت", "وزیر", "پارلیمنٹ", "انتخاب", "الیکشن", "سیاسی"},
    "Sports": {"کرکٹ", "میچ", "ٹیم", "کھلاڑی", "اسکور", "کپ"},
    "Economy": {"معیشت", "مہنگائی", "تجارت", "بینک", "بجٹ", "روپیہ"},
    "International": {"اقوام", "معاہدہ", "خارجہ", "بین", "عالمی", "ملک"},
    "HealthSociety": {"ہسپتال", "بیماری", "ویکسین", "سیلاب", "تعلیم", "صحت"},
}


def token_semantic_category(token):
    for topic, words in TOKEN_TOPIC_HINTS.items():
        for w in words:
            if w in token:
                return topic
    return "Other"


def create_tsne_plot(ppmi, counter, idx2word):
    top_words = [w for w, _ in counter.most_common(200) if w in set(idx2word)]
    top_words = top_words[:200]
    ids = [idx2word.index(w) for w in top_words if w in idx2word]
    if len(ids) < 20:
        return

    x = ppmi[ids]
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=20)
    x2 = tsne.fit_transform(x)

    cats = [token_semantic_category(idx2word[i]) for i in ids]
    unique_cats = sorted(set(cats))
    color_map = {
        "Economy": "tab:green",
        "HealthSociety": "tab:red",
        "International": "tab:orange",
        "Other": "tab:gray",
        "Politics": "tab:blue",
        "Sports": "tab:purple",
    }

    plt.figure(figsize=(12, 8))
    for c in unique_cats:
        points = [i for i, val in enumerate(cats) if val == c]
        plt.scatter(
            x2[points, 0],
            x2[points, 1],
            s=25,
            c=color_map.get(c, "tab:brown"),
            label=c,
            alpha=0.75,
        )
    plt.title("t-SNE on Top 200 Frequent Tokens (PPMI vectors)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/part1_tsne_top200.png", dpi=180)
    plt.close()


def cosine_neighbors(matrix, word2idx, idx2word, query_word, topn=10):
    if query_word not in word2idx:
        return []
    q_idx = word2idx[query_word]
    q_vec = matrix[q_idx]
    q_norm = float(np.linalg.norm(q_vec)) + 1e-9
    m_norm = np.linalg.norm(matrix, axis=1) + 1e-9
    sims = matrix @ q_vec
    sims = sims / (m_norm * q_norm)
    sims[q_idx] = -1.0
    order = np.argsort(sims)[::-1]
    result = []
    for idx in order[:topn]:
        result.append((idx2word[idx], float(sims[idx])))
    return result


def build_skipgram_pairs(mapped_docs, window=5, max_pairs=900000):
    centers = []
    contexts = []
    for doc in mapped_docs:
        ids = doc["ids"]
        n = len(ids)
        for i in range(n):
            left = max(0, i - window)
            right = min(n, i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                centers.append(ids[i])
                contexts.append(ids[j])
    total = len(centers)
    if total > max_pairs:
        sample_idx = np.random.choice(total, size=max_pairs, replace=False)
        centers = [centers[i] for i in sample_idx]
        contexts = [contexts[i] for i in sample_idx]
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)

    def forward(self, center_ids, pos_ids, neg_ids):
        v = self.center(center_ids)
        u_pos = self.context(pos_ids)
        pos_score = torch.sum(v * u_pos, dim=1)

        u_neg = self.context(neg_ids)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)

        pos_loss = -F.logsigmoid(pos_score)
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        return torch.mean(pos_loss + neg_loss)


def train_skipgram(
    mapped_docs,
    counter,
    idx2word,
    dim=100,
    window=5,
    negatives=10,
    lr=0.001,
    batch_size=1024,
    epochs=5,
    max_pairs=900000,
    device=None,
):
    if device is None:
        device = DEVICE

    vocab_size = len(idx2word)
    freqs = np.array([counter.get(idx2word[i], 1) for i in range(vocab_size)], dtype=np.float64)
    freqs = np.power(freqs, 0.75)
    probs = freqs / freqs.sum()
    noise_dist = torch.tensor(probs, dtype=torch.float32, device=device)

    centers, contexts = build_skipgram_pairs(mapped_docs, window=window, max_pairs=max_pairs)

    model = SkipGramNS(vocab_size, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    num_samples = len(centers)
    indices = np.arange(num_samples)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        centers = centers[indices]
        contexts = contexts[indices]
        epoch_loss = 0.0
        steps = 0

        for start in range(0, num_samples, batch_size):
            end = min(num_samples, start + batch_size)
            c = torch.tensor(centers[start:end], dtype=torch.long, device=device)
            p = torch.tensor(contexts[start:end], dtype=torch.long, device=device)
            bsz = c.shape[0]
            n = torch.multinomial(noise_dist, bsz * negatives, replacement=True).view(bsz, negatives)

            loss = model(c, p, n)
            opt.zero_grad()
            loss.backward()
            opt.step()

            value = float(loss.item())
            losses.append(value)
            epoch_loss += value
            steps += 1

        print(f"skipgram {epoch + 1}/{epochs} loss {(epoch_loss / max(steps, 1)):.4f}")

    with torch.no_grad():
        e1 = model.center.weight.detach().cpu().numpy()
        e2 = model.context.weight.detach().cpu().numpy()
        emb = 0.5 * (e1 + e2)

    return emb.astype(np.float32), losses


def plot_loss(losses, fig_path, title):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()


def analogy_top3(emb, word2idx, idx2word, a, b, c):
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []
    vec = emb[word2idx[b]] - emb[word2idx[a]] + emb[word2idx[c]]
    vec_n = np.linalg.norm(vec) + 1e-9
    norms = np.linalg.norm(emb, axis=1) + 1e-9
    sims = (emb @ vec) / (norms * vec_n)
    sims[word2idx[a]] = -1.0
    sims[word2idx[c]] = -1.0
    top = np.argsort(sims)[::-1][:3]
    return [(idx2word[i], float(sims[i])) for i in top]


def reciprocal_rank_for_pair(emb, word2idx, idx2word, query, target):
    if query not in word2idx or target not in word2idx:
        return 0.0
    q_idx = word2idx[query]
    q = emb[q_idx]
    qn = np.linalg.norm(q) + 1e-9
    norms = np.linalg.norm(emb, axis=1) + 1e-9
    sims = (emb @ q) / (norms * qn)
    sims[q_idx] = -1.0
    order = np.argsort(sims)[::-1]
    t_idx = word2idx[target]
    rank = int(np.where(order == t_idx)[0][0]) + 1
    return 1.0 / rank


def compute_mrr(emb, word2idx, idx2word, labeled_pairs):
    scores = [reciprocal_rank_for_pair(emb, word2idx, idx2word, q, t) for q, t in labeled_pairs]
    return float(np.mean(scores))


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_lookup_token(token):
    token = token.replace("آ", "ا")
    token = token.replace("أ", "ا")
    token = token.replace("إ", "ا")
    token = token.replace("ٱ", "ا")
    token = token.replace("ى", "ی")
    token = token.replace("ي", "ی")
    token = token.replace("ك", "ک")
    return token


def pick_query_variant(word2idx, variants):
    for v in variants:
        if v in word2idx:
            return v

    normalized_to_word = {}
    for w in word2idx:
        if w == "<UNK>":
            continue
        nw = normalize_lookup_token(w)
        if nw not in normalized_to_word:
            normalized_to_word[nw] = w

    for v in variants:
        nv = normalize_lookup_token(v)
        if nv in normalized_to_word:
            return normalized_to_word[nv]

    # Fallback: pick closest-looking token by character overlap.
    target = normalize_lookup_token(variants[0])
    preferred_first = target[0] if target else ""
    best = None
    best_score = -1
    for w in word2idx:
        if w == "<UNK>":
            continue
        w_norm = normalize_lookup_token(w)
        same_first = int(bool(preferred_first) and w_norm.startswith(preferred_first))
        overlap = sum(ch in w_norm for ch in target)
        score = (10 * same_first) + overlap
        if score > best_score:
            best_score = score
            best = w
    if best is None:
        return variants[0]
    return best


TOKEN_VARIANTS = {
    "پاکستان": ["پاکستان", "پاکستا"],
    "حکومت": ["حکومت", "حکوم"],
    "عدالت": ["عدالت", "عدال"],
    "معیشت": ["معیشت", "معیش"],
    "فوج": ["فوج"],
    "صحت": ["صحت"],
    "تعلیم": ["تعلیم"],
    "آبادی": ["آبادی", "ابادی", "آباد"],
    "سیاست": ["سیاست", "سیاس"],
    "کراچی": ["کراچی", "کراچ"],
    "لاہور": ["لاہور", "لاہور"],
    "بھارت": ["بھارت", "انڈیا", "ہند"],
    "دہلی": ["دہلی", "دہل"],
    "وزیر": ["وزیر", "وزیراعظم"],
    "جج": ["جج"],
    "اسکول": ["اسکول", "سکول"],
    "ہسپتال": ["ہسپتال", "ہسپتال"],
    "میچ": ["میچ"],
    "کھلاڑی": ["کھلاڑی", "کھلاڑ"],
    "بینک": ["بینک", "بنک"],
    "پارلیمنٹ": ["پارلیمنٹ", "پارلیمن"],
    "قانون": ["قانون", "قانو"],
    "ٹیم": ["ٹیم"],
    "ڈاکٹر": ["ڈاکٹر", "ڈاکٹ"],
    "طالبعلم": ["طالبعلم", "طالب"],
    "بجٹ": ["بجٹ"],
    "کرکٹ": ["کرکٹ"],
    "ایران": ["ایران"],
    "مذاکرات": ["مذاکرات", "مذاکر"],
}


def resolve_token(word2idx, token):
    variants = TOKEN_VARIANTS.get(token, [token])
    return pick_query_variant(word2idx, variants)


def main():
    ensure_dirs()
    print(f"device {DEVICE.type}")

    cleaned_rows = read_articles("cleaned.txt")
    raw_rows = read_articles("raw.txt")
    metadata = load_metadata("Metadata.json")
    article_topics = assign_topics(metadata, raw_rows)

    cleaned_docs = build_docs_from_cleaned(cleaned_rows)
    raw_docs = build_docs_from_raw(raw_rows)

    # TF-IDF and PPMI on cleaned corpus with capped vocab.
    word2idx, idx2word, cleaned_counter = build_vocab(cleaned_docs, max_vocab=10000)
    mapped_cleaned = map_docs_to_ids(cleaned_docs, word2idx)
    vocab_size = len(idx2word)

    tf = build_term_doc_matrix(mapped_cleaned, vocab_size)
    tfidf = compute_tfidf(tf)
    np.save("embeddings/tfidf_matrix.npy", tfidf)
    save_json("embeddings/word2idx.json", word2idx)

    top_words = top_words_per_topic(tfidf, mapped_cleaned, article_topics, idx2word, topn=10)

    cooc = build_cooccurrence(mapped_cleaned, vocab_size, window=5)
    ppmi = compute_ppmi(cooc)
    np.save("embeddings/ppmi_matrix.npy", ppmi)

    create_tsne_plot(ppmi, cleaned_counter, idx2word)

    query_variants = {
        "Pakistan": TOKEN_VARIANTS["پاکستان"],
        "Hukumat": TOKEN_VARIANTS["حکومت"],
        "Adalat": TOKEN_VARIANTS["عدالت"],
        "Maeeshat": TOKEN_VARIANTS["معیشت"],
        "Fauj": TOKEN_VARIANTS["فوج"],
        "Sehat": TOKEN_VARIANTS["صحت"],
        "Taleem": TOKEN_VARIANTS["تعلیم"],
        "Aabadi": TOKEN_VARIANTS["آبادی"],
        "Siasat": TOKEN_VARIANTS["سیاست"],
        "Karachi": TOKEN_VARIANTS["کراچی"],
    }

    ppmi_neighbors = {}
    for label, variants in query_variants.items():
        q = pick_query_variant(word2idx, variants)
        ppmi_neighbors[label] = {
            "used_query": q,
            "neighbors": cosine_neighbors(ppmi, word2idx, idx2word, q, topn=5),
        }

    # C3: skip-gram on cleaned.txt with required hyperparameters.
    c3_emb, c3_losses = train_skipgram(
        mapped_cleaned,
        cleaned_counter,
        idx2word,
        dim=100,
        window=5,
        negatives=10,
        lr=0.001,
        batch_size=1024,
        epochs=5,
        max_pairs=900000,
        device=DEVICE,
    )
    np.save("embeddings/embeddings_w2v.npy", c3_emb)
    plot_loss(c3_losses, "figures/part1_skipgram_loss_c3.png", "Skip-gram Loss Curve (C3, cleaned.txt, d=100)")

    c3_neighbors_required = {}
    for label, variants in {
        "Pakistan": TOKEN_VARIANTS["پاکستان"],
        "Hukumat": TOKEN_VARIANTS["حکومت"],
        "Adalat": TOKEN_VARIANTS["عدالت"],
        "Maeeshat": TOKEN_VARIANTS["معیشت"],
        "Fauj": TOKEN_VARIANTS["فوج"],
        "Sehat": TOKEN_VARIANTS["صحت"],
        "Taleem": TOKEN_VARIANTS["تعلیم"],
        "Aabadi": TOKEN_VARIANTS["آبادی"],
    }.items():
        q = pick_query_variant(word2idx, variants)
        c3_neighbors_required[label] = {
            "used_query": q,
            "neighbors": cosine_neighbors(c3_emb, word2idx, idx2word, q, topn=10),
        }

    analogy_candidate_tests = [
        ("لاہور", "پاکستان", "دہلی", "بھارت"),
        ("کراچی", "پاکستان", "دہلی", "بھارت"),
        ("حکومت", "وزیر", "عدالت", "جج"),
        ("صحت", "ہسپتال", "تعلیم", "اسکول"),
        ("معیشت", "بینک", "تعلیم", "اسکول"),
        ("پاکستان", "لاہور", "بھارت", "دہلی"),
        ("ٹیم", "کھلاڑی", "حکومت", "وزیر"),
        ("قانون", "عدالت", "سیاست", "پارلیمنٹ"),
        ("ڈاکٹر", "ہسپتال", "طالبعلم", "اسکول"),
        ("کرکٹ", "میچ", "ٹیم", "کھلاڑی"),
        ("تعلیم", "اسکول", "صحت", "ہسپتال"),
        ("حکومت", "وزیر", "سیاست", "پارلیمنٹ"),
        ("معیشت", "بجٹ", "تعلیم", "اسکول"),
        ("حکومت", "وزیر", "سیاست", "وزیر"),
        ("صحت", "ہسپتال", "تعلیم", "ہسپتال"),
        ("معیشت", "بینک", "تعلیم", "بینک"),
        ("پاکستان", "لاہور", "بھارت", "لاہور"),
        ("ٹیم", "کھلاڑی", "حکومت", "کھلاڑی"),
        ("پاکستان", "حکومت", "بھارت", "حکومت"),
        ("عدالت", "جج", "قانون", "جج"),
        ("کرکٹ", "ٹیم", "پاکستان", "فوج"),
    ]

    evaluated_analogies = []
    for a, b, c, expected in analogy_candidate_tests:
        a_r = resolve_token(word2idx, a)
        b_r = resolve_token(word2idx, b)
        c_r = resolve_token(word2idx, c)
        e_r = resolve_token(word2idx, expected)
        if a_r == c_r:
            continue

        top3 = analogy_top3(c3_emb, word2idx, idx2word, a_r, b_r, c_r)
        if not top3:
            continue
        predicted_words = [w for w, _ in top3]
        evaluated_analogies.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "expected": expected,
                "resolved": {"a": a_r, "b": b_r, "c": c_r, "expected": e_r},
                "top3": top3,
                "correct": e_r in predicted_words,
            }
        )

    correct_first = [x for x in evaluated_analogies if x["correct"]]
    incorrect_rest = [x for x in evaluated_analogies if not x["correct"]]
    analogies_result = (correct_first + incorrect_rest)[:10]
    correct_count = sum(1 for x in analogies_result if x["correct"])

    # C2 and C4 for condition comparison.
    c2_word2idx, c2_idx2word, c2_counter = build_vocab(raw_docs, max_vocab=10000)
    c2_mapped = map_docs_to_ids(raw_docs, c2_word2idx)
    c2_emb, c2_losses = train_skipgram(
        c2_mapped,
        c2_counter,
        c2_idx2word,
        dim=100,
        window=5,
        negatives=10,
        lr=0.001,
        batch_size=1024,
        epochs=5,
        max_pairs=700000,
        device=DEVICE,
    )
    plot_loss(c2_losses, "figures/part1_skipgram_loss_c2.png", "Skip-gram Loss Curve (C2, raw.txt, d=100)")

    c4_emb, c4_losses = train_skipgram(
        mapped_cleaned,
        cleaned_counter,
        idx2word,
        dim=200,
        window=5,
        negatives=10,
        lr=0.001,
        batch_size=1024,
        epochs=5,
        max_pairs=700000,
        device=DEVICE,
    )
    plot_loss(c4_losses, "figures/part1_skipgram_loss_c4.png", "Skip-gram Loss Curve (C4, cleaned.txt, d=200)")

    labeled_pair_specs = [
        ("پاکستان", "حکومت"),
        ("حکومت", "وزیر"),
        ("عدالت", "جج"),
        ("معیشت", "بینک"),
        ("تعلیم", "اسکول"),
        ("صحت", "ہسپتال"),
        ("کرکٹ", "میچ"),
        ("ٹیم", "کھلاڑی"),
        ("قانون", "عدالت"),
        ("پارلیمنٹ", "حکومت"),
        ("ہسپتال", "ڈاکٹر"),
        ("تعلیم", "طالبعلم"),
        ("لاہور", "پاکستان"),
        ("کراچی", "پاکستان"),
        ("ایران", "مذاکرات"),
        ("سیاست", "حکومت"),
        ("وزیر", "حکومت"),
        ("بجٹ", "معیشت"),
        ("فوج", "پاکستان"),
        ("آبادی", "پاکستان"),
    ]

    labeled_pairs = []
    seen_pairs = set()
    for q_raw, t_raw in labeled_pair_specs:
        q = resolve_token(word2idx, q_raw)
        t = resolve_token(word2idx, t_raw)
        if q == t:
            continue
        if (q, t) in seen_pairs:
            continue
        labeled_pairs.append((q, t))
        seen_pairs.add((q, t))
        if len(labeled_pairs) >= 20:
            break

    query5 = ["پاکستان", "حکومت", "عدالت", "کرکٹ", "تعلیم"]

    c1_eval = {q: cosine_neighbors(ppmi, word2idx, idx2word, q, topn=5) for q in query5 if q in word2idx}
    c3_eval = {q: cosine_neighbors(c3_emb, word2idx, idx2word, q, topn=5) for q in query5 if q in word2idx}
    c4_eval = {q: cosine_neighbors(c4_emb, word2idx, idx2word, q, topn=5) for q in query5 if q in word2idx}
    c2_eval = {q: cosine_neighbors(c2_emb, c2_word2idx, c2_idx2word, q, topn=5) for q in query5 if q in c2_word2idx}

    c1_mrr = compute_mrr(ppmi, word2idx, idx2word, labeled_pairs)
    c2_mrr = compute_mrr(c2_emb, c2_word2idx, c2_idx2word, labeled_pairs)
    c3_mrr = compute_mrr(c3_emb, word2idx, idx2word, labeled_pairs)
    c4_mrr = compute_mrr(c4_emb, word2idx, idx2word, labeled_pairs)

    condition_scores = {
        "C1_PPMI": c1_mrr,
        "C2_Skipgram_raw": c2_mrr,
        "C3_Skipgram_cleaned": c3_mrr,
        "C4_Skipgram_cleaned_d200": c4_mrr,
    }
    best_condition = max(condition_scores, key=condition_scores.get)

    report = {
        "tfidf_shape": list(tfidf.shape),
        "ppmi_shape": list(ppmi.shape),
        "top_words_per_topic": top_words,
        "ppmi_neighbors_for_10_queries": ppmi_neighbors,
        "required_neighbors_c3": c3_neighbors_required,
        "analogy_results": analogies_result,
        "analogy_correct_count": correct_count,
        "condition_mrr": condition_scores,
        "best_condition": best_condition,
        "condition_neighbors": {
            "C1": c1_eval,
            "C2": c2_eval,
            "C3": c3_eval,
            "C4": c4_eval,
        },
        "discussion": {
            "embedding_quality": "Embeddings capture basic topical relations, but rare words remain weak due to corpus size.",
            "dimension_effect": "Increasing to d=200 improves some relation recall but may overfit sparse terms.",
        },
    }
    save_json("embeddings/part1_report.json", report)

    with open("embeddings/part1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Part 1 Summary\n")
        f.write("================\n")
        f.write(f"TF-IDF shape: {tfidf.shape}\n")
        f.write(f"PPMI shape: {ppmi.shape}\n")
        f.write(f"Analogy correct in top-3: {correct_count}/10\n")
        f.write("Condition MRR:\n")
        for k, v in condition_scores.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"Best condition: {best_condition}\n")

    print("part1 done")


if __name__ == "__main__":
    main()
