# i23-2523 NLP Assignment 2

Repository URL:
https://github.com/ApatheticMioz/i23-2523-NLP-Assignment

## Environment Setup

1. Create and activate virtual environment.
2. Install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce Part 0 (Base Inputs)

This generates `raw.txt`, `cleaned.txt`, `metadata.json`, and `Metadata.json`.

```powershell
.\.venv\Scripts\python.exe scripts\generate_base_corpus.py --target_count 240 --sleep 0.1 --min_body_chars 250
```

## Reproduce Part 1 (Word Embeddings)

This generates:
- `embeddings/tfidf_matrix.npy`
- `embeddings/ppmi_matrix.npy`
- `embeddings/embeddings_w2v.npy`
- `embeddings/word2idx.json`
- Part 1 report/plots

```powershell
.\.venv\Scripts\python.exe scripts\part1_embeddings.py
```

## Reproduce Part 2 (POS + NER)

This generates:
- CoNLL files in `data/`
- `models/bilstm_pos.pt`, `models/bilstm_ner.pt`
- Part 2 plots and report

```powershell
.\.venv\Scripts\python.exe scripts\part2_sequence_labeling.py
```

## Reproduce Part 3 (Transformer Topic Classification)

This generates:
- `models/transformer_cls.pt`
- Part 3 report, confusion matrix, heatmaps, and comparison writeup

```powershell
.\.venv\Scripts\python.exe scripts\part3_transformer_classifier.py
```

## Notebook

Primary notebook:
- `i23-2523_Assignment2_DS-A.ipynb`

The notebook loads generated outputs and displays all required metrics, tables, and figures.

## Notes

- No pretrained models, no Gensim, no HuggingFace.
- Transformer implementation is custom; no `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder` are used.