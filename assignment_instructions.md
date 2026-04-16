```
FAST NATIONAL UNIVERSITY OF COMPUTER & EMERGING SCIENCES
```
# CS-4063: Natural Language Processing

### Assignment 2 — Neural NLP Pipeline

```
A Continuation of the BBC Urdu NLP Pipeline
```
```
Total Marks: 75 Framework: PyTorch (from scratch)
Due Date: 20 April 2026, 11:59 PM Prerequisite: Assignment 1 complete
```
```
Restrictions — Read Before You Start
No pretrained models, no Gensim, no HuggingFace. Everything must be implemented
from scratch in PyTorch. The following built-ins are not allowed: nn.Transformer,
nn.MultiheadAttention, nn.TransformerEncoder.
```
Required Input Files:

File Used In Purpose

cleaned.txt All parts Primary training corpus
raw.txt Parts 1 & 2 Ablation baseline
Metadata.json Part 3 Topic labels for classification


### Part 1 Word Embeddings [25 Marks]

1. TF-IDF and PMI Weighted Representations

1.1 TF-IDF Weighting [4 marks]

- Build a term–document matrix from cleaned.txt. Restrict vocabulary to the 10,000 most
    frequent tokens; all others map to <UNK>.
- Compute TF-IDF weights using the standard formula:

```
TF-IDF(w,d) = TF(w,d)× log
```
#### 

#### N

```
1 + df(w)
```
#### 

```
where N is the total number of documents and df(w) is the document frequency of word
w.
```
- Save the resulting weighted matrix as tfidf_matrix.npy.
- Identify and report the top-10 most discriminative words per topic category using TF-IDF
    scores.

1.2 Pointwise Mutual Information (PMI) [5 marks]

- Build a word–word co-occurrence matrix from cleaned.txt with a symmetric context
    window of size k = 5.
- Apply Positive PMI (PPMI) weighting to the co-occurrence matrix:

```
PPMI(w 1 ,w 2 ) = max
```
#### 

```
0 , log 2
```
```
P(w 1 ,w 2 )
P(w 1 )P(w 2 )
```
#### 

- Save the PPMI-weighted matrix as ppmi_matrix.npy.
- Produce a 2-D t-SNE visualisation of the 200 most frequent tokens, colour-coded by
    semantic category (e.g. politics, sports, geography). Include a legend.
- Report the top-5 nearest neighbours by cosine similarity for at least 10 query words.
2. Skip-gram Word2Vec

2.1 Implementation [9 marks]

Train a Skip-gram Word2Vec model on cleaned.txt. The model must:

- Maintain separate centre and context embedding matrices, V and U , both of dimension
    |V|×d.
- Use a noise distribution Pn(w)∝ f(w)^3 /^4 for K = 10 noise samples per positive pair.
- Optimise the binary cross-entropy loss over a context window of size k = 5:

```
L =− log σ(u⊤ovc)−
```
```
K
```
## ∑

```
k= 1
```
```
log σ(−u⊤wkvc)
```
- Train for at least 5 epochs with batch size≥ 512.
- Report training loss at regular intervals with a plotted loss curve.
- Save the averaged final embeddings^12 (V +U) as embeddings_w2v.npy.


```
Required Hyperparameters
d = 100, k = 5, K = 10, η = 0 .001 (Adam)
```
2.2 Evaluation [7 marks]

Nearest Neighbours and Analogy [4 marks]

- Report the top-10 nearest neighbours for each of the following query words:
    Pakistan, Hukumat, Adalat, Maeeshat, Fauj, Sehat, Taleem, Aabadi.
- Construct 10 analogy tests of the form a : b :: c :? using the vector arithmetic v(b)−
    v(a)+ v(c). Report top-3 candidates per test. At least 5 must be correct.
- In 2–3 sentences, assess whether the embeddings capture meaningful semantic relation-
    ships.

Four-Condition Comparison [3 marks]

Train and evaluate under all four conditions below. For each, report the top-5 neighbours for
5 query words and MRR on 20 manually labelled word pairs. Discuss which condition yields
the best embeddings and whether increasing d helps.

```
ID Condition Description
```
C1 PPMI baseline PPMI-weighted co-occurrence vectors
C2 Skip-gram on raw.txt Word2Vec trained on the unprocessed corpus
C3 Skip-gram on cleaned.txt Word2Vec trained on the cleaned corpus
C4 Skip-gram, d = 200 Condition C3 with doubled embedding dimension


### Part 2 Sequence Labeling: POS Tagging & NER [25 Marks]

3. Dataset Preparation [5 marks]
    1. Randomly select 500 sentences from cleaned.txt, ensuring at least 100 sentences from
       each of 3 distinct topic categories in Metadata.json.
    2. POS annotation. Assign one of the 12 tags below to every token. Use a rule-based
       tagger built on the stemmer/lemmatizer from Assignment 1, supported by a hand-crafted
       lexicon of at least 200 entries per major category.
          NOUN VERB ADJ ADV PRON DET CONJ POST NUM PUNC UNK
    3. NER annotation. Annotate every token using the BIO scheme below. A seed gazetteer
       must cover at least 50 Pakistani persons, 50 locations, and 30 organisations.
          B-PER I-PER B-LOC I-LOC B-ORG I-ORG B-MISC I-MISC O
    4. Split the annotated data 70/15/15 (train/val/test), stratified by topic. Report the class-
       label distribution for both tasks.
4. BiLSTM Sequence Labeler [10 marks]

Build a 2-layer bidirectional LSTM sequence labeler. The model must:

- Be initialised with the Word2Vec embeddings from Part 1 (condition C3), evaluated in
    both frozen and fine-tuned modes. Report validation F1 for each.
- Produce per-token contextual representations by concatenating forward and backward
    hidden states: ht= [

#### −→

```
ht∥
```
#### ←−

```
ht].
```
- Apply dropout of p = 0 .5 between LSTM layers.
- For NER: decode using a CRF output layer with a learnable tag-transition matrix. Infer-
    ence must use the Viterbi algorithm.
- For POS: decode with a linear classifier and cross-entropy loss.
- Handle variable-length sequences correctly; padding positions must not contribute to the
    loss.
- Train with Adam (η = 10 −^3 , weight decay 10−^4 ) and early stopping on validation F
    with patience of 5 epochs. Plot training and validation loss per epoch.
5. Evaluation [10 marks]

5.1 POS Tagging [4 marks]

- Report token-level accuracy and macro-F1 on the test set.
- Present a confusion matrix over all 12 tags.
- Identify the 3 most confused tag pairs and provide at least 2 example sentences per pair.
- Compare frozen vs. fine-tuned embedding modes in a summary table.

5.2 NER [4 marks]

- Report entity-level precision, recall, and F1 per type (PER, LOC, ORG, MISC) and over-
    all, evaluated with conlleval.
- Compare results with and without the CRF output layer.


- Provide an error analysis of 5 false positives and 5 false negatives, with explanations.

5.3 Ablation Study [2 marks]

Run each ablation independently on the same data split. Report numeric results and discuss
each finding.

```
ID Change What is being tested
```
A1 Unidirectional LSTM only Value of backward context for sequence
labeling
A2 No dropout Effect of dropout regularisation
A3 Random embedding initialisation Contribution of pre-trained embeddings from
Part 1
A4 Softmax output instead of CRF (NER) Whether structured decoding improves entity
detection


### Part 3 Transformer Encoder for Topic Classification [20 Marks]

6. Dataset Preparation [2 marks]
    1. Assign each article from Metadata.json to one of the 5 categories below, guided by the
       listed keywords.

```
# Category Indicative keywords
1 Politics election, government, minister, parliament
2 Sports cricket, match, team, player, score
3 Economy inflation, trade, bank, GDP, budget
4 International UN, treaty, foreign, bilateral, conflict
5 Health & Society hospital, disease, vaccine, flood, education
```
2. Represent each article as a token-ID sequence from cleaned.txt, padded or truncated to
    256 tokens.
3. Split 70/15/15 stratified by category. Report the class distribution.
7. Transformer Encoder [10 marks]

Implement a Transformer encoder for 5-class topic classification. Each component listed
below must be a separate, self-contained module. No PyTorch built-in Transformer classes
may be used.

- Scaled dot-product attention that accepts an optional padding mask and returns both
    the output and the attention weights.

```
Attention(Q,K,V) = softmax
```
#### 

#### QK⊤

#### √

```
dk
```
#### 

#### V

- Multi-head self-attention with h = 4 heads, dmodel= 128, and dk= dv= 32 per head,
    with separate projection matrices per head and a shared output projection.
- Position-wise feed-forward network: two linear layers with a ReLU activation, inner
    dimension dff= 512.
- Sinusoidal positional encoding stored as a fixed (non-learned) buffer and added to the
    input embeddings.

```
PE(pos, 2 i)= sin
```
```
 pos
100002 i/d
```
#### 

```
, PE(pos, 2 i+ 1 )= cos
```
```
 pos
100002 i/d
```
#### 

- 4 stacked encoder blocks, each using Pre-Layer Normalisation:

```
x← x+ Dropout(MultiHead(LN(x)))
x← x+ Dropout(FFN(LN(x)))
```
- Classification head: a learned [CLS] token prepended to every sequence; its output rep-
    resentation is passed through an MLP (128→ 64 → 5) for classification.


```
Training Requirements
```
```
Optimise with AdamW (η = 5 × 10 −^4 , weight decay 0.01) and a cosine learning-rate
schedule with 50 warmup steps. Train for 20 epochs and plot training and validation loss
and accuracy per epoch.
```
8. Evaluation [8 marks]

8.1 Results [4 marks]

- Report test accuracy and macro-F1.
- Present a 5× 5 confusion matrix.
- For 3 correctly classified articles, plot attention weight heatmaps from at least 2 heads of
    the final encoder layer.

8.2 BiLSTM vs. Transformer Comparison [4 marks]

Write 10–15 sentences in your notebook addressing all five questions below:

1. Which model achieves higher accuracy, and by how much?
2. Which model converged in fewer epochs?
3. Which model was faster to train per epoch, and why?
4. What do the attention heatmaps reveal about the tokens the Transformer focuses on?
5. Given a dataset of only 200–300 articles, which architecture is more appropriate and
    why?


### GitHub Submission Version Control Component [5 Marks]

All code must be in a public GitHub repository before the deadline. Include the repository
URL in both your report and your notebook.

- Repository name: i23-XXXX-NLP-Assignment
- Folder structure must mirror the zip submission layout.
- Commit history must reflect incremental progress — a single bulk commit receives zero.
- Include a README.md with instructions to reproduce each part.

GitHub URL:

Criteria Marks

Public repository with correct name and folder structure 1
All code committed (notebook + scripts) 2
Meaningful commit history (≥ 5 commits) 1
README.md with reproduction instructions 1

Total 5


### Grading Rubric Mark Breakdown [75 Marks]

```
Reminder
Any notebook cell with no output receives zero. All figures must include axis labels and
a title.
```
Part 1 — Word Embeddings [25 marks]

Component Marks Key Criteria

Term–document matrix
construction

```
2 Correct frequency counts; vocabulary capped at 10K
```
TF-IDF weighting +
saved matrix

```
2 Correct IDF formula; top-10 words per category reported
```
PPMI co-occurrence
matrix

```
3 Context window k = 5; PPMI formula correctly applied
```
t-SNE plot + nearest
neighbours

```
2 Labelled; colour-coded; 10 query words reported
```
Vocabulary, noise table,
training pairs

```
3 f(w)^3 /^4 distribution; correct skip-gram pairs
```
Skip-gram model with
BCE loss

```
3 Two embedding matrices; correct loss formulation
```
Training with loss
curve + saved embed-
dings

```
2 Loss plotted;^12 (V+U) saved
```
Nearest-neighbour and
analogy evaluation

```
4 ≥ 5 correct analogies; 10 query words reported
```
Four-condition compar-
ison with MRR

```
4 All 4 conditions run; MRR computed; discussion present
```
Written analysis of em-
bedding quality

```
0 Assessed holistically alongside condition comparison
```
Total 25 —

Part 2 — BiLSTM Sequence Labeling [25 marks]


Component Marks Key Criteria

POS and NER annotation 5 500 sentences; rule-based tagger; gazetteer
Embedding initialisation, frozen and
fine-tuned

```
2 Both modes evaluated and compared
```
2-layer bidirectional LSTM with
dropout

```
3 Bidirectional; 2 layers; dropout applied
```
CRF + Viterbi decoding (NER) 2 Learnable transitions; Viterbi inference
Variable-length sequence handling 1 Padding excluded from loss
POS metrics, confusion matrix, error
analysis

```
4 Accuracy; F1; 3 confused pairs analysed
```
NER metrics, CRF comparison, error
analysis

```
4 conlleval scores; with/without CRF
```
Ablation A1–A4 with discussion 2 All 4 ablations run and discussed
Training curves + early stopping 2 Val F1 monitored; curves plotted

Total 25 —

Part 3 — Transformer Encoder [20 marks]

Component Marks Key Criteria

5-class labelled dataset, stratified split 2 Distribution reported
Scaled dot-product attention module 1 Mask supported; weights returned
Multi-head self-attention (4 heads) 2 Independent projections per head
Position-wise FFN 1 Correct dimensions; ReLU; dropout
Sinusoidal PE as fixed buffer 1 Non-learned; added before first block
Pre-LN encoder block, stacked× 4 2 Pre-LN residual connections; 4 blocks
CLS token + AdamW + cosine LR + train-
ing curves

```
3 All four present and correct
```
Accuracy, F1, confusion matrix 2 All three reported
Attention heatmaps (≥ 2 heads, 3 articles) 2 Token labels readable
Written BiLSTM vs. Transformer compar-
ison

```
4 All 5 questions addressed
```
Total 20 —

GitHub Submission [5 marks]

Component Marks Key Criteria

Public repository, correct name and structure 1 Matches submission layout
All code committed (notebook + scripts) 2 Runnable from README.md
Meaningful commit history (≥ 5 commits) 1 No single bulk commit
README.md with reproduction instructions 1 Clear setup and run steps

Total 5 —

Grand Total 75 —


### Submission Guidelines Deliverables [—]

Zip file name: i23-XXXX_Assignment2_DS-X.zip

```
i23 -XXXX_Assignment2_DS -X/
|-- i23 -XXXX_Assignment2_DS -X.ipynb (all cells executed)
|-- report.pdf (2-3 pages , Times New
Roman 12pt)
|-- embeddings/
| |-- tfidf_matrix.npy
| |-- ppmi_matrix.npy
| |-- embeddings_w2v.npy
| ‘-- word2idx.json
|-- models/
| |-- bilstm_pos.pt
| |-- bilstm_ner.pt
| ‘-- transformer_cls.pt
‘-- data/
|-- pos_train.conll
|-- pos_test.conll
|-- ner_train.conll
‘-- ner_test.conll
```
```
Report Format
PDF only. Times New Roman, 12pt, 1.5 line spacing, 2–3 pages. Sections: Overview,
Part 1 Results, Part 2 Results, Part 3 Results, Conclusion.
No .md, .docx, or any other format will be accepted.
```
### — Good Luck —


