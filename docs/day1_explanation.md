# Day 1 Code Explanation (Detailed)

This document explains **each line** of `day1.py` and also covers the conceptual background you asked about: what a confusion matrix is, why we chose the GLUE SST‑2 dataset, and why each step matters.

---  

## 📦 Imports

```python
import numpy as np
```
NumPy provides the foundation for numerical computing in Python. Many scikit‑learn functions expect NumPy arrays (or sparse matrices built on top of them). We import it as `np` – although we don’t call it directly in this tiny script, keeping the import makes it easy to extend later (e.g., custom weighting, manual matrix ops).

```python
import pandas as pd
```
Pandas offers the `DataFrame` structure, handy for inspecting tabular data (`.head()`, `.describe()`). Not used in the core pipeline here, but useful if you want to peek at the raw sentences/labels.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
Scikit‑learn’s TF‑IDF vectorizer turns a list of raw strings into a **sparse matrix** of TF‑IDF features (see Section 4). This is the workhorse of traditional‑ML text pipelines.

```python
from sklearn.linear_model import LogisticRegression
```
A linear classifier that outputs calibrated probabilities via the sigmoid function. Works especially well with high‑dimensional sparse data like TF‑IDF.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
```
These functions compute the standard evaluation metrics for classification:
* **accuracy_score** – fraction of correct predictions.
* **precision_recall_fscore_support** – precision, recall, F1‑score (and support).
* **confusion_matrix** – builds the TP/FP/TN/FN table (see Section 7.3).

```python
import matplotlib.pyplot as plt
```
The core plotting library; we use its pyplot interface to create figures.

```python
import seaborn as sns
```
Built on top of matplotlib, seaborn provides a high‑level interface for drawing attractive statistical graphics (e.g., heatmaps with automatic colour bars).

```python
from datasets import load_dataset
```
The 🤗datasets library offers a unified way to download and cache many public NLP datasets (GLUE, SuperGLUE, SQuAD, etc.) with a single line, eliminating the need to hunt for CSV files or write custom loaders.

---  

## 📚 Why GLUE SST‑2?

* **GLUE** = General Language Understanding Evaluation benchmark. It bundles several sentence‑level tasks (sentiment, paraphrase detection, entailment, etc.) that are widely used to test and compare models.
* **SST‑2** (Stanford Sentiment Treebank – binary) is the sentiment‑analysis subset of GLUE:
  * **Binary labels:** 0 = negative, 1 = positive → ideal for logistic regression.
  * **Size:** ~67 k training sentences, 872 validation, 1.8 k test (test labels are hidden for the official leaderboard, so we use validation for evaluation).
  * **Clean & well‑studied:** Many baselines exist (TF‑IDF + LR typically achieves ~78‑80 % accuracy), making it easy to verify that the pipeline is working.

Using a standard benchmark lets you **compare your results** to published numbers without worrying about label noise or custom splits. It’s a small, clean, widely‑used sentence‑classification task that lets us focus on the *pipeline* (vectorization → model → evaluation) rather than data‑collection headaches.

---  

## 1️⃣ Loading the data

```python
dataset = load_dataset('glue', 'sst2')
train_data = dataset['train']
val_data   = dataset['validation']
test_data  = dataset['test']
```
* `load_dataset` downloads the dataset (if not cached) and returns a `DatasetDict` with keys `'train'`, `'validation'`, `'test'`.
* We assign each split to its own variable for readability.

```python
X_train = train_data['sentence']
y_train = train_data['label']
X_val   = val_data['sentence']
y_val   = val_data['label']
X_test  = test_data['sentence']
y_test  = test_data['label']
```
* Extract the two columns we need:
  * `'sentence'` → raw text (`X_*`).
  * `'label'`    → integer class (`y_*`). In SST‑2, labels are 0 (negative) and 1 (positive). The official test set uses label `-1` (unset) – we therefore evaluate only on the validation split for now.

---  

## 2️⃣ From raw text to numbers – TF‑IDF Vectorizer

```python
vectorizer = TfidfVectorizer(
    stop_words='english',   # drop common words like “the”, “and”, “is”
    ngram_range=(1,2),      # use both unigrams and bigrams
    max_features=5000       # keep the 5 000 most frequent terms
)
```
| Parameter | What it does |
|-----------|--------------|
| `stop_words='english'` | Removes high‑frequency, low‑information words that would otherwise dominate the feature space. |
| `ngram_range=(1,2)` | Considers both single words (unigrams) **and** pairs of consecutive words (bigrams). Bigrams capture simple linguistic patterns like “not good” or “very happy”. |
| `max_features=5000` | Limits the dimensionality to keep the model tractable and training fast. We retain the 5 000 terms with highest corpus‑wide term frequency. |

**Fit & Transform**

```python
X_train_vec = vectorizer.fit_transform(X_train)   # learn vocab + idf, then convert
X_val_vec   = vectorizer.transform(X_val)         # transform using the same vocab (no refit!)
```

* `fit_transform` does two things:
  1. **Learn** the vocabulary (which terms to keep) and compute **inverse document frequency (idf)** scores for each term.
  2. **Transform** the raw sentences into a **sparse matrix** where each row = a sentence, each column = a term, and the cell value = TF‑IDF weight.
* `transform` applies the *exact same* vocabulary and idf to new data (validation/test) – we must **not** refit, otherwise we would leak information from the validation set into training (data leakage → overly optimistic scores).

The resulting objects are `scipy.sparse.csr_matrix` instances – memory‑efficient because most entries are zero (a sentence contains only a handful of the 5 000 possible terms).

---  

## 3️⃣ Model – Logistic Regression

```python
clf = LogisticRegression(max_iter=1000)   # increased max_iter to ensure convergence
clf.fit(X_train_vec, y_train)
```
* Logistic regression models the probability that a sentence belongs to the positive class (`label=1`) via the sigmoid function:

\[
P(y=1|x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
\]

* It learns a weight vector `w` (one coefficient per TF‑IDF feature) and a bias `b` by minimizing the **log‑loss** (cross‑entropy) on the training data.
* `max_iter=1000` tells the solver (default “lbfgs” for small‑to‑medium problems) to keep iterating up to 1 000 times if it hasn’t converged yet. The default (`100`) sometimes stops early on high‑dimensional sparse data, leading to a *ConvergenceWarning*.
* Because the input is sparse, scikit‑learn internally uses a solver that works efficiently with CSR matrices – no need to densify the data.

---  

## 4️⃣ Making Predictions

```python
val_pred = clf.predict(X_val_vec)   # returns 0 or 1 for each validation sentence
```
* `predict` applies the learned decision rule: if `P(y=1|x) ≥ 0.5` → predict `1`, else predict `0`.
* (If you wanted probabilities you’d call `clf.predict_proba(X_val_vec)`.)

---  

## 5️⃣ Evaluation Metrics – Why more than just accuracy?

### 5.1 Accuracy
```python
val_acc = accuracy_score(y_val, val_pred)
```
* Fraction of correctly classified sentences.
* **Limitation:** Accuracy can be misleading if classes are imbalanced. For example, if 90 % of sentences are negative, a model that always predicts “negative” would achieve 90 % accuracy while being useless for detecting positives.

### 5.2 Precision, Recall, F1 (binary case)
```python
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    y_val, val_pred, average='binary')
```
* **Precision** = TP / (TP + FP) – “Of all sentences I predicted as positive, how many were truly positive?” (measures false‑positive rate).
* **Recall**    = TP / (TP + FN) – “Of all actual positive sentences, how many did I catch?” (measures false‑negative rate).
* **F1‑score**  = harmonic mean of precision and recall:

\[
F1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
\]

* Setting `average='binary'` tells the function to compute these metrics for the **positive class** (`label=1`).  
* For multi‑class problems you could use `average='macro'` (unweighted mean of per‑class scores) or `average='weighted'` (weighted by class frequency).

### 5.3 Confusion Matrix
```python
cm = confusion_matrix(y_val, val_pred)
```
* Builds a 2×2 table:

|                     | **Pred = 0** (Neg) | **Pred = 1** (Pos) |
|---------------------|-------------------|-------------------|
| **True = 0** (Neg)  | TN                | FP                |
| **True = 1** (Pos)  | FN                | TP                |

* **Interpretation**
  * **TN (True Negative)** – correctly predicted negative.
  * **TP (True Positive)** – correctly predicted positive.
  * **FP (False Positive)** – we said “positive” but it was actually negative (type I error).
  * **FN (False Negative)** – we said “negative” but it was actually positive (type II error).

* Visualising it with a heatmap makes it easy to see where the model confuses the classes.

```python
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Test Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print('Saved confusion_matrix.png')
```
* `annot=True` writes the numeric count inside each cell.  
* `fmt='d'` formats the numbers as integers.  
* `cmap='Blues'` uses a blue colour scale.  
* Axis tick labels are replaced with readable class names.  
* The figure is saved as a PNG file in the working directory and then closed to free memory.

---  

## 6️⃣ Reproducibility & Next Steps

| Item | What you have | How to reuse / extend |
|------|---------------|-----------------------|
| **Script** (`day1.py`) | End‑to‑end runnable file. | `python day1.py` (ensure `nlp-env` is activated). |
| **Notebook** (`day1.ipynb`) | Same code in an interactive format. | `jupyter notebook day1.ipynb` to tweak parameters live. |
| **Explanation doc** (`docs/day1_explanation.md`) | This file – detailed line‑by‑line + conceptual background. | Read it, copy sections into your own wiki or notes. |
| **Git repo** | All files committed (see `git log`). | Push to a remote (GitHub/GitLab) for backup/sharing. |
| **Model/vectorizer** | Not persisted yet. | Add `joblib.dump(vectorizer, 'tfidf_vec.joblib')` and `joblib.dump(clf, 'logreg_model.joblib')` to reuse later without retraining. |
| **Potential experiments** | - Try different `ngram_range` (e.g., (1,3) for trigrams). <br> - Increase `max_features` to 10 000 or 20 000. <br> - Swap Logistic Regression for `LinearSVC` or `RandomForestClassifier`. <br> - Use `cross_val_score` for a more robust estimate. <br> - Finally, evaluate on the official GLUE test set (once you obtain the true labels) and compare to the published leaderboard. |

---  

## 📚 Quick Glossary

| Term | Meaning in this context |
|------|--------------------------|
| **Confusion Matrix** | Table showing counts of true vs. predicted class; reveals where the model makes mistakes (FP/FN). |
| **GLUE** | General Language Understanding Evaluation – a collection of diverse NLU tasks used as a standard benchmark. |
| **SST‑2** | Binary sentiment‑analysis subset of GLUE (Stanford Sentiment Treebank); ideal for intro text‑classification demos. |
| **TF‑IDF** | Term Frequency‑Inverse Document Frequency – weights words by how often they appear in a document but down‑weights frequent‑across‑corpus words, highlighting distinctive terms. |
| **Logistic Regression** | Linear model predicting probability of the positive class via a sigmoid; works well with high‑dimensional sparse features like TF‑IDF. |
| **Precision / Recall / F1** | Metrics focusing on the quality of positive predictions. Precision = “Of the ones I said were positive, how many were actually positive?” Recall = “Of the actual positives, how many did I capture?” F1 balances both. |

---  

## 🎉 What You’ve Accomplished in Day 1

1. Set up a clean, reproducible Python virtual environment.  
2. Loaded a standard NLP benchmark (GLUE SST‑2).  
3. Converted raw sentences into a numeric TF‑IDF feature matrix.  
4. Trained a Logistic Regression classifier.  
5. Evaluated with accuracy, precision, recall, F1, and visualised errors via a confusion matrix.  
6. Saved code, notebook, plots, and a detailed line‑by‑line explanation.  
7. Committed everything to a local Git repository, ready for sharing or further work.  

You now have a solid baseline (~78 % validation accuracy) that you can improve with the techniques we’ll cover in the upcoming days (word embeddings → shallow neural nets → transformers).  

---  

### 👉 Ready for Day 2?

When you are, let me know and we’ll:
- Activate the `nlp-env` again.  
- Install `gensim`.  
- Train Word2Vec (or GloVe) embeddings on the same sentences, examine word similarities, solve analogies, and visualise the vectors with t‑SNE/UMAP.  

Happy modeling! 🚀