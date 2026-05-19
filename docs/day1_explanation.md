# Day 1 Code Explanation

This document explains each line of `day1.py` in detail.

## Imports
```python
import numpy as np
```
NumPy provides support for large, multi‑dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. We import it as `np` for convenience, though it isn't directly used in this script (kept for potential future extensions).

```python
import pandas as pd
```
Pandas offers data structures like DataFrames for easy data manipulation. Imported as `pd`; not used directly here but useful if you later want to inspect the dataset in a tabular form.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
Scikit‑learn's TF‑IDF vectorizer converts a collection of raw documents to a matrix of TF‑IDF features.

```python
from sklearn.linear_model import LogisticRegression
```
Logistic regression model for binary classification.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
```
Metrics to evaluate the classifier: accuracy, precision, recall, F1‑score, and confusion matrix.

```python
import matplotlib.pyplot as plt
```
The core plotting library; we use its pyplot interface.

```python
import seaborn as sns
```
Built on top of matplotlib, provides a high‑level interface for drawing attractive statistical graphics (e.g., heatmap).

```python
from datasets import load_dataset
```
The 🤗datasets library provides easy access to many public datasets (including GLUE SST‑2).

## Load the dataset
```python
dataset = load_dataset('glue', 'sst2')
```
Downloads and loads the GLUE benchmark's SST‑2 (Stanford Sentiment Treebank) binary sentiment dataset. The returned object is a `DatasetDict` with splits: 'train', 'validation', 'test'.

```python
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']
```
Assign each split to its own variable for readability.

```python
X_train = train_data['sentence']
y_train = train_data['label']
X_val = val_data['sentence']
y_val = val_data['label']
X_test = test_data['sentence']
y_test = test_data['label']
```
Extract the text (`sentence`) and label (`label`) columns. In SST‑2, labels are 0 (negative) and 1 (positive). Note: the test split has label `-1` (unset) for the official leaderboard; we therefore use only train/validation for training/evaluation.

## TF‑IDF Vectorizer
```python
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
```
- `stop_words='english'`: removes common English stop words (e.g., "the", "and").
- `ngram_range=(1,2)`: considers both unigrams and bigrams as features.
- `max_features=5000`: keeps only the top 5000 features sorted by term frequency across the corpus, limiting dimensionality.

```python
X_train_vec = vectorizer.fit_transform(X_train)
```
Learn vocabulary and inverse document frequencies from the training sentences, then transform them to a sparse TF‑IDF matrix.

```python
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)
```
Transform validation and test sentences using the *same* vocabulary (no refitting).

## Logistic Regression Model
```python
clf = LogisticRegression(max_iter=1000)
```
- `max_iter=1000`: increases the maximum number of iterations for the solver to converge (default is 100, which may be insufficient for this data).

```python
clf.fit(X_train_vec, y_train)
```
Fit the model to the TF‑IDF‑transformed training data.

## Predictions
```python
val_pred = clf.predict(X_val_vec)
test_pred = clf.predict(X_test_vec)
```
Generate class predictions (0 or 1) for validation and test sets.

## Evaluation Metrics
```python
val_acc = accuracy_score(y_val, val_pred)
```
Proportion of correctly predicted labels in the validation set.

```python
test_acc = accuracy_score(y_test, test_pred)
```
Same for test set (though test labels are -1, so this metric is not meaningful; we therefore focus on validation).

```python
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average='binary')
```
- Computes precision, recall, and F1‑score for the binary case.
- `average='binary'` reports metrics for the positive class (label 1); requires that the target be binary.

```python
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
```
Same for test set (again, not meaningful due to missing labels).

```python
print(f'Validation Accuracy: {val_acc:.4f}')
print(f'Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
```
Formatted output showing four decimal places.

## Confusion Matrix
```python
cm = confusion_matrix(y_test, test_pred)
```
Builds a 2×2 matrix where rows are true labels and columns are predicted labels.

```python
plt.figure(figsize=(5,4))
```
Creates a new figure of size 5×4 inches.

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
```
- `annot=True`: writes the numeric value inside each cell.
- `fmt='d'`: integer formatting.
- `cmap='Blues'`: blue colour scale.
- Axis labels are set to human‑readable class names.

```python
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Test Confusion Matrix')
```
Add axis labels and a title.

```python
plt.savefig('confusion_matrix.png')
```
Saves the figure to a PNG file in the current working directory.

```python
plt.close()
```
Closes the figure to free memory.

```python
print('Saved confusion_matrix.png')
```
Confirmation message.

--- 
### How to Run
1. Ensure you are inside the virtual environment (`nlp-env`) where the required packages are installed.
2. Execute: `python day1.py`
3. The script will print metrics and create `confusion_matrix.png`.

### Extending This Script
- Try different vectorizer parameters (e.g., `max_features`, `ngram_range`).
- Experiment with other classifiers (e.g., `LinearSVC`, `RandomForestClassifier`).
- Use the test set for final evaluation once you have obtained true labels (e.g., from Kaggle or the GLUE leaderboard).
- Save the trained model and vectorizer with `joblib` or `pickle` for later reuse.