import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Load SST-2 dataset (binary sentiment)
dataset = load_dataset('glue', 'sst2')
train_data = dataset['train']
val_data = dataset['validation']
# test_data = dataset['test']  # test labels are -1, skip

X_train = train_data['sentence']
y_train = train_data['label']
X_val = val_data['sentence']
y_val = val_data['label']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Predictions on validation set
val_pred = clf.predict(X_val_vec)

# Metrics
val_acc = accuracy_score(y_val, val_pred)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average='binary')

print(f'Validation Accuracy: {val_acc:.4f}')
print(f'Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Validation Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print('Saved confusion_matrix.png')