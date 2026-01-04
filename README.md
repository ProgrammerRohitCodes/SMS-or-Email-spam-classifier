# Email / SMS Spam Classifier

An end-to-end machine learning application that detects whether a given SMS or email message is Spam or Not Spam.  
This project focuses on classical NLP techniques, model evaluation, and practical deployment using Streamlit.

The aim is not just high accuracy, but interpretability, robustness, and a clear ML workflow.

---

## Overview

Spam detection is a common real-world NLP problem where text data must be cleaned, transformed, and classified reliably.  
This repository demonstrates the complete pipeline — from preprocessing raw text to serving predictions through a simple web interface.

---

## Features

- Classifies messages as Spam or Not Spam
- Text preprocessing including:
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Stemming (Porter Stemmer)
- TF‑IDF vectorization for feature extraction
- Ensemble learning using a Soft Voting Classifier
- Lightweight web interface built with Streamlit
- Runs locally without external services

---

## Model & Approach

### Preprocessing
Raw text messages are cleaned and normalized before feature extraction to reduce noise and improve generalization.

### Vectorization
TF‑IDF Vectorizer is used to convert text into numerical features — chosen for interpretability and effectiveness on sparse text data.

### Models Used
Final predictions are produced with a Soft Voting Classifier combining:
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

Hyperparameters for individual models were tuned using GridSearchCV.

---

## Model Performance

Evaluation was performed on a held-out test set.

- Accuracy: ~98.45%

Classification report (high level):

- Spam (Class 1):
  - Precision: 1.00
  - Recall: 0.89
  - F1-score: 0.94
- Not Spam (Class 0):
  - Precision: 0.98
  - Recall: 1.00
  - F1-score: 0.99

Confusion matrix summary:
- Correctly identified most spam messages.
- A small number of spam messages were misclassified as non-spam — expected in realistic datasets.

These results reflect a practical balance between precision and recall without signs of obvious overfitting given the reported metrics.

---

## Project Structure

SMS-or-Email-spam-classifier/
│
├── app.py                     # Streamlit application
├── SMS_Spam_Classifier.ipynb  # Training and evaluation notebook
├── requirements.txt
├── README.md
└── artifacts/
    ├── spam_model.pkl         # Trained voting classifier
    └── vectorizer.pkl         # TF‑IDF vectorizer

---

## Setup & Usage

1. Clone the repository

```bash
git clone https://github.com/ProgrammerRohitCodes/SMS-or-Email-spam-classifier.git
cd SMS-or-Email-spam-classifier
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download NLTK resources (one-time)

Run a short Python snippet (or from a Python REPL / notebook):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal to use the app.

---

## Design Decisions

- TF‑IDF was chosen over dense embeddings to keep the model interpretable and lightweight.
- A voting classifier reduces reliance on a single model and smooths out individual model biases.
- Streamlit was selected for fast prototyping and a simple local UI.
- Trained model artifacts are separated from source code for clarity and reproducibility.

---

## Limitations & Future Improvements

- Currently trained on a static dataset; no automatic handling for concept drift or evolving spam behaviour.
- UI could be extended with confidence/probability scores, batch predictions, and explainability (e.g., important tokens).
- Compare and benchmark transformer-based approaches (BERT, DistilBERT) for potential performance gains.
- Add monitoring to detect when model performance degrades on new incoming data.

---

## Learning Outcome

This project strengthened understanding of:
- Practical NLP preprocessing
- Feature engineering for text data
- Ensemble models and evaluation metrics
- Deploying ML models as usable local applications

---

## Next steps (suggested)

If you'd like, we can:
- Add probability/confidence output in the UI
- Refactor the notebook logic into modular .py files for cleaner code and testing
- Prepare a short LinkedIn post that explains the project and core results
- Add model monitoring or a small pipeline for periodic retraining

Bol, next kya lock karein.

---
```
