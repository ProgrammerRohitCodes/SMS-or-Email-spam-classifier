# SMS / Email Spam Classifier

A lightweight spam classifier for SMS or short email messages using classic NLP and scikit-learn. The project trains a TF-IDF vectorizer and an ensemble VotingClassifier (soft voting) composed of tuned Logistic Regression, Multinomial Naive Bayes, and Random Forest classifiers.

This repository contains the model training notebook, saved model and vectorizer, and example code to run predictions.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Approach](#approach)
- [Modeling & Hyperparameter Tuning](#modeling--hyperparameter-tuning)
- [Performance](#performance)
- [Saved Artifacts](#saved-artifacts)
- [Usage / Prediction Example](#usage--prediction-example)
- [Notes & Considerations](#notes--considerations)
- [License](#license)
- [Credits](#credits)

---

## Getting Started

Requirements:
- Python 3.8+
- scikit-learn
- pandas
- joblib

Install common dependencies:
```bash
pip install -r requirements.txt
# or
pip install scikit-learn pandas joblib
```

---

## Dataset

This project was built with a typical SMS spam dataset (short messages labeled as ham (0) or spam (1)). The training/test split used in the notebook resulted in 1,034 test samples (889 ham, 145 spam).

---

## Approach

- Text preprocessing: tokenization and TF-IDF vectorization.
- Models trained and tuned via GridSearchCV:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Random Forest
- Final classifier: VotingClassifier (voting='soft') combining the best estimators from above.

TF-IDF was used to convert raw text to features, and GridSearchCV was used to find the best hyperparameters for each base model before ensembling.

---

## Modeling & Hyperparameter Tuning

- Logistic Regression: tuned C and solver (best found: C=1, solver='liblinear')
- MultinomialNB: tuned alpha (best found: alpha=0.1)
- Random Forest: tuned n_estimators, max_depth, and min_samples_split (best combination chosen by GridSearch)
- Final ensemble: soft voting VotingClassifier built from the best estimators

The vectorizer and trained VotingClassifier were saved using joblib.

---

## Performance

Final evaluation (on the held-out test set):

- Accuracy: 0.9845261121856866 (≈ 0.98)

Classification report:
- Test set size: 1,034 samples
- Class mapping: 0 = ham, 1 = spam

Precision / Recall / F1-score / Support
- Class 0 (ham): precision 0.98, recall 1.00, f1-score 0.99, support 889
- Class 1 (spam): precision 1.00, recall 0.89, f1-score 0.94, support 145

Averages:
- accuracy: 0.98 (overall)
- macro avg: precision 0.99, recall 0.94, f1-score 0.97
- weighted avg: precision 0.98, recall 0.98, f1-score 0.98

Confusion matrix:
```
[[889   0]
 [ 16 129]]
```
Interpreting the confusion matrix:
- True Negatives (ham correctly classified): 889
- False Positives (ham -> spam): 0
- False Negatives (spam -> ham): 16
- True Positives (spam correctly classified): 129

This indicates high precision (few false positives) and very high overall accuracy. Most errors are false negatives (spam misclassified as ham), which may be important to address depending on the use case.

---

## Saved Artifacts

The notebook saves the trained vectorizer and model using joblib. In the training notebook these were saved to:
- Vectorizer: `/content/vectorizer.pkl`
- Trained VotingClassifier: `/content/spam_model.pkl`

(If you run the notebook locally you can change the save paths as required.)

---

## Usage / Prediction Example

Example code to load the saved artifacts and make a prediction:

```python
import joblib

# load artifacts (update paths as needed)
vectorizer = joblib.load("path/to/vectorizer.pkl")
model = joblib.load("path/to/spam_model.pkl")

def predict_message(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {"label": int(pred), "probability": proba.tolist()}

# Example
message = "Congratulations! You've won a free voucher. Call now!"
result = predict_message(message)
print(result)  # e.g. {'label': 1, 'probability': [0.02, 0.98]}
```

Notes:
- `label` is 0 for ham and 1 for spam.
- `probability` is a 2-element list [P(ham), P(spam)].

---

## Notes & Considerations

- The dataset was imbalanced (more ham than spam). The model performs very well overall, but most remaining errors are false negatives (spam classified as ham). If minimizing spam misses is critical, consider:
  - Using class-weighted classifiers
  - Resampling (SMOTE / upsampling spam)
  - Threshold adjustment on predicted probabilities
- This solution is optimized for short messages. For longer emails, more preprocessing may be necessary (HTML stripping, headers, attachments).
- Always validate model behavior on your target data before deployment.

---

## License

This repository is provided under the MIT License. See LICENSE file for details.

---

## Credits

- Author / Maintainer: ProgrammerRohitCodes
- Built with: scikit-learn, pandas, joblib

If you want additions (API example, Dockerfile, or model export instructions), open an issue or submit a PR. Thank you!
