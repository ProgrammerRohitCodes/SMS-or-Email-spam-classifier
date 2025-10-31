# Email/SMS Spam Classifier (Streamlit + ML)

An interactive machine learning web app that detects whether a given message is **Spam** or **Not Spam**.  
Built using **Python**, **Streamlit**, and a **Voting Classifier** trained on SMS/Email data.

---

## Features
- Classifies messages as **Spam** or **Not Spam**
- Uses **TF-IDF vectorization** for text processing  
- Built with a **Voting Classifier** combining Logistic Regression, Naive Bayes, and Random Forest  
- Lightweight and runs locally with Streamlit  
- Includes preprocessing (tokenization, stopword removal, stemming)

---

## Setup & Usage

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/Spam-Classifier.git
cd Spam-Classifier
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn nltk joblib
```

### 3. Download NLTK resources (only once)
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

---

## How It Works
1. **Preprocessing**  
   - Converts text to lowercase  
   - Removes punctuation and stopwords  
   - Applies stemming using `PorterStemmer`

2. **Vectorization**  
   - Transforms text into numerical form using `TfidfVectorizer`

3. **Model Prediction**  
   - Trained with a soft-voting ensemble of Logistic Regression, Naive Bayes, and Random Forest  
   - Outputs: `Spam` or `Not Spam`

---

## Model Training
If you want to train your own model, open the notebook:  
```bash
spam_training_notebook.ipynb
```
It includes data cleaning, model building, tuning, and saving:
```python
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(voting, "spam_model.pkl")
```

---

## Tech Stack
- **Python**
- **Streamlit**
- **Scikit-learn**
- **NLTK**
- **Joblib**

---

## License
Open-source project — free for educational and non-commercial use.

---

*(Created by a learner passionate about AI, ML, and automation.)*
