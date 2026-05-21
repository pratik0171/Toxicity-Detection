# Toxicity Detection System

An NLP-based Toxicity Detection System that classifies user comments as Toxic or Non-Toxic using TF-IDF Vectorization and Support Vector Machine (LinearSVC).

---

# Project Overview

This project detects toxic comments from textual data using Natural Language Processing (NLP) and Machine Learning techniques.

The system can identify:
- Abusive comments
- Hate speech
- Offensive language
- Toxic behavior in online conversations

The model was trained on a dataset containing over 100,000 comments.

---

# Features

- Text preprocessing pipeline
- TF-IDF feature engineering
- Toxicity classification using LinearSVC
- Real-time comment prediction
- Streamlit web application
- Model deployment support

---

# Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Pickle

---

# Machine Learning Pipeline

Raw Text
↓
Text Cleaning
↓
Tokenization
↓
Stopword Removal
↓
Lemmatization
↓
TF-IDF Vectorization
↓
LinearSVC Classification
↓
Prediction Output

---

# Dataset Information

The dataset contains:
- Text comments
- Toxicity labels

Target Labels:
- 0 → Non-Toxic
- 1 → Toxic

---

# NLP Preprocessing Steps

The following preprocessing techniques were applied:

- Lowercasing
- URL removal
- Number removal
- Punctuation removal
- Stopword removal
- Lemmatization

---

# Model Used

## LinearSVC

Linear Support Vector Classification was used because:
- Excellent performance on sparse TF-IDF vectors
- Fast training speed
- Strong NLP classification capability
- High accuracy for text classification tasks

---

# Feature Engineering

## TF-IDF Vectorization

TF-IDF converts text into numerical vectors by assigning importance scores to words and phrases.

Configuration:
- max_features = 10000
- ngram_range = (1,2)

---

# Project Structure

```bash
toxicity_detection/
│
├── app.py
├── requirements.txt
├── Procfile
├── svm_toxicity_model.pkl
├── tfidf_vectorizer.pkl
├── toxicity.csv
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/toxicity-detection.git
```

## Move into Project Folder

```bash
cd toxicity-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Example Predictions

| Input Text | Prediction |
|---|---|
| "You are amazing" | Non-Toxic |
| "Nobody likes you" | Toxic |
| "Have a great day" | Non-Toxic |
| "You are an idiot" | Toxic |

---

# Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~94% |
| Precision | High |
| Recall | Moderate |
| F1-Score | Strong |

---

# Future Improvements

- BERT-based toxicity detection
- Multi-class toxicity classification
- Real-time chat moderation
- REST AtPI integration
- Docker deploymen
- Cloud deployment using AWS/GCP

---

# Deployment

The application can be deployed using:
- Render
- Hugging Face Spaces
- Streamlit Cloud


# Learning Outcomes

Through this project, the following concepts were implemented:

- Natural Language Processing
- Text preprocessing
- Feature engineering
- Machine learning classification
- Model evaluation
- Web application deployment

