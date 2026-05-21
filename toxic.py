import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
model = pickle.load(open("svm_toxicity_model.pkl", "rb"))

# Load TF-IDF vectorizer
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = text.strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Prediction function
def predict_toxicity(text):

    cleaned_text = clean_text(text)

    vectorized_text = tfidf.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    if prediction == 1:
        return "⚠ Toxic Comment"
    else:
        return "✅ Non-Toxic Comment"

# Streamlit UI
st.title("Toxicity Detection System")

user_input = st.text_area("Enter a comment")

if st.button("Predict"):

    result = predict_toxicity(user_input)

    st.subheader("Prediction")
    st.write(result)