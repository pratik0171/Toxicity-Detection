import streamlit as st
import pickle
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load trained model
model = pickle.load(open("svm_toxicity_model.pkl", "rb"))

# Load TF-IDF vectorizer
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Stopwords
stop_words = set(ENGLISH_STOP_WORDS)

# Text cleaning function
def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove extra spaces
    text = text.strip()

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
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

st.write(
    "Enter a comment to check whether it is toxic or non-toxic."
)

user_input = st.text_area("Enter a comment")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        result = predict_toxicity(user_input)

        st.subheader("Prediction")

        st.write(result)
