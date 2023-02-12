import streamlit as st
import pandas as pd
import re
import pickle
import string
from langdetect import detect
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')


def clear_text(text):
    # List of all English characters and numbers
    eng_chars = string.ascii_letters + string.digits
    # List of all punctuation characters
    punc_chars = string.punctuation
    # Combine the two lists
    remove_chars = eng_chars + punc_chars
    # Remove all characters in the remove_chars list from the text
    text = ''.join(c for c in text if c not in remove_chars)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


punjabi_lexicon = pd.read_csv("./data/punjabi_lexicon.csv")
models_marathi = []
model_files = ["./models/GBC_marathi.pickle", "./models/LogReg_marathi.pickle",
               "./models/SupportVec_marathi.pickle", "./models/RandForest_marathi.pickle"]
for file in model_files:
    with open(file, "rb") as f:
        model = pickle.load(f)
        models_marathi.append(model)

models_gujarati = []
model_files = ["./models/GBC_gujarati.pickle", "./models/LogReg_gujarati.pickle", "./models/SupportVec_gujarati.pickle",
               "./models/RandForest_gujarati.pickle", "./models/SupportVec_gujarati.pickle", "./models/KNeighbors_gujarati.pickle"]
for file in model_files:
    with open(file, "rb") as f:
        model = pickle.load(f)
        models_gujarati.append(model)

with open("./models/vectorizer_marathi.pickle", "rb") as f:
    vectorizer_marathi = pickle.load(f)
with open("./models/vectorizer_gujarati.pickle", "rb") as f:
    vectorizer_gujarati = pickle.load(f)


def punjabi_sentiment_score(text, lexicon):
    words = word_tokenize(text)
    positive_score = 0
    negative_score = 0
    for word in words:
        if word in lexicon['Word'].values:
            positive_score += lexicon.loc[lexicon['Word']
                                          == word, 'Positive Score'].values[0]
            negative_score += lexicon.loc[lexicon['Word']
                                          == word, 'Negative Score'].values[0]
            finalScore = positive_score - negative_score
    return positive_score - negative_score


def mr_gu_sentiment(text, models, vectorizer):
    sentence = vectorizer.transform([text])
    predictions = []
    for model in models:
        prediction = model.predict(sentence)
        predictions.append(prediction)
    # Take the most common prediction
    sentiment = max(set(map(tuple, predictions)), key=predictions.count)
    return sentiment

import streamlit as st

def positive():
    st.markdown("<h2 style='text-align:center; color:green'>Sentence is Positive</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:80px; text-align:center'>😊</h2>", unsafe_allow_html=True)
    
def negative():
    st.markdown("<h3 style='text-align:center; color:red'>Sentence is Negative</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:80px; text-align:center'>😔</h2>", unsafe_allow_html=True)

def neutral():
    st.markdown("<h3 style='text-align:center; color:blue'>Sentence is Neutral</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:80px; text-align:center'>😐</h2>", unsafe_allow_html=True)


def run_streamlit_app():
    st.set_page_config(page_icon=":smile:",
                       layout="wide")
    st.markdown("<h1 style='text-align:center;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    text_input = st.text_input("Enter Text (In Punjabi, Gujarati or Marathi)")
    text_input = clear_text(text_input)
    if text_input:
        language = detect(text_input)
        if language == 'mr':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Marathi</h2>", unsafe_allow_html=True)
            sentiment = mr_gu_sentiment(text_input, models_marathi, vectorizer_marathi)
            sentiment = sentiment[0]
            if sentiment == 0:
                neutral()
            elif sentiment == 1:
                positive()
            else:
                negative()
        elif language == 'pa':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Punjabi</h2>", unsafe_allow_html=True)
            score = punjabi_sentiment_score(text_input, punjabi_lexicon)
            if score > 0.15:
                positive()
            elif score < -0.15:
                negative()
            else:
                neutral()

        elif language == 'gu':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Gujarati</h2>", unsafe_allow_html=True)
            sentiment = mr_gu_sentiment(text_input, models_gujarati, vectorizer_gujarati)
            if sentiment == 0:
                negative()
            else:
                positive()
        else:
            st.warning("The language is not Marathi, Punjabi or Gujarati.")

run_streamlit_app()