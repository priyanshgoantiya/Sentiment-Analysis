# code 
import streamlit as st
import pickle
import pandas as pd

def load_model():
    with open('model/vectoriser-ngram-(1,2).pickle', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('model/Sentiment-LR.pickle', 'rb') as file:
        LR = pickle.load(file)
    return vectorizer, LR

def predict(vectorizer, model, text):
    text_data = vectorizer.transform(text)
    sentiment = model.predict(text_data)
    df1 = pd.DataFrame({'text': text, 'sentiment': sentiment})
    df1 = df1.replace([0, 1, 4], ["Negative", "Positive", "Neutral"])
    return df1

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text to analyze sentiment:")

vectorizer, LR = load_model()
user_input = st.text_area("Enter text (one sentence per line)")

if st.button("Predict Sentiment"):
    if user_input.strip():
        text_list = user_input.split('\n')
        df_result = predict(vectorizer, LR, text_list)
        st.write(df_result)
    else:
        st.warning("Please enter some text.")
