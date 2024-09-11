import streamlit as st
import pickle
st.title("Twitter Sentiment Analysis")

model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

title = st.text_input("Twitter Text: ", "")


def predict_sentiment(text):
    # Transform the input text using the fitted vectorizer
    text_transformed = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_transformed)

    # Return sentiment based on prediction
    return "Positive Tweet" if prediction[0] == 1 else "Negative Tweet"


# Run prediction and display result when input is given
if title:
    sentiment = predict_sentiment(title)
    st.write("The Sentiment of the tweet is: ", sentiment)
