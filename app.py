import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter Tweet")

if st.button("Predict"):

    tweet_vector = vectorizer.transform([tweet])

    prediction = model.predict(tweet_vector)

    if prediction[0] == 1:
        st.success("Positive Sentiment")
    else:
        st.error("Negative Sentiment")