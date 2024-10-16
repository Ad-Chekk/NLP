import streamlit as st
from transformers import pipeline

# Set up the Streamlit app
st.title("Text Summarizer and Sentiment Analyzer")

# Text input field
text_input = st.text_area("Enter the text you want to process", height=300)

# Load summarizer and sentiment analyzer models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# Button to trigger summarization
if st.button("Summarize"):
    if text_input:
        summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if text_input:
        sentiment = sentiment_analyzer(text_input)[0]
        st.subheader("Sentiment Analysis:")
        st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}")
    else:
        st.write("Please enter some text for sentiment analysis.")
