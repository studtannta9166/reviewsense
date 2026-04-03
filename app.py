import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

# Setting up the page — title, icon, and wide layout (to give professional look)
st.set_page_config(
    page_title="ReviewSense",
    page_icon="🎯",
    layout="centered"
)

# using st.cache_resource so the model only loads once
# without this it would reload every time someone clicks a button (makes it very slow)
@st.cache_resource
def load_model():
    # distilbert is a lighter version of BERT, faster but still accurate
    # it was fine-tuned on SST-2 dataset which is movie/product reviews
    return pipeline("sentiment-analysis", 
                   model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

# Header
st.title("🎯 ReviewSense")
st.markdown("**AI-powered sentiment analysis for product reviews**")
st.markdown("Built with HuggingFace Transformers + DistilBERT")


# Tabs make it cleaner — single review vs batch analysis are two different use cases
tab1, tab2 = st.tabs(["📝 Single Review", "📊 Batch Analysis"])

with tab1:
    st.markdown("### Analyse a single review")
    st.markdown("Paste any product review below and the AI will tell you if it's positive or negative")
    
    # text area for user input
    user_input = st.text_area(
        "Your review:",
        placeholder="e.g. This product is absolutely amazing, works perfectly!",
        height=150
    )
    
    if st.button("Analyse Sentiment →", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            st.info("Analysing... our team(AI model) is reading your review")

with tab2:
    st.markdown("### Coming soon — batch analysis")
    st.markdown("Upload a CSV of reviews and analyse them all at once")
