import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="ReviewSense", page_icon="🎯")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

st.title("🎯 ReviewSense")
st.write("Paste a review and get sentiment instantly")

user_input = st.text_area("Your review:")

if st.button("Analyse"):
    if user_input:
        result = classifier(user_input)[0]
        st.write(result)