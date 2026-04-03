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
tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])

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
            # show a spinner while model is thinking
            with st.spinner("Reading your review..."):
                # model has a 512 token limit so we slice the input
                # most reviews are under 512 tokens anyway
                result = classifier(user_input[:512])[0]
                label = result["label"]
                score = result["score"]
                confidence = round(score * 100, 1)

            # decide what to show based on result
            if label == "POSITIVE":
                emoji = "😊"
                colour = "#22c55e"
                message = "Positive review"
                explanation = "The model detected positive language — words suggesting satisfaction, happiness, or approval."
            else:
                emoji = "😞"
                colour = "#ef4444"
                message = "Negative review"
                explanation = "The model detected negative language — words suggesting dissatisfaction, frustration, or disappointment."

            # result card
            st.markdown("---")
            st.markdown(f"## {emoji} {message}")

            # three metric columns — looks clean and professional
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Result", label)
            with col2:
                st.metric("Confidence", f"{confidence}%")
            with col3:
                # word count is a nice extra detail to show
                st.metric("Word count", len(user_input.split()))

            # confidence bar — visual representation is easier to read than just a number
            fig = go.Figure(go.Bar(x=[confidence],
                y=["Confidence"],
                orientation="h",
                marker_color=colour,
                text=[f"{confidence}%"],
                textposition="inside"
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 100], title="Confidence %"),
                height=100,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # plain english explanation — makes the app feel transparent
            st.info(f"💡 {explanation}")

            # show raw output for people who want to see under the hood
            with st.expander("🔍 See raw model output"):
                st.json(result)
with tab2:
    st.markdown("### Batch Analysis")
    st.markdown("Upload a CSV of reviews and analyse them all at once")

