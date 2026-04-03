import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
import pandas as pd  # pandas reads CSV files and turns them into tables we can work with

# Setting up the page — title, icon, and centered layout for professional look
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

# --- SIDEBAR --- (added in commit 6)
# sidebar keeps extra info out of the main view — cleaner experience
with st.sidebar:
    st.markdown("##ReviewSense")
    st.markdown("**Version 2.0**")
    st.markdown("---")

    # what this app does in simple words
    st.markdown("### About")
    st.markdown(
        "ReviewSense uses **DistilBERT** — a transformer model "
        "trained on thousands of product reviews — to instantly "
        "classify sentiment and confidence."
    )

    st.markdown("---")

    # showing the tech stack — good for transparency
    st.markdown("### Built with")
    st.markdown("HuggingFace Transformers")
    st.markdown("Streamlit")
    st.markdown("PyTorch")
    st.markdown("Plotly")
    st.markdown("Pandas")

    st.markdown("---")

    # personal section — my name, university, github
    st.markdown("### Developer")
    st.markdown("Made by **Tannu**")
    st.markdown("🎓 B.Sc Applied AI — TH Rosenheim")
    st.markdown("[GitHub](https://github.com/studtannta9166)")

# --- HEADER ---
st.title("🎯 ReviewSense")
st.markdown("**AI-powered sentiment analysis for product reviews**")
st.markdown("Built with HuggingFace Transformers + DistilBERT")

# tabs separate the two main features — keeps app organised
tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])

# ============================================================
# TAB 1 — SINGLE REVIEW
# ============================================================
with tab1:
    st.markdown("### Analyse a single review")
    st.markdown("Paste any product review below and the AI will tell you if it's positive or negative")
    
    # text area for user input
    user_input = st.text_area(
        "Your review:",
        placeholder="e.g. This product is absolutely amazing, works perfectly!",
        height=150
    )

    # character counter — small UX detail so user knows the 512 limit
    char_count = len(user_input)
    st.caption(f"Characters: {char_count} / 512 model limit")
    
    if st.button("Analyse Sentiment →", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # spinner shows while model is thinking — without this app just freezes
            with st.spinner("Reading your review..."):
                # model has a 512 token limit so we slice the input
                # most reviews are under 512 tokens anyway
                result = classifier(user_input[:512])[0]
                label = result["label"]
                score = result["score"]
                confidence = round(score * 100, 1)

            # decide colours and emoji based on what model returned
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

            st.markdown("---")
            st.markdown(f"## {emoji} {message}")

            # three metric cards side by side — clean and professional
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Result", label)
            with col2:
                st.metric("Confidence", f"{confidence}%")
            with col3:
                # .split() breaks text into words, len() counts them
                st.metric("Word count", len(user_input.split()))

            # horizontal confidence bar — easier to read than just a number
            fig = go.Figure(go.Bar(
                x=[confidence],
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

            # plain english explanation of what the model found
            st.info(f"💡 {explanation}")

            # collapsible section — shows raw model output for curious users
            with st.expander("🔍 See raw model output"):
                st.json(result)

# ============================================================
# TAB 2 — BATCH ANALYSIS
# ============================================================
with tab2:
    st.markdown("### Analyse multiple reviews at once")
    st.markdown("Upload a CSV file with a column of reviews — we will analyse all of them")

    # showing expected format so user never has to guess
    st.markdown("**Your CSV should look like this:**")
    
    # small demo table — makes it crystal clear what to upload
    example_df = pd.DataFrame({
        "review": [
            "This product is amazing!",
            "Terrible quality, broke in a day",
            "Average product, nothing special"
        ]
    })
    
    st.dataframe(example_df)

    # letting user download the sample CSV directly (added in commit 6)
    # so they dont have to manually create a test file
    sample_csv = example_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download sample CSV",
        data=sample_csv,
        file_name="sample_reviews.csv",
        mime="text/csv"
    )

    # file uploader — CSV only so we dont get unexpected file types
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # everything below only runs once a file is actually uploaded
    if uploaded_file is not None:

        # pandas reads CSV and turns it into a table we can loop through
        df = pd.read_csv(uploaded_file)

        # showing first 5 rows so user can confirm it loaded correctly
        st.markdown("**Preview of your uploaded file:**")
        st.dataframe(df.head())

        st.markdown(f"**Columns found:** {list(df.columns)}")

        # dropdown to select review column
        # flexible — works even if column is not named "review"
        review_column = st.selectbox(
            "Which column contains the reviews?",
            options=df.columns
        )

        if st.button("Analyse All Reviews →", type="primary"):

            # empty list we fill up as we go through each review
            results = []

            # progress bar — important for large files so user isnt staring at blank screen
            progress_bar = st.progress(0)

            # updates live to show which review number we are on
            status_text = st.empty()

            total = len(df)

            # looping through every row in the selected column
            # enumerate gives us i (the row number) and review (the text)
            for i, review in enumerate(df[review_column]):

                status_text.text(f"Analysing review {i+1} of {total}...")

                # skipping empty rows — real world datasets always have some
                if pd.isna(review) or str(review).strip() == "":
                    continue

                # running each review through distilbert
                result = classifier(str(review)[:512])[0]

                # storing review + result together as one dictionary
                results.append({
                    "review": review,
                    "sentiment": result["label"],
                    "confidence": round(result["score"] * 100, 1)
                })

                # (i+1)/total gives 0 to 1 decimal — what progress bar needs
                progress_bar.progress((i + 1) / total)

            status_text.text("Analysis complete! ✅")

            # turning list of dictionaries into a proper dataframe table
            results_df = pd.DataFrame(results)

            st.markdown("---")
            st.markdown("### Results")
            st.dataframe(results_df)

            st.markdown("### Summary")

            # counting positives and negatives
            # == "POSITIVE" gives True/False for each row, .sum() counts Trues
            positive_count = (results_df["sentiment"] == "POSITIVE").sum()
            negative_count = (results_df["sentiment"] == "NEGATIVE").sum()

            # four summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total reviewed", len(results_df))
            with col2:
                st.metric("Positive 😊", positive_count)
            with col3:
                st.metric("Negative 😞", negative_count)
            with col4:
                # mean() gets the average confidence across all reviews
                avg_conf = round(results_df["confidence"].mean(), 1)
                st.metric("Avg confidence", f"{avg_conf}%")

            st.markdown("---")
            st.markdown("### Visual breakdown")

            # two charts sitting side by side
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Positive vs Negative split**")

                # donut chart — hole=0.4 makes it look modern not old school
                pie_fig = go.Figure(go.Pie(
                    labels=["Positive", "Negative"],
                    values=[positive_count, negative_count],
                    marker_colors=["#22c55e", "#ef4444"],
                    hole=0.4
                ))
                pie_fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            with chart_col2:
                st.markdown("**Confidence scores per review**")

                # colour each bar green or red depending on sentiment
                # list comprehension — short way to build a list with a condition
                bar_colours = [
                    "#22c55e" if s == "POSITIVE" else "#ef4444"
                    for s in results_df["sentiment"]
                ]

                bar_fig = go.Figure(go.Bar(
                    x=list(range(1, len(results_df) + 1)),
                    y=results_df["confidence"],
                    marker_color=bar_colours,
                    text=results_df["confidence"],
                    textposition="outside"
                ))
                bar_fig.update_layout(
                    height=300,
                    xaxis_title="Review number",
                    yaxis_title="Confidence %",
                    yaxis=dict(range=[0, 110]),
                    margin=dict(l=0, r=0, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("---")

            # download button — user gets full results as CSV they can open in Excel
            csv_download = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_download,
                file_name="reviewsense_results.csv",
                mime="text/csv"
            )