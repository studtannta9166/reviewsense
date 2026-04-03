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
    st.markdown("### Analyse multiple reviews at once")
    st.markdown("Upload a CSV file with a column of reviews — we will analyse all of them")

    # showing the user what format we expect
    # learned that good UX means never making user guess the input format
    st.markdown("**Your CSV should look like this:**")
    
    # creating a small demo table so user knows exactly what to upload
    example_df = pd.DataFrame({
        "review": [
            "This product is amazing!",
            "Terrible quality, broke in a day",
            "Average product, nothing special"
        ]
    })
    
    st.dataframe(example_df)

    # file uploader — restricted to CSV only so we dont get unexpected formats
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # everything below only runs once a file is actually uploaded
    # if no file yet, uploaded_file is None so we skip all of this
    if uploaded_file is not None:

        # pandas reads the CSV and turns it into a dataframe
        # basically an excel table we can work with in python
        df = pd.read_csv(uploaded_file)

        # showing first 5 rows so user can confirm it loaded correctly
        st.markdown("**Preview of your uploaded file:**")
        st.dataframe(df.head())

        # showing column names so user knows what we detected
        st.markdown(f"**Columns found:** {list(df.columns)}")

        # dropdown to pick which column has the reviews
        # made it flexible — column doesnt have to be named "review"
        # works with any CSV structure the user has
        review_column = st.selectbox(
            "Which column contains the reviews?",
            options=df.columns
        )

        if st.button("Analyse All Reviews →", type="primary"):

            # empty list — we fill this up as we analyse each review
            results = []

            # progress bar so user knows something is happening
            # especially important for large files — could take a while
            progress_bar = st.progress(0)

            # this text updates live — shows which review we are on
            status_text = st.empty()

            total = len(df)

            # looping through every review in the selected column
            # enumerate gives us i (row number) and review (the actual text)
            for i, review in enumerate(df[review_column]):

                status_text.text(f"Analysing review {i+1} of {total}...")

                # skipping empty rows — real datasets always have some
                # pd.isna catches null/NaN values that pandas uses for empty cells
                if pd.isna(review) or str(review).strip() == "":
                    continue

                # running the review through distilbert
                # str() converts to string just in case column has numbers
                # [:512] keeps us within the model token limit
                result = classifier(str(review)[:512])[0]

                # storing original review + model output together
                # so we can show and download everything later
                results.append({
                    "review": review,
                    "sentiment": result["label"],
                    "confidence": round(result["score"] * 100, 1)
                })

                # updating progress bar after each review
                # (i+1)/total gives a decimal between 0 and 1 which is what progress bar expects
                progress_bar.progress((i + 1) / total)

            status_text.text("Analysis complete! ✅")

            # converting our list of dictionaries into a proper dataframe
            # so we can display it as a table and also export it
            results_df = pd.DataFrame(results)

            st.markdown("---")
            st.markdown("### Results")
            st.dataframe(results_df)

            # --- SUMMARY STATS ---
            st.markdown("### Summary")

            # counting positives and negatives for summary cards and charts
            positive_count = (results_df["sentiment"] == "POSITIVE").sum()
            negative_count = (results_df["sentiment"] == "NEGATIVE").sum()

            # four metric cards — added average confidence as extra insight
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total reviewed", len(results_df))
            with col2:
                st.metric("Positive 😊", positive_count)
            with col3:
                st.metric("Negative 😞", negative_count)
            with col4:
                # mean() calculates average of the confidence column
                avg_conf = round(results_df["confidence"].mean(), 1)
                st.metric("Avg confidence", f"{avg_conf}%")

            st.markdown("---")
            st.markdown("### Visual breakdown")

            # two charts side by side — pie on left, bar on right
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Positive vs Negative split**")

                # pie chart showing the ratio of positive to negative
                # values are the counts, labels are the category names
                pie_fig = go.Figure(go.Pie(
                    labels=["Positive", "Negative"],
                    values=[positive_count, negative_count],
                    # green for positive, red for negative — consistent with rest of app
                    marker_colors=["#22c55e", "#ef4444"],
                    # hole makes it a donut chart — looks more modern
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

                # bar chart showing confidence score of every review
                # x axis is review number, y axis is confidence percentage
                # colour coded — green if positive, red if negative
                bar_colours = [
                    "#22c55e" if s == "POSITIVE" else "#ef4444"
                    for s in results_df["sentiment"]
                ]

                bar_fig = go.Figure(go.Bar(
                    # review number on x axis — just 1,2,3,4...
                    x=list(range(1, len(results_df) + 1)),
                    y=results_df["confidence"],
                    marker_color=bar_colours,
                    # showing confidence value on top of each bar
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

            # download button at the bottom — after user has seen all results
            csv_download = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_download,
                file_name="reviewsense_results.csv",
                mime="text/csv"
            )

            # counting positives and negatives separately for the summary cards
            # == "POSITIVE" creates a True/False series, .sum() counts the Trues
            positive_count = (results_df["sentiment"] == "POSITIVE").sum()
            negative_count = (results_df["sentiment"] == "NEGATIVE").sum()

            # three metric cards side by side — cleaner than just writing numbers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total reviewed", len(results_df))
            with col2:
                st.metric("Positive 😊", positive_count)
            with col3:
                st.metric("Negative 😞", negative_count)

            # --- DOWNLOAD ---
            # converting results back to CSV so user can open in Excel
            # index=False removes the row numbers from the download
            csv_download = results_df.to_csv(index=False).encode("utf-8")

            # download button — one click and they get the full results file
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_download,
                file_name="reviewsense_results.csv",
                mime="text/csv"
            )