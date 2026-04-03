# 🎯 ReviewSense — AI Sentiment Analysis for Product Reviews

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 🔗 Live Demo
👉 **[Try it here → reviewsense-im.streamlit.app](https://reviewsense-im.streamlit.app/)**

---

## 📌 What is this?

ReviewSense is a web app that uses a real transformer model (DistilBERT) to analyse product reviews and instantly tell you if they are positive or negative — with a confidence score.

I built this project to apply what I learned in my AI degree practically. The goal was to go from raw text input to a deployed, usable AI product.

---

## 🎬 What it does

**Single Review Analysis**
- Paste any product review
- Get instant sentiment prediction (Positive / Negative)
- See confidence score with a visual bar chart
- Word count and character counter included

**Batch Analysis**
- Upload a CSV file with hundreds of reviews
- App analyses all of them automatically with a live progress bar
- See summary stats — total, positive count, negative count, average confidence
- Visual breakdown with pie chart and bar chart
- Download full results as CSV

---

## 🧠 How it works
```
User inputs text
      ↓
Text gets tokenized (broken into pieces the model understands)
      ↓
DistilBERT reads the tokens and predicts sentiment
      ↓
Returns label (POSITIVE/NEGATIVE) + confidence score
      ↓
Streamlit displays result with charts
```

**Model used:** `distilbert-base-uncased-finetuned-sst-2-english`
- Made by HuggingFace
- Fine-tuned on SST-2 dataset (Stanford Sentiment Treebank)
- Lighter and faster version of BERT by Google
- 512 token input limit — reviews longer than that get sliced

**Known limitation:** DistilBERT tends to be overconfident (scores near 99% or 1%). Version 2 will swap to RoBERTa for more realistic confidence distribution.

---

## 🛠️ Tech stack

| Tool | What I used it for |
|------|-------------------|
| Python | Main language |
| HuggingFace Transformers | Loading and running DistilBERT model |
| Streamlit | Building the web UI |
| PyTorch | Backend engine that runs the model |
| Plotly | Interactive charts |
| Pandas | Reading and processing CSV files |
| Git + GitHub | Version control |
| Streamlit Cloud | Free deployment |

---

## 🚀 Run it locally
```bash
# Clone the repo
git clone https://github.com/studtannta9166/reviewsense.git
cd reviewsense

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

First run will download the DistilBERT model (~250MB). After that it's cached and instant.

---

## 📁 Project structure
```
reviewsense/
│
├── app.py              # Main application — all the Streamlit UI and model logic
├── requirements.txt    # All libraries needed to run the project
└── README.md           # You are here
```

---

## 📈 What I learned building this

- How transformer models work in practice (tokenization, inference, confidence scores)
- Why model caching matters for user experience
- How to handle real-world data issues (empty rows, encoding, token limits)
- Deploying a machine learning app for free with Streamlit Cloud
- The difference between model accuracy and model confidence calibration

---

## 🔮 Version 2 plans

- [ ] Swap DistilBERT for RoBERTa — better confidence calibration
- [ ] Add emotion detection (not just positive/negative but happy, angry, sad)
- [ ] Add support for non-English reviews
- [ ] Add review comparison feature

---

## 👩‍💻 About me

**Tannu** — 4th semester B.Sc Applied Artificial Intelligence student at TH Rosenheim, Germany.

Currently looking for an AI internship for 2026.

📧 tannu.tannu@stud.th-rosenheim.de
🔗 [GitHub](https://github.com/studtannta9166)
🎓 [TH Rosenheim](https://www.th-rosenheim.de/en/technology/computer-science-mathematics/applied-artificial-intelligence-bachelors-degree/)
