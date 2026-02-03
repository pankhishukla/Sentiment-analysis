# ui/ui.py

import sys
import os

# ---- Fix Python path so root files are importable ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import joblib
from utils import clean_review


# ---- Load model artifacts ----
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
model = joblib.load("artifacts/sentiment_model.pkl")


# ---- Prediction function ----
def predict_sentiment(text: str):
    cleaned = clean_review(text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    negative_prob, positive_prob = probs

    sentiment = "positive" if positive_prob >= 0.5 else "negative"
    confidence = max(positive_prob, negative_prob)

    return sentiment, round(confidence, 3)


# ---- Explainability function ----
def explain_prediction(text, top_k=5):
    cleaned = clean_review(text)
    vec = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    nonzero_idx = vec.nonzero()[1]

    contributions = []
    for idx in nonzero_idx:
        word = feature_names[idx]
        impact = coefficients[idx] * vec[0, idx]
        contributions.append((word, impact))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_k]


# ---- Streamlit UI ----
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("Sentiment Analysis System")
st.write("Analyze the emotional tone of a review using an explainable AI model.")

review = st.text_area(
    "Enter a review",
    height=150,
    placeholder="Example: The acting was amazing, but the story felt slow and predictable."
)

if review and len(review.split()) < 3:
    st.warning("This input is very short. Predictions may be unreliable.")

# initialize to avoid NameError
top_factors = []

if st.button("Analyze"):
    if not review.strip():
        st.error("Please enter some text.")
        st.stop()

    sentiment, confidence = predict_sentiment(review)
    top_factors = explain_prediction(review)

    st.subheader("Prediction")

    if sentiment == "positive":
        st.success("😊 Positive sentiment")
    else:
        st.error("😞 Negative sentiment")

    st.markdown("**Model confidence**")
    st.progress(min(confidence, 1.0))

    if confidence < 0.6:
        st.caption("Low confidence — model is uncertain")
    elif confidence < 0.8:
        st.caption("Moderate confidence")
    else:
        st.caption("High confidence")

    if top_factors:
        st.subheader("Why this prediction?")
        for word, impact in top_factors:
            direction = "positive" if impact > 0 else "negative"
            st.write(
                f"• **{word}** pushed the prediction toward **{direction}** sentiment"
            )

    with st.expander("⚠️ Model limitations"):
        st.write(
            "This model may struggle with sarcasm, mixed emotions, "
            "very short reviews, or domain-specific language."
        )
