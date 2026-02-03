# # ui.py

# import streamlit as st
# # import requests
# import os

# import joblib
# from src.utils import clean_review   # adjust if utils.py is elsewhere

# vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
# model = joblib.load("artifacts/sentiment_model.pkl")


# API_URL = os.getenv(
#     "API_URL",
#     "http://127.0.0.1:8000/predict"
# )


# st.set_page_config(
#     page_title="Sentiment Analyzer",
#     layout="centered"
# )

# st.title("Sentiment Analysis System")
# st.write(
#     "Analyze the emotional tone of a review using an explainable AI model."
# )

# # -------- Input --------
# review = st.text_area(
#     "Enter a review",
#     height=150,
#     placeholder="Example: The acting was amazing, but the story felt slow and predictable."
# )

# word_count = len(review.split())
# if review and word_count < 3:
#     st.warning(
#         "This input is very short. Predictions may be unreliable."
#     )

# # -------- Analyze Button --------
# if st.button("Analyze"):
#     if not review.strip():
#         st.error("Please enter some text to analyze.")
#         st.stop()

#     # -------- Call API --------
#     try:
#         response = requests.post(
#             API_URL,
#             json={"review": review},
#             timeout=5
#         )
#         result = response.json()
#     except requests.exceptions.ConnectionError:
#         st.error("Backend API is not running. Please start FastAPI.")
#         st.stop()
#     except Exception as e:
#         st.error(f"Unexpected error: {e}")
#         st.stop()

#     # -------- Handle API Errors --------
#     if not isinstance(result, dict):
#         st.error("Unexpected response from server.")
#         st.stop()

#     if "error" in result:
#         st.error(result["error"])
#         st.stop()

#     # -------- Sentiment Result --------
#     sentiment = result["sentiment"]
#     confidence = result["confidence"]

#     st.subheader("Prediction")

#     if sentiment == "positive":
#         st.success(f"Positive sentiment")
#     else:
#         st.error(f"Negative sentiment")

#     # -------- Confidence Visualization --------
#     st.markdown("**Model confidence**")
#     st.progress(min(confidence, 1.0))

#     if confidence < 0.6:
#         st.caption("Low confidence — model is uncertain")
#     elif confidence < 0.8:
#         st.caption("Moderate confidence")
#     else:
#         st.caption("High confidence")

#     # -------- Emotional Tone (if present) --------
#     if "tone" in result:
#         st.info(f"Emotional tone detected: **{result['tone']}**")

#     # -------- Explainability --------
#     if "top_factors" in result and result["top_factors"]:
#         st.subheader("Why this prediction?")

#         for word, impact in result["top_factors"]:
#             direction = "positive" if impact > 0 else "negative"
#             st.write(
#                 f"• **{word}** pushed the prediction toward **{direction}** sentiment"
#             )

#     # -------- Transparency --------
#     with st.expander("⚠️ Model limitations"):
#         st.write(
#             "This model may struggle with sarcasm, mixed emotions, "
#             "very short reviews, or domain-specific language. "
#             "Confidence reflects model certainty, not correctness."
#         )


# ui.py

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import joblib
from utils import clean_review

# Load model artifacts
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
model = joblib.load("artifacts/sentiment_model.pkl")


def explain_prediction(text, top_k=5):
    cleaned = clean_review(text)
    vec = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    nonzero_idx = vec.nonzero()[1]

    contributions = []

    for idx in nonzero_idx:
        word = feature_names[idx]
        tfidf_value = vec[0, idx]
        weight = coefficients[idx]

        impact = weight * tfidf_value
        contributions.append((word, impact))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return contributions[:top_k]


def predict_sentiment(text: str):
    cleaned = clean_review(text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    negative_prob, positive_prob = probs

    sentiment = "positive" if positive_prob >= 0.5 else "negative"
    confidence = max(positive_prob, negative_prob)

    return sentiment, round(confidence, 3)


st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered"
)

st.title("Sentiment Analysis System")
st.write("Analyze the emotional tone of a review using an explainable AI model.")

review = st.text_area(
    "Enter a review",
    height=150,
    placeholder="Example: The acting was amazing, but the story felt slow and predictable."
)

if review and len(review.split()) < 3:
    st.warning("This input is very short. Predictions may be unreliable.")

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

    with st.expander("⚠️ Model limitations"):
        st.write(
            "This model may struggle with sarcasm, mixed emotions, "
            "very short reviews, or domain-specific language."
        )

if top_factors:
    st.subheader("Why this prediction?")

    for word, impact in top_factors:
        direction = "positive" if impact > 0 else "negative"
        st.write(
            f"• **{word}** pushed the prediction toward **{direction}** sentiment"
        )



