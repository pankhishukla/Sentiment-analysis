# ui.py

import streamlit as st
import requests
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)


st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered"
)

st.title("Sentiment Analysis System")
st.write(
    "Analyze the emotional tone of a review using an explainable AI model."
)

# -------- Input --------
review = st.text_area(
    "Enter a review",
    height=150,
    placeholder="Example: The acting was amazing, but the story felt slow and predictable."
)

word_count = len(review.split())
if review and word_count < 3:
    st.warning(
        "This input is very short. Predictions may be unreliable."
    )

# -------- Analyze Button --------
if st.button("Analyze"):
    if not review.strip():
        st.error("Please enter some text to analyze.")
        st.stop()

    # -------- Call API --------
    try:
        response = requests.post(
            API_URL,
            json={"review": review},
            timeout=5
        )
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend API is not running. Please start FastAPI.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    # -------- Handle API Errors --------
    if not isinstance(result, dict):
        st.error("Unexpected response from server.")
        st.stop()

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # -------- Sentiment Result --------
    sentiment = result["sentiment"]
    confidence = result["confidence"]

    st.subheader("Prediction")

    if sentiment == "positive":
        st.success(f"Positive sentiment")
    else:
        st.error(f"Negative sentiment")

    # -------- Confidence Visualization --------
    st.markdown("**Model confidence**")
    st.progress(min(confidence, 1.0))

    if confidence < 0.6:
        st.caption("Low confidence — model is uncertain")
    elif confidence < 0.8:
        st.caption("Moderate confidence")
    else:
        st.caption("High confidence")

    # -------- Emotional Tone (if present) --------
    if "tone" in result:
        st.info(f"Emotional tone detected: **{result['tone']}**")

    # -------- Explainability --------
    if "top_factors" in result and result["top_factors"]:
        st.subheader("Why this prediction?")

        for word, impact in result["top_factors"]:
            direction = "positive" if impact > 0 else "negative"
            st.write(
                f"• **{word}** pushed the prediction toward **{direction}** sentiment"
            )

    # -------- Transparency --------
    with st.expander("⚠️ Model limitations"):
        st.write(
            "This model may struggle with sarcasm, mixed emotions, "
            "very short reviews, or domain-specific language. "
            "Confidence reflects model certainty, not correctness."
        )
