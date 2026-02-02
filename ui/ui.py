#ui.py

import streamlit as st
import requests

st.set_page_config(page_title = "Sentiment Analyzer", layout = "centered")

st.title("Sentiment Analysis")
st.write("Enter text below to analyze")


review = st.text_area("Your text", height = 150)

word_count = len(review.split())

if word_count < 3:
    st.warning("This input is very short. The prediction can be very unrealiable because of this")

if st.button("Analyze"):
    if review.strip() == "":
        st.error("Please enter some text")

    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json = {"review": review}
        )

        result = response.json()
        st.write(result)

        if not isinstance(result, dict):
            st.error("Unexpected response from server.")

        elif "error" in result:
            st.error(result["error"])

        elif "sentiment" in result and "confidence" in result:
            st.success(f"Sentiment: {result['sentiment']}")
            # st.info(f"Confidence: {result['confidence']}")

            if "tone" in result:
                st.info(f"Tone: {result['tone']}")

            confidence = result["confidence"]
            if confidence <= 0.6:
                st.warning(f"Low confidence ({confidence}). The model seems to be uncertain")

            elif confidence <= 0.8:
                st.warning(f"Moderate confidence ({confidence})")

            else:
                st.warning(f"Highest confidence ({confidence}). The model is pretty certain about its predictions")

        else:
            st.error("Malformed response from server.")
            st.write(result)

st.caption(
    "This model may struggle with mixed emotions, very short answers and sarcastic comments"
)