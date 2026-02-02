An end-to-end sentiment analysis system that predicts whether a movie review is positive or negative, along with a confidence score and word-level explanations for each prediction. This project focuses on deployability, explainability, and system design, not just model accuracy.

1. What This System Does?
- Takes an English text review as input
- Predicts sentiment: positive or negative
- Returns a confidence score for the prediction
- Detects the tone of the review (Positive Emotion, Negative Emotion, Mixed Emotion, or Neutral)
- Explains predictions by highlighting the most influential words
- Logs predictions with ground-truth labels for evaluation and error analysis

The system is designed for learning, experimentation, and low-stakes sentiment analysis use cases.

2. How the System Works??
- Text Preprocessing
Input text is cleaned using a fixed preprocessing pipeline:

- Lowercasing
> Removal of HTML tags and URLs
> Removal of non-alphabetic characters
> Whitespace normalization

The same preprocessing logic is used during training, inference, and explanation to avoid training–serving skew.

3. Feature Extraction

Text is converted into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency):
> Each review is represented as a sparse vector
> Each feature corresponds to a word in a fixed vocabulary
> TF-IDF captures how important a word is relative to the dataset

4. Model

A Logistic Regression classifier is used for binary sentiment classification

This model was chosen because it is:
> Fast and lightweight
> Easy to deploy
> Inherently interpretable
> Each word has a learned weight that contributes positively or negatively to the final prediction.

5. Explainability

For each prediction, the system provides local explainability:
> Only words present in the input review are considered
> Each word’s contribution is calculated as:
    contribution = TF-IDF value × model coefficient
> The top contributing words are returned as explanation factors
This allows users to understand why a particular sentiment was predicted.

6. Tone Detection

The system also detects the emotional tone of a review based on the strength of positive and negative sentiment contributions:
> **Positive Emotion**: Strong positive sentiment (strength > 0.7)
> **Negative Emotion**: Strong negative sentiment (strength > 0.7)
> **Mixed Emotion**: Balanced positive and negative sentiments (conflict ratio > 0.4)
> **Neutral**: Weak or balanced sentiments that don't fit other categories

This provides a more nuanced understanding of the emotional content beyond binary positive/negative classification.

An evaluation endpoint logs predictions along with known ground-truth labels:
> Timestamp
> Input review
> True label
> Predicted sentiment
> Confidence score

These logs can be analyzed later to study errors, bias, and confidence calibration.

7. Limitations

- The model uses a bag-of-words representation and does not understand context or word order
- Sarcasm, irony, and mixed sentiments may be misclassified
- Explanations reflect statistical correlations, not true human reasoning
- Confidence scores indicate model certainty, not correctness
- Performance depends on the domain and quality of training data
- Tone detection relies on sentiment strength thresholds and may not capture all emotional nuances

8. Open 2 terminals:

run 2 commands separately on each terminal
- uvicorn api:app --reload 
- streamlit run ui.py


9. Datasets used

The link is as given below
IMDB Dataset from Kaggle
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews