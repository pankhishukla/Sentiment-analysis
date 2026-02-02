# import joblib #lib to save and load python objects efficiently
# import numpy as np #Numerical computing lib
# import pandas as pd #tabular data handling

# vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl") #This loads the trained tfidf vectorizer from disk

# model = joblib.head("artifacts/sentiment_model.pkl") #Loads the trained LR model

# feature_names = vectorizer.get_feature_names_out() #Returns a list of all words used by the tf-idf vectorizer

# coefficients = model.coef_[0] #Extracts the weight vector from LR model
# #[0] because we take the first (and only) row

# top_positive_idx = np.argsort(coefficients)[-20:]

import joblib #lib to save and load python objects efficiently
import numpy as np #deals with numerical computign
import pandas as pd #deals with tabular data handling
from utils import clean_review

vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl") #This loads the trained tfidf vectorizer from disk
model = joblib.load("artifacts/sentiment_model.pkl") #Loads the trained LR model

review = input("Enter a review to explain: ") #Get review from user

cleaned = clean_review(review)
vec = vectorizer.transform([cleaned])
nonzero_idx = vec.nonzero()[1]

feature_names = vectorizer.get_feature_names_out() #Returns a list of all words used by the tf-idf vectorizer
coefficients = model.coef_[0] ##attributes ending with _ are learned during the training,[0] extracting the first and only row

top_positive_idx = np.argsort(coefficients)[-20:] #Getting the positions of the top 20 positive weights
top_negative_idx = np.argsort(coefficients)[:20] #Getting the positions of the top 20 negative weights

top_positive = pd.DataFrame({ #Creating a dataframe for better visualization
    "feature": feature_names[top_positive_idx], #This maps the indices to actual words
    "weight": coefficients[top_positive_idx] #This maps the indices to actual weights
}).sort_values(by="weight", ascending=False) #Sorting by weight for better visualization

top_negative = pd.DataFrame({ #Creating a dataframe for better visualization
    "feature": feature_names[top_negative_idx], #This maps the indices to actual words
    "weight": coefficients[top_negative_idx] #This maps the indices to actual weights
})

#The list will store how strongly the model is being pushed towards the positive/negative sentiment
positive_contributions = [] #This creates a separate list to store all the positive sentiment contributions
negative_contributions = [] #This creates a separate list to store all the negative sentiment contributions

for idx in nonzero_idx: #This loops around only the words which are present in the input review
    #word = feature_names[idx] #This maps the index back to the actual word

    contribution = coefficients[idx] * vec[0, idx] #Coefficients[idx] is what the model thinks about the word
    #vec[0, idx] is how strongly does the word appear in the review
    #contribution = opinion * presence

    if contribution > 0:
        positive_contributions.append(contribution)

    else:
        negative_contributions.append(abs(contribution)) #abs is used to eliminate the - sign

positive_strength = sum(positive_contributions)
negative_strength = sum(negative_contributions)

conflict_ratio = min(positive_strength, negative_strength) / max(positive_strength, negative_strength)

if conflict_ratio > 0.4:
    tone = "Mixed Emotion"

elif positive_strength > 0.7:
    tone = "Positive Emotion"

elif negative_strength > 0.7:
    tone = "Negative Emotion"

else:
    tone = "Neutral"

print("\nDetected Tone: ",tone)

print("\nTop Positive Words:") #Printing the top positive words
print(top_positive)

print("\nTop Negative Words:") #Printing the top negative words
print(top_negative)


