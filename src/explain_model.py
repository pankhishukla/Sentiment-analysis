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

vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl") #This loads the trained tfidf vectorizer from disk
model = joblib.load("artifacts/sentiment_model.pkl") #Loads the trained LR model

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

print("Top Positive Words:") #Printing the top positive words
print(top_positive)

print("\nTop Negative Words:") #Printing the top negative words
print(top_negative)


