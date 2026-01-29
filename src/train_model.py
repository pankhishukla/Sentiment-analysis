#train_model.py

#This file will only train -> save -> Exit
import pandas as pd #for data manipulation
import re #for regular expressions
from sklearn.model_selection import train_test_split #for splitting the dataset
from sklearn.feature_extraction.text import TfidfVectorizer #for text vectorization
from sklearn.linear_model import LogisticRegression #for classification

import os #This lets python interact with OS
#It is just file and folder managment

import joblib #used to load the tf-idf model vectorizer and LR model efficiently
#This saves the model to the disk so that we can reuse it later

from utils import clean_review #importing the clean_review function from utils

df = pd.read_csv('data/IMDB Dataset.csv') #loading the dataset
pd.set_option('display.max_colwidth', None) #to display the full review text in the terminal


df['clean_review'] = df['review'].apply(clean_review) #applying the cleaning function to the review column
# print(df['review'].iloc[0]) #printing the first review before cleaning
# print(df['clean_review'].iloc[0]) #printing the first review after cleaning

df['label'] = df['sentiment'].map({ 
    'positive' : 1, #mapping sentiments to binary labels
    'negative' : 0 
}).astype(int) #converting the labels to integers

x = df['clean_review'] #feature
y = df['label'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y , #splitting the dataset into training and testing sets
    test_size = 0.2, #20% for testing
    random_state = 42 
)

df.to_csv("cleaned_imdb.csv", index=False) #all the cleaned reviews will be stored in a cleaned imdb csv file 

# df.head() #displays the firt few rows of the dataframe

vectorizer = TfidfVectorizer(
    max_features = 2000, #Limits the vocabulary size to 3000 most important words
    ngram_range = (1, 1) #Can take up 1 word (good) and 2 word(not-good) phrases for reviews #Currently shifting it to unigrams
)

x_train_tfidf = vectorizer.fit_transform(x_train) #learns vocabulary from the training data set and then converts it to tfidf
x_test_tfidf = vectorizer.transform(x_test) #converts the test text from the same learnt vocabulary

#Making predictions
model = LogisticRegression(max_iter=1000) #The whole LR model 
model.fit(x_train_tfidf, y_train) #trains the data on the vectorized training data
y_pred = model.predict(x_test_tfidf) #predicts the sentiment labels for the reviews

# print(model) #prints the model configuration

os.makedirs("artifacts", exist_ok = True) #This creats a folder named artifacts if is doesnt exist, exist_ok=true means if it doesnt exist, then create it, if it does exist, then do nothing

joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.pkl") #This takes the trained tfidf model and saves it to this file
joblib.dump(model, "artifacts/sentiment_model.pkl") #Thsi takes the lr-model and saves it to this file 

print("Model and vectorizer saved!")