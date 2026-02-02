#api.py
#http://127.0.0.1:8000/docs

from fastapi import FastAPI
import joblib

from fastapi import Body

from datetime import datetime
import csv
import os

from pydantic import BaseModel

from .utils import clean_review

app = FastAPI()

vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
model = joblib.load("artifacts/sentiment_model.pkl")

class ReviewRequest(BaseModel):
    review: str

def calculate_tone(text: str): #Function to calculate the tone based on word contributions
    cleaned = clean_review(text)
    vec = vectorizer.transform([cleaned])
    nonzero_idx = vec.nonzero()[1]
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    positive_contributions = []
    negative_contributions = []
    
    for idx in nonzero_idx:
        contribution = coefficients[idx] * vec[0, idx]
        
        if contribution > 0:
            positive_contributions.append(contribution)
        else:
            negative_contributions.append(abs(contribution))
    
    positive_strength = sum(positive_contributions)
    negative_strength = sum(negative_contributions)
    
    if positive_strength == 0 and negative_strength == 0:
        return "Neutral"
    
    conflict_ratio = min(positive_strength, negative_strength) / max(positive_strength, negative_strength)
    
    if conflict_ratio > 0.4:
        tone = "Mixed Emotion"
    elif positive_strength > 0.7:
        tone = "Positive Emotion"
    elif negative_strength > 0.7:
        tone = "Negative Emotion"
    else:
        tone = "Neutral"
    
    return tone

def predict_sentiment(text: str): #This function mandatorily takes the input as text and in string datatype it is type hint, not a rule
    if not isinstance(text, str) or text.strip() == "": #Checks if the input is actually text or not, without this the API throws an error
        return{"error" : "Invalid Input"} #Stops the model early
    
    cleaned = clean_review(text) #This passes the text through the clean_Review text and saves it to a variable named cleaned
    vec = vectorizer.transform([cleaned]) #This converts the text to numbers 
    # prob = model.predict_proba(vec)[0][1] #model.predict_proba(vec) means that it will return soemthing like [[0.18, 0.82]] and it says like it is 18% confident that it is negative and 82% confident that it is positive because [0][1] is [negative][positive]

    # return{
    #     "sentiment" : "positive" if prob >= 0.5 else "negative",
    #     "confidence" : round(prob,3) #Rounding it up to 3 places
    # }

    probs = model.predict_proba(vec)[0]
    positive_prob = probs[1]
    negative_prob = probs[0]

    sentiment = "positive" if positive_prob > 0.5 else "negative"
    confidence = max(positive_prob, negative_prob)
    tone = calculate_tone(text)

    return {
        "sentiment" : sentiment,
        "confidence" : round(confidence, 3),
        "tone" : tone
    }

def explain_prediction(text: str): #explaining one prediction
    cleaned = clean_review(text) #cleans the input text
    vec = vectorizer.transform([cleaned]) #converts the cleaned text in a TF-IDF vector

    feature_names = vectorizer.get_feature_names_out() #gets the list of words corresponding to tf-idf columns
    coefficients = model.coef_[0] #attributes ending with _ are learned during the training,[0] extracting the first and only row

    nonzero_idx = vec.nonzero()[1] #finding the indices of words that actually appear in the review. We need this because we just want to explain the words that were present

    contributions = [] #Creating a list to store the word impacts

    for idx in nonzero_idx: #this loops through each word present in the review
        word = feature_names[idx] #The actual word
        weight = coefficients[idx] #The model's opinion about the word
        value = vec[0, idx] #how strongly does the word appear (tf-idf)
        contributions.append((word, weight * value)) #This is contributions = importance x presence

    contributions.sort(key=lambda x: abs(x[1]), reverse=True) #Sorts out by the impact strength

    return contributions[:5] #This returns the top 5 most influential words



#The below given endpoint is just doing a simple job that is making a prediction and then logging what actually happened.
#What this /evaluate does is,
#1. It runs the model
#2. It compares prediction with the ground truth
#3. It stores the result
#4. It returns the prediction anyways
@app.post("/evaluate") #This creates a post API endpoint at /evaluate
def evaluate(review: str, true_label: int): #Bascially writing the logic for this is what the user should have answered
    prediction = predict_sentiment(review) #This calls our existing inference function

    if "error" in prediction:
        return prediction
    
    log_exists = os.path.exists("evaluation_log.csv") #Checking if the log file exists or not

    with open("evaluation_log.csv", mode = "a", newline = "", encoding = "utf-8") as f: #mode a is appending mode (not overwriting the older data), newline="" this avoids the blank lines in the csv,  the keyword with means that open the file, do you stuff and close the file safely.
        writer = csv.writer(f) 

        if not log_exists:
            writer.writerow([ #The headings of the csv
                "timestamp",
                "review",
                "true_label",
                "predicted_sentiment",
                "confidence"
            ])

        #The logging happens here
        writer.writerow([
            datetime.now().isoformat(), #When the prediction happened
            review, #What text was used
            true_label, #What the truth was 
            prediction["sentiment"], #What the model predicted
            prediction["confidence"] #How confident it was
        ])

        return{
            "message": "Evaluation logged",
            "prediction": prediction
        }
    

@app.post("/predict") #registers a post api endpoint at /predict
def predict(data: ReviewRequest): #This automatically validates the input
    prediction = predict_sentiment(data.review) #Core prediction, and this runs the inference on the input text
    explanation = explain_prediction(data.review) #This computes why the model made the prediction, basically the reason for the answer

    if "error" in prediction:
        return prediction
    
    return {
        **prediction, #** is the dictionary unpacking operator, this takes up all the key-value pairs from the dictionary and spread them into a new dictionary
        "top_factors": explanation #add the explaination data
    }







    


