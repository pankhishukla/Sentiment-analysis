#analyze_log.py

import pandas as pd #used for data manipulation, analysis and cleaning of structured data

df = pd.read_csv("evaluation_log.csv") #This reads the csv file. The csv file has the log of all the evaluations we did using the /evaluate endpoint
# print(df.head())

#This creates a new column 'correct' which checks if the prediction model was correct for a particular row
df['correct'] = (
    (df["predicted_sentiment"] == "positive") & ["true_label"] == 1 #predicted positive AND true label is 1
) | (
    (df["predicted_sentiment"] == "negative") & ["true_label"] == 0 #predicted negative AND true label is 0
)

print(df["correct"].value_counts()) #This counts how many True and False values are present in the 'correct' column

mistakes = df[df["correct"] == False] #This filters the rows where the model made a mistake , basically all the rows where correct is false

print(mistakes[["review", "true_label", "predicted_sentiment", "confidence"]]) #this prints the reviews where the model made a mistake with the true label, predicted sentiment and confidence 

