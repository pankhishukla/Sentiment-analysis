import re

def clean_review(text: str) -> str: #This means that the function expects a string and the -> means that     output is a string
    text = text.lower() #this converts the text to lowercase
    text = re.sub(r'<.*?>',' ', text) #removing the tags
    text = re.sub(r'http\S+', ' ', text)   #removing the URLs
    text = re.sub(r'[^a-z\s]', ' ', text) #keeping letters only
    text = re.sub(r'\s+', ' ', text) #removing extra spaces
    return text.strip() #removing leading and trailing spaces