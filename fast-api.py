from typing import Optional
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{user_text}")
def read_item(user_text: str):
    #Load our model again
    text = user_text
    new_text = [text]
    vectorizer = pickle.load(open("vectorizer","rb"))
    integers = vectorizer.transform(new_text)
    model = pickle.load(open("model_save","rb"))
    x = model.predict(integers)
    # print(x)
    if x == 1:
        print ("Message is SPAM")
        result = "Message is SPAM"
    else:
        print ("Message is NOT Spam")
        result = "Message is NOT Spam"
    return {"final pridiction": result}
