import pickle
import string
import re
from fastapi import FastAPI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from schema import TextData, TextBatchData

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("svd.pkl", "rb") as file:
    svd = pickle.load(file)

app = FastAPI()



def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')
    text = "".join([i for i in text if i not in string.punctuation])
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([i for i in words if i not in stop_words])
    text = re.sub(r"\s+", " ", text).strip()
    return text

sentiment_mapping = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

@app.post("/predict/")
def predict_sentiment(data: TextData):
    text = clean_text(data.text)
    text_tfidf = tfidf.transform([text])
    text_pca = svd.transform(text_tfidf)
    prediction = model.predict(text_pca)[0]
    sentiment = sentiment_mapping.get(prediction, "Unknown")
    return {"prediction": int(prediction)}

@app.post("/predict_batch/")
def predict_batch_sentiment(data: TextBatchData):
    predictions = []
    for text in data.texts:
        cleaned_text = clean_text(text)
        text_tfidf = tfidf.transform([cleaned_text])
        text_pca = svd.transform(text_tfidf)
        prediction = model.predict(text_pca)[0]
        sentiment = sentiment_mapping.get(prediction, "Unknown")
        predictions.append(sentiment)
    return {"predictions": predictions}

@app.post("/predict_with_confidence/")
def predict_with_confidence(data: TextData):
    text = clean_text(data.text)
    text_tfidf = tfidf.transform([text])
    text_pca = svd.transform(text_tfidf)
    prediction = model.predict(text_pca)[0]
    prediction_proba = model.predict_proba(text_pca)[0]
    sentiment = sentiment_mapping.get(prediction, "Unknown")
    confidence = prediction_proba[prediction]
    return {"prediction": sentiment, "confidence": confidence}

