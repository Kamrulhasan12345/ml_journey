from fastapi import FastAPI
from pydantic import BaseModel
from .classes import EmotionModelLoader, EmotionPredictor

app = FastAPI()

distilbert_loader = EmotionModelLoader(model_path="kamrulhasan12345/goemotions-distilbert")
distilbert_predictor = EmotionPredictor(model_loader=distilbert_loader)

bert_loader = EmotionModelLoader(model_path="kamrulhasan12345/goemotions-bert")
bert_predictor = EmotionPredictor(model_loader=bert_loader)

class TextInput(BaseModel):
    text: str

@app.post("/predict-distilbert")
def predict_distilbert(input: TextInput):
    result = distilbert_predictor.predict(input.text)
    return result

@app.post("/predict-bert")
def predict_bert(input: TextInput):
    result = bert_predictor.predict(input.text)
    return result