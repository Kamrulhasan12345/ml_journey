from fastapi import FastAPI
from pydantic import BaseModel
from api.classes import EmotionModelLoader, EmotionPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global distilbert_predictor
    global bert_predictor
    try:
        logger.info("Loading models on startup...")
        distilbert_model_loader = EmotionModelLoader(model_path="kamrulhasan12345/goemotions-distilbert")
        distilbert_predictor = EmotionPredictor(distilbert_model_loader)
        bert_model_loader = EmotionModelLoader(model_path="kamrulhasan12345/goemotions-bert")
        bert_predictor = EmotionPredictor(bert_model_loader)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.post("/predict-distilbert")
def predict_distilbert(input: TextInput):
    return distilbert_predictor.predict(input.text)

@app.post("/predict-bert")
def predict_bert(input: TextInput):
    return bert_predictor.predict(input.text)