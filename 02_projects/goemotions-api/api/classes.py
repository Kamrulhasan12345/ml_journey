import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionModelLoader:
  def __init__(self, model_path: str, device: str = None):
    self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    self.model.to(self.device)
    self.model.eval()

class EmotionPredictor:
  def __init__(self, model_loader: EmotionModelLoader, threshold: float = 0.5):
    self.model_loader = model_loader
    self.threshold = threshold
    self.emotion_labels = [ "admiration","amusement","anger","annoyance","approval","caring","confusion", "curiosity","desire","disappointment","disapproval","disgust","embarrassment", "excitement","fear","gratitude","grief","joy","love","nervousness","optimism", "pride","realization","relief","remorse","sadness","surprise","neutral" ]

  def predict(self, text: str):
    inputs = self.model_loader.tokenizer(
      text,
      return_tensors='pt',
      truncation=True,
      padding=True,
      max_length=128
    ).to(self.model_loader.device)

    with torch.no_grad():
      logits = self.model_loader.model(**inputs).logits
      probs = torch.sigmoid(logits).cpu().numpy()

    predictions = {
      self.emotion_labels[i]: float(prob) for i, prob in enumerate(probs[0]) if prob >= self.threshold
    }
    
    return {"text": text, "predicted_emotions": predictions}