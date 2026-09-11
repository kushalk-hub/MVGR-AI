from transformers import pipeline

from .base import BaseDetector


class RoBERTaDetector(BaseDetector):
    def __init__(self, model_name: str = "openai-community/roberta-base-openai-detector"):
        self.pipe = pipeline("text-classification", model=model_name, device=0)
        self.fake_label = None

    def _prob_ai(self, entry):
        label = entry["label"]
        if label == "Fake":
            return float(entry["score"])
        return 1.0 - float(entry["score"])

    def score(self, text: str) -> float:
        out = self.pipe(text, truncation=True, max_length=512)[0]
        return self._prob_ai(out)

    def score_batch(self, texts):
        results = self.pipe(texts, truncation=True, max_length=512, batch_size=32)
        return [self._prob_ai(r) for r in results]