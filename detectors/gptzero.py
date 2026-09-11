import os
import time

import requests

from .base import BaseDetector

GPTZERO_URL = "https://api.gptzero.me/v2/predict/text"


class GPTZeroDetector(BaseDetector):
    def __init__(self, api_key: str = None):
        self.api_key = (
            api_key
            or os.environ.get("GPTZERO_API_KEY")
            or os.environ.get("SECRET_GPTZERO_API_KEY")
        )
        self.available = bool(self.api_key)
        if not self.available:
            print("WARNING: GPTZERO_API_KEY not found. GPTZero API detector unavailable.")

    def score(self, text: str) -> float:
        if not self.available:
            raise RuntimeError("GPTZero API key missing.")
        response = requests.post(
            GPTZERO_URL,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json={"document": text, "multilingual": False},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        doc = data.get("document", data)
        probs = doc.get("class_probabilities", {})
        if probs:
            for key, value in probs.items():
                if key.lower() == "ai":
                    return float(value)
        classification = doc.get("document_classification", "").lower()
        if "ai_only" in classification:
            return 1.0
        if "human_only" in classification:
            return 0.0
        return 0.5

    def score_batch(self, texts, limit=None, sleep=1.0):
        scores = []
        for i, text in enumerate(texts):
            if limit is not None and i >= limit:
                break
            try:
                scores.append(self.score(text))
            except Exception as exc:
                print(f"GPTZero error on sample {i}: {exc}")
                scores.append(float("nan"))
            time.sleep(sleep)
        return scores


class GPTZeroStyleDetector(BaseDetector):
    """Local, key-free, GPTZero-style detector.

    GPTZero's core statistical signal is token-level perplexity (plus
    burstiness = sentence-level perplexity variance). AI-generated text tends
    to be more predictable (lower perplexity / lower burstiness) than
    human-written text. This replaces the paid API with a local equivalent
    computed from a small open model, documented as a proxy signal.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.model.eval()
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def _logppl(self, texts):
        import torch

        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        ).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            logits = self.model(**encodings).logits
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = encodings.input_ids[..., 1:].contiguous()
        shifted_mask = encodings.attention_mask[..., 1:].contiguous().float()
        ce = self.ce(shifted_logits.transpose(1, 2), shifted_labels) * shifted_mask
        logppl = ce.sum(1) / shifted_mask.sum(1)
        return logppl.cpu().float()

    def score(self, text: str) -> float:
        return -float(self._logppl([text])[0])

    def score_batch(self, texts):
        scores = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            scores.extend((-logppl.item() for logppl in self._logppl(chunk)))
        return scores


def make_gptzero(api_key: str = None):
    api = GPTZeroDetector(api_key)
    if api.available:
        return api, "api"
    return GPTZeroStyleDetector(), "local"