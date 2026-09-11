from typing import List, Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseDetector

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def assert_tokenizer_consistency(model_id_1, model_id_2):
    vocab_1 = AutoTokenizer.from_pretrained(model_id_1).vocab
    vocab_2 = AutoTokenizer.from_pretrained(model_id_2).vocab
    if vocab_1 != vocab_2:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


def perplexity(encoding, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    ce = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask
    ppl = ce.sum(1) / shifted_attention_mask.sum(1)
    return ppl.to("cpu").float().numpy()


def entropy(p_logits, q_logits, encoding, pad_token_id):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_proba = softmax_fn(p_logits).view(-1, vocab_size)
    q_scores = q_logits.view(-1, vocab_size)
    ce = ce_loss_fn(q_scores, p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    agg_ce = (ce * padding_mask).sum(1) / padding_mask.sum(1)
    return agg_ce.to("cpu").float().numpy()


class BinocularsDetector(BaseDetector):
    def __init__(
        self,
        observer_name_or_path: str = "TinyLlama/TinyLlama-1.1B",
        performer_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_token_observed: int = 512,
        batch_size: int = 4,
    ):
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        self.max_token_observed = max_token_observed
        self.batch_size = batch_size

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path, device_map="auto", torch_dtype=torch.float16
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path, device_map="auto", torch_dtype=torch.float16
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(self, batch):
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(next(self.observer_model.parameters()).device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings):
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        if self.observer_model.device.type == "cuda":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[List[str], str]) -> Union[float, List[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits, performer_logits, encodings, self.tokenizer.pad_token_id)
        scores = ppl / x_ppl
        scores = scores.tolist()
        return scores[0] if isinstance(input_text, str) else scores

    def score(self, text: str) -> float:
        raw = self.compute_score(text)
        return -float(raw)

    def score_batch(self, texts: List[str]) -> List[float]:
        raw_scores = self.compute_score(texts)
        return [-float(s) for s in raw_scores]