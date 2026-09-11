from abc import ABC, abstractmethod
from typing import List, Union


class BaseDetector(ABC):
    @abstractmethod
    def score(self, text: str) -> float:
        raise NotImplementedError

    def score_batch(self, texts: List[str]) -> List[float]:
        return [self.score(t) for t in texts]