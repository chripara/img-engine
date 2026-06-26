from abc import ABC, abstractmethod

class BaseBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, seed: int | None) -> bytes:
        pass