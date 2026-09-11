from abc import ABC, abstractmethod
from typing import Type
from PIL import Image


class ImageModelRunner(ABC):

    @abstractmethod
    def load(
        self,
        checkpoint_id: str,
        vae_id: str | None,
        controlnet_model_cls: Type | None,
        controlnet_repo_ids: list[str],
        use_cpu_offload: bool,
    ) -> None:
        pass

    @abstractmethod
    def apply_lora(self, lora_repo_id: str, adapter_name: str, strength: float) -> None:
        pass

    @abstractmethod
    def bind_scheduler(self, scheduler_cls: Type) -> None:
        pass

    @abstractmethod
    def run(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int | None,
        control_images: list[Image.Image] | None,
        control_strengths: list[float] | None,
        index: int,
    ) -> Image.Image:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass
