from contextlib import contextmanager
from PIL import Image

from app.services.image.backends.model_runner import ImageModelRunner
from app.services.image.backends.sdxl_backend import SDXLBackend
from tests.support.rig import Rig
from utils.enums.profile import Profile


class _DumbFakeImageModelRunner(ImageModelRunner):
    
    def load(self, **kwargs) -> None:
        pass

    def apply_lora(self, lora_repo_id, adapter_name, strength) -> None:
        pass

    def bind_scheduler(self, scheduler_cls) -> None:
        pass

    def run(self, **kwargs) -> Image.Image:
        return Image.new("RGB", (8, 8))

    def unload(self) -> None:
        pass


def _minimal_load_kwargs() -> dict:
    return dict(
        profile=Profile.CHARACTER,
        style_preset=None,
        lora_weight=None,
        use_controlnet=False,
        guidance_types=[],
    )


@contextmanager
def _sdxl_backend_loaded():
    backend = SDXLBackend(profile=Profile.CHARACTER, runner=_DumbFakeImageModelRunner())
    backend.load(**_minimal_load_kwargs())
    try:
        yield backend
    finally:
        backend.unload()


BACKEND_RIGS: dict[type, Rig] = {
    SDXLBackend: Rig(
        backend_cls=SDXLBackend,
        make_unloaded=lambda: SDXLBackend(profile=Profile.CHARACTER, runner=_DumbFakeImageModelRunner()),
        make_loaded=_sdxl_backend_loaded,
        reload=lambda backend: backend.load(**_minimal_load_kwargs()),
    ),
}
