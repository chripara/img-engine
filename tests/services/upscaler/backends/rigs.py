from contextlib import contextmanager
from PIL import Image

from app.schemas.generate import GenerateRequest
from app.services.upscaler.backends.esrgan_backend import ESRGANBackend
from app.services.upscaler.backends.latent_diffusion_backend import LatentDiffusionBackend
from app.services.upscaler.backends.tensor_upscale_runner import TensorUpscaleRunner
from app.services.upscaler.backends.tiled_diffusion_upscale_runner import TiledDiffusionUpscaleRunner
from tests.support.rig import Rig
from utils.enums.profile import Profile
from utils.enums.upscale import Upscaler


def _minimal_request() -> GenerateRequest:
    return GenerateRequest(profile=Profile.CHARACTER, num_images=1, prompt="a cat",
                            subject=None, environment=None, feeling=None)


class _DumbFakeTensorRunner(TensorUpscaleRunner):
    def load(self, model_path) -> None:
        pass

    def run(self, image) -> Image.Image:
        return image

    def unload(self) -> None:
        pass


class _DumbFakeTiledRunner(TiledDiffusionUpscaleRunner):
    def load(self, model_path) -> None:
        pass

    def run_tile(self, prompt, tile, noise_level, steps) -> Image.Image:
        return tile

    def unload(self) -> None:
        pass


@contextmanager
def _esrgan_loaded():
    backend = ESRGANBackend(upscaler=Upscaler.ESRGAN, runner=_DumbFakeTensorRunner())
    backend.load()
    try:
        yield backend
    finally:
        backend.unload()


@contextmanager
def _latent_loaded():
    backend = LatentDiffusionBackend(runner=_DumbFakeTiledRunner())
    backend.load()
    try:
        yield backend
    finally:
        backend.unload()


BACKEND_RIGS: dict[type, Rig] = {
    ESRGANBackend: Rig(
        backend_cls=ESRGANBackend,
        make_unloaded=lambda: ESRGANBackend(upscaler=Upscaler.ESRGAN, runner=_DumbFakeTensorRunner()),
        make_loaded=_esrgan_loaded,
        reload=lambda backend: backend.load(),
    ),
    LatentDiffusionBackend: Rig(
        backend_cls=LatentDiffusionBackend,
        make_unloaded=lambda: LatentDiffusionBackend(runner=_DumbFakeTiledRunner()),
        make_loaded=_latent_loaded,
        reload=lambda backend: backend.load(),
    ),
}
