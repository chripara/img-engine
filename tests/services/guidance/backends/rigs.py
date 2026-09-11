from contextlib import contextmanager

from app.services.guidance.backends.preprocessor_runner import GuidancePreprocessorRunner
from app.services.guidance.backends.sdxl_guidance_backend import SDXLGuidanceBackend
from tests.support.rig import Rig
from utils.enums.guidance import GuidanceType


class _DumbFakePreprocessorRunner(GuidancePreprocessorRunner):
    def load(self, guidance_type) -> None:
        pass

    def run(self, image):
        return image

    def unload(self) -> None:
        pass


@contextmanager
def _sdxl_guidance_backend_loaded():
    backend = SDXLGuidanceBackend(runner=_DumbFakePreprocessorRunner())
    backend.load(GuidanceType.CANNY)
    try:
        yield backend
    finally:
        backend.unload()


BACKEND_RIGS: dict[type, Rig] = {
    SDXLGuidanceBackend: Rig(
        backend_cls=SDXLGuidanceBackend,
        make_unloaded=lambda: SDXLGuidanceBackend(runner=_DumbFakePreprocessorRunner()),
        make_loaded=_sdxl_guidance_backend_loaded,
        reload=lambda backend: backend.load(GuidanceType.CANNY),
    ),
}
