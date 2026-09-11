import pytest
from PIL import Image

from app.services.upscaler.backends.base_backend import BaseBackend
from tests.support.contract import assert_every_concrete_class_has_a_rig
from tests.services.upscaler.backends.rigs import BACKEND_RIGS, _minimal_request


def test_every_concrete_upscaler_backend_has_a_registered_rig():
    assert_every_concrete_class_has_a_rig(BaseBackend, BACKEND_RIGS)


@pytest.mark.parametrize("rig", BACKEND_RIGS.values(), ids=lambda r: r.backend_cls.__name__)
class TestUpscalerBackendLifecycleContract:
    def test_construct_unloaded_does_not_raise(self, rig):
        rig.make_unloaded()

    def test_load_then_upscale_returns_a_pil_image(self, rig):
        with rig.make_loaded() as backend:
            result = backend.upscale(Image.new("RGB", (16, 16)), _minimal_request(), index=0, seed=1)
            assert isinstance(result, Image.Image)

    def test_unload_after_load_does_not_raise(self, rig):
        with rig.make_loaded():
            pass

    def test_unload_without_load_does_not_raise(self, rig):
        rig.make_unloaded().unload()

    def test_reload_does_not_raise(self, rig):
        with rig.make_loaded() as backend:
            rig.reload(backend)
