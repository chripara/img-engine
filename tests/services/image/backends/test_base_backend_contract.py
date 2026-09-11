import pytest
from PIL import Image

from app.services.image.backends.base_backend import BaseBackend
from app.services.registries.image_registry import Dimensions
from tests.support.contract import assert_every_concrete_class_has_a_rig
from tests.services.image.backends.rigs import BACKEND_RIGS


def test_every_concrete_image_backend_has_a_registered_rig():
    assert_every_concrete_class_has_a_rig(BaseBackend, BACKEND_RIGS)

@pytest.mark.parametrize("rig", BACKEND_RIGS.values(), ids=lambda r: r.backend_cls.__name__)
class TestImageBackendLifecycleContract:

    def test_construct_unloaded_does_not_raise(self, rig):
        rig.make_unloaded()

    def test_load_then_generate_returns_a_pil_image(self, rig):
        with rig.make_loaded() as backend:
            result = backend.generate(
                prompt="a cat",
                negative_prompt=None,
                dimensions=Dimensions(width=64, height=64),
                seed=123,
                controls=None,
            )
            assert isinstance(result, Image.Image)

    def test_unload_after_load_does_not_raise(self, rig):
        with rig.make_loaded():
            pass

    def test_unload_without_load_does_not_raise(self, rig):
        rig.make_unloaded().unload()

    def test_reload_does_not_raise(self, rig):
        with rig.make_loaded() as backend:
            rig.reload(backend)
