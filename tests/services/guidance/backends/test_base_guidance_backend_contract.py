import pytest
from PIL import Image

from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from tests.support.contract import assert_every_concrete_class_has_a_rig
from tests.services.guidance.backends.rigs import BACKEND_RIGS

pytestmark = pytest.mark.contract

def test_every_concrete_guidance_backend_has_a_registered_rig():
    assert_every_concrete_class_has_a_rig(BaseGuidanceBackend, BACKEND_RIGS)


@pytest.mark.parametrize("rig", BACKEND_RIGS.values(), ids=lambda r: r.backend_cls.__name__)
class TestGuidanceBackendLifecycleContract:

    def test_construct_unloaded_does_not_raise(self, rig):
        rig.make_unloaded()

    def test_load_then_preprocess_returns_an_image(self, rig):
        with rig.make_loaded() as backend:
            result = backend.preprocess(Image.new("RGB", (4, 4)))
            assert isinstance(result, Image.Image)

    def test_unload_after_load_does_not_raise(self, rig):
        with rig.make_loaded():
            pass

    def test_unload_without_load_does_not_raise(self, rig):
        rig.make_unloaded().unload()

    def test_reload_does_not_raise(self, rig):
        with rig.make_loaded() as backend:
            rig.reload(backend)
