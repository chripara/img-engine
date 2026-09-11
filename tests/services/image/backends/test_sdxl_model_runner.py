from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.services.image.backends.sdxl_model_runner import SDXLModelRunner

MODULE = "app.services.image.backends.sdxl_model_runner"


@pytest.fixture
def mocks():
    with patch(f"{MODULE}.AutoencoderKL") as vae_cls, \
         patch(f"{MODULE}.DiffusionPipeline") as plain_pipe_cls, \
         patch(f"{MODULE}.StableDiffusionXLControlNetPipeline") as cn_pipe_cls, \
         patch(f"{MODULE}.Compel") as compel_cls, \
         patch(f"{MODULE}.torch") as torch_mod:
        yield {
            "vae_cls": vae_cls, "plain_pipe_cls": plain_pipe_cls,
            "cn_pipe_cls": cn_pipe_cls, "compel_cls": compel_cls, "torch": torch_mod,
        }


def _fake_pipe():
    pipe = MagicMock(name="pipe")
    pipe.tokenizer = MagicMock(name="tokenizer")
    pipe.tokenizer_2 = MagicMock(name="tokenizer_2")
    pipe.text_encoder = MagicMock(name="text_encoder")
    pipe.text_encoder_2 = MagicMock(name="text_encoder_2")
    return pipe


def _loaded_runner(mocks, **load_overrides):
    fake_pipe = _fake_pipe()
    fake_pipe.return_value.images = [Image.new("RGB", (2, 2))]
    mocks["plain_pipe_cls"].from_pretrained.return_value = fake_pipe
    mocks["compel_cls"].return_value.side_effect = [("cond", "pooled")] * 5

    runner = SDXLModelRunner()
    kwargs = dict(checkpoint_id="x", vae_id=None, controlnet_model_cls=None, controlnet_repo_ids=[], use_cpu_offload=False)
    kwargs.update(load_overrides)
    runner.load(**kwargs)
    return runner, fake_pipe


def test_load_without_vae_without_controlnet_uses_plain_pipeline(mocks):
    runner, fake_pipe = _loaded_runner(mocks)

    mocks["vae_cls"].from_pretrained.assert_not_called()
    mocks["cn_pipe_cls"].from_pretrained.assert_not_called()
    _, kwargs = mocks["plain_pipe_cls"].from_pretrained.call_args
    assert "vae" not in kwargs
    fake_pipe.to.assert_called_once_with("cuda")
    fake_pipe.enable_model_cpu_offload.assert_not_called()


def test_load_with_vae_id_passes_vae_to_pipeline(mocks):
    fake_vae = MagicMock(name="vae")
    mocks["vae_cls"].from_pretrained.return_value = fake_vae

    _, fake_pipe = _loaded_runner(mocks, vae_id="madebyollin/sdxl-vae-fp16-fix")

    mocks["vae_cls"].from_pretrained.assert_called_once_with(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=mocks["torch"].float16
    )
    _, kwargs = mocks["plain_pipe_cls"].from_pretrained.call_args
    assert kwargs["vae"] is fake_vae


def test_load_with_controlnet_repo_ids_uses_controlnet_pipeline(mocks):
    controlnet_cls = MagicMock(name="ControlNetModel")
    fake_controlnets = [MagicMock(name="cn1"), MagicMock(name="cn2")]
    controlnet_cls.from_pretrained.side_effect = fake_controlnets
    fake_cn_pipe = _fake_pipe()
    fake_cn_pipe.return_value.images = [Image.new("RGB", (2, 2))]
    mocks["cn_pipe_cls"].from_pretrained.return_value = fake_cn_pipe

    runner = SDXLModelRunner()
    runner.load(checkpoint_id="x", vae_id=None, controlnet_model_cls=controlnet_cls,
                controlnet_repo_ids=["repo/a", "repo/b"], use_cpu_offload=False)

    assert controlnet_cls.from_pretrained.call_count == 2
    _, kwargs = mocks["cn_pipe_cls"].from_pretrained.call_args
    assert kwargs["controlnet"] == fake_controlnets
    mocks["plain_pipe_cls"].from_pretrained.assert_not_called()


def test_load_use_cpu_offload_true_skips_to_cuda(mocks):
    _, fake_pipe = _loaded_runner(mocks, use_cpu_offload=True)
    fake_pipe.enable_model_cpu_offload.assert_called_once()
    fake_pipe.to.assert_not_called()


def test_apply_lora_calls_pipe_methods_in_expected_order(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    runner.apply_lora("some/lora-repo", "my_style", 0.6)

    fake_pipe.load_lora_weights.assert_called_once_with("some/lora-repo", adapter_name="my_style")
    fake_pipe.set_adapters.assert_called_once_with("my_style", adapter_weights=0.6)
    fake_pipe.fuse_lora.assert_called_once()


def test_bind_scheduler_replaces_scheduler_via_from_config(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    fake_pipe.scheduler.config = "some-config"
    scheduler_cls = MagicMock(name="SchedulerCls")

    runner.bind_scheduler(scheduler_cls)

    scheduler_cls.from_config.assert_called_once_with("some-config")
    assert fake_pipe.scheduler == scheduler_cls.from_config.return_value


def test_run_uses_generator_only_when_seed_given(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    runner.run(prompt="a cat", negative_prompt=None, width=64, height=64, steps=10, cfg=5.0,
               seed=42, control_images=None, control_strengths=None, index=0)

    mocks["torch"].Generator.assert_called_once_with(device="cuda")
    mocks["torch"].Generator.return_value.manual_seed.assert_called_once_with(42)


def test_run_without_seed_passes_none_generator(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    runner.run(prompt="a cat", negative_prompt=None, width=64, height=64, steps=10, cfg=5.0,
               seed=None, control_images=None, control_strengths=None, index=0)

    mocks["torch"].Generator.assert_not_called()
    _, kwargs = fake_pipe.call_args
    assert kwargs["generator"] is None


def test_run_includes_control_kwargs_only_when_given(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    fake_images = [Image.new("RGB", (1, 1))]
    runner.run(prompt="a cat", negative_prompt=None, width=64, height=64, steps=10, cfg=5.0,
               seed=None, control_images=fake_images, control_strengths=[0.5], index=0)

    _, kwargs = fake_pipe.call_args
    assert kwargs["image"] == fake_images
    assert kwargs["controlnet_conditioning_scale"] == [0.5]


def test_run_omits_control_kwargs_when_none(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    runner.run(prompt="a cat", negative_prompt=None, width=64, height=64, steps=10, cfg=5.0,
               seed=None, control_images=None, control_strengths=None, index=0)

    _, kwargs = fake_pipe.call_args
    assert "image" not in kwargs
    assert "controlnet_conditioning_scale" not in kwargs


def test_run_returns_first_result_image(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    expected = Image.new("RGB", (3, 3))
    fake_pipe.return_value.images = [expected]

    result = runner.run(prompt="a cat", negative_prompt=None, width=64, height=64, steps=10, cfg=5.0,
                         seed=None, control_images=None, control_strengths=None, index=0)
    assert result is expected


def test_unload_moves_pipe_to_cpu_and_clears_cuda_cache(mocks):
    runner, fake_pipe = _loaded_runner(mocks)
    runner.unload()
    fake_pipe.to.assert_called_with("cpu")
    mocks["torch"].cuda.empty_cache.assert_called_once()


def test_unload_is_safe_before_load_and_when_called_twice(mocks):
    runner = SDXLModelRunner()
    runner.unload()
    runner.unload()
