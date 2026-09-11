import numpy as np
import torch, gc
from PIL import Image
from spandrel import ModelLoader

from app.services.upscaler.backends.tensor_upscale_runner import TensorUpscaleRunner


class SpandrelTensorRunner(TensorUpscaleRunner):
    def __init__(self) -> None:
        self._model = None

    def load(self, model_path: str) -> None:
        self._model = ModelLoader().load_from_file(model_path).cuda()

    def run(self, image: Image.Image) -> Image.Image:
        if self._model is None:
            raise RuntimeError("SpandrelTensorRunner not loaded. Call load() first.")

        # (Finding: λείπει `.convert("RGB")` πριν το np.array — RGBA input θα
        # έσπαγε στο permute(2,0,1). ΔΕΝ το διορθώνω εδώ, το μεταφέρω ως έχει.)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to("cuda")

        with torch.no_grad():
            output = self._model(tensor)

        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1)
        return Image.fromarray((output * 255).byte().cpu().numpy())

    def unload(self) -> None:
        if self._model is not None:
            self._model = self._model.cpu()
        self._model = None
        torch.cuda.empty_cache()
        gc.collect()
