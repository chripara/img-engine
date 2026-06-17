from app.services.backends.base_backend import BaseBackend
from PIL import Image
import io

class SDXLBackend(BaseBackend):
    def generate(self, prompt: str) -> bytes:


        img = Image.new("RGBA", (200, 200), (0, 128, 255, 255))  # μπλε τετράγωνο

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        # png_bytes = (
        #     b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        #     b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
        #     b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
        #     b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        # )
        
        return png_bytes