from PIL import Image
import io

class ImageConverter:
    @staticmethod
    def Pil_Image_to_Bytes_Png(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", dpi=(300, 300))
        return buffer.getvalue()