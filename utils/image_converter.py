from PIL import Image

class ImageConverter:
    @staticmethod
    def Pil_Image_to_Bytes_Png(image: Image.Image) -> bytes:
        return image.tobytes()