import base64
import io
from PIL import Image, ImageDraw
from smoke_tests._client import check, post_generate, summarize

GUIDANCE_TYPES = ["canny", "depth", "pose", "scribble"]


def _sample_image_b64():
    img = Image.new("RGB", (512, 512), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 150, 350, 350), fill=(220, 220, 220))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def run():
    failures = []
    reference_image = _sample_image_b64()

    for guidance_type in GUIDANCE_TYPES:
        payload = {
            "profile": "character",
            "num_images": 1,
            "prompt": "a hooded figure standing still",
            "seed": 11,
            "controls": {
                "images": [reference_image],
                "controls": [{"selector": 0, "type": guidance_type}],
            },
        }
        r = post_generate(payload)
        check(failures, f"controlnet '{guidance_type}' request succeeds", r.status_code == 200, f"status={r.status_code}")

    return summarize("test_controlnet", failures)


if __name__ == "__main__":
    run()
