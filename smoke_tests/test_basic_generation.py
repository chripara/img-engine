import base64
from smoke_tests._client import check, post_generate, summarize

EXPECTED_GATES = {"tiling", "clip", "hands", "face", "iqa"}

def run():
    failures = []

    payload = {
        "profile": "product",
        "num_images": 1,
        "prompt": "a red apple on a wooden table",
        "seed": 42,
    }
    r = post_generate(payload)
    check(failures, "request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code != 200:
        return summarize("test_basic_generation", failures)

    body = r.json()

    check(failures, "response has images list of length 1", len(body.get("images", [])) == 1)

    image_result = body["images"][0]
    check(failures, "image field is non-empty base64", bool(image_result.get("image")))

    try:
        base64.b64decode(image_result.get("image", ""), validate=True)
        decodable = True
    except Exception:
        decodable = False
    check(failures, "image field decodes as valid base64", decodable)

    check(failures, "seed echoed back matches request", image_result.get("seed") == 42)

    quality = image_result.get("quality") or []
    check(failures, "quality has 5 gates", len(quality) == 5, f"got {len(quality)}")

    gate_names = {g.get("gate") for g in quality}
    check(failures, "quality gates match expected set", gate_names == EXPECTED_GATES, f"got {gate_names}")

    check(failures, "refined_prompt echoes original prompt when refine=false", body.get("refined_prompt") == payload["prompt"])

    return summarize("test_basic_generation", failures)


if __name__ == "__main__":
    run()
