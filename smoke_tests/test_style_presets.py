from smoke_tests._client import check, post_generate, summarize

PRESETS = [
    "fantasy",
    "dark_fantasy",
    "cartoonish_fantasy",
    "cyberpunk",
    "realism_cartoonish",
    "scifi_fantasy",
    "medieval_fantasy",
    "anime_aesthetic",
]


def run():
    failures = []

    for preset in PRESETS:
        payload = {
            "profile": "character",
            "num_images": 1,
            "prompt": "a warrior standing in a misty forest",
            "seed": 21,
            "style_preset": preset,
        }
        r = post_generate(payload)
        check(failures, f"style_preset '{preset}' request succeeds", r.status_code == 200, f"status={r.status_code}")

    payload = {
        "profile": "character",
        "num_images": 1,
        "prompt": "a warrior standing in a misty forest",
        "seed": 22,
        "style_preset": "cyberpunk",
        "lora_strength": 0.3,
    }
    r = post_generate(payload)
    check(failures, "style_preset with custom lora_strength succeeds", r.status_code == 200, f"status={r.status_code}")

    return summarize("test_style_presets", failures)


if __name__ == "__main__":
    run()
