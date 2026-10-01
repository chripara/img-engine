from smoke_tests._client import check, post_generate, summarize


def run():
    failures = []

    payload = {
        "profile": "product",
        "num_images": 1,
        "prompt": "a ceramic vase on a stone pedestal",
        "seed": 31,
        "upscale_quality": "enhanced",
    }
    r = post_generate(payload)
    check(failures, "upscale_quality='enhanced' (ESRGAN) request succeeds", r.status_code == 200, f"status={r.status_code}")

    payload = {
        "profile": "product",
        "num_images": 1,
        "prompt": "a ceramic vase on a stone pedestal",
        "seed": 32,
        "upscale_quality": "generative",
    }
    r = post_generate(payload)
    check(failures, "upscale_quality='generative' (latent diffusion) request succeeds", r.status_code == 200, f"status={r.status_code}")

    return summarize("test_upscaling", failures)


if __name__ == "__main__":
    run()
