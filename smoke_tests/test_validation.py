from smoke_tests._client import check, post_generate, summarize

BASE_PAYLOAD = {
    "profile": "product",
    "num_images": 1,
    "prompt": "a red apple on a wooden table",
}


def run():
    failures = []

    r = post_generate({**BASE_PAYLOAD, "prompt": ""})
    check(failures, "empty prompt rejected", r.status_code == 422, f"status={r.status_code}")

    payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "prompt"}
    r = post_generate(payload)
    check(failures, "missing prompt field rejected", r.status_code == 422, f"status={r.status_code}")

    payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "profile"}
    r = post_generate(payload)
    check(failures, "missing profile field rejected", r.status_code == 422, f"status={r.status_code}")

    r = post_generate({**BASE_PAYLOAD, "num_images": 0})
    check(failures, "num_images below range rejected", r.status_code == 422, f"status={r.status_code}")

    r = post_generate({**BASE_PAYLOAD, "num_images": 11})
    check(failures, "num_images above range rejected", r.status_code == 422, f"status={r.status_code}")

    r = post_generate({**BASE_PAYLOAD, "prompt": "a" * 601})
    check(failures, "over-length prompt rejected", r.status_code == 422, f"status={r.status_code}")

    r = post_generate({**BASE_PAYLOAD, "lora_strength": 1.5})
    check(failures, "out-of-range lora_strength rejected", r.status_code == 422, f"status={r.status_code}")

    return summarize("test_validation", failures)


if __name__ == "__main__":
    run()
