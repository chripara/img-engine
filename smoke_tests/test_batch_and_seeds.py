from smoke_tests._client import check, post_generate, summarize


def run():
    failures = []

    payload = {
        "profile": "product",
        "num_images": 3,
        "prompt": "a small wooden chest, closed, on a stone floor",
        "seed": 100,
    }
    r = post_generate(payload)
    check(failures, "seeded batch request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        images = r.json().get("images", [])
        check(failures, "seeded batch returns 3 images", len(images) == 3, f"got {len(images)}")
        seeds = [img.get("seed") for img in images]
        check(failures, "seeded batch seeds are sequential", seeds == [100, 101, 102], f"got {seeds}")

    payload = {
        "profile": "product",
        "num_images": 2,
        "prompt": "a small wooden chest, closed, on a stone floor",
    }
    r = post_generate(payload)
    check(failures, "unseeded batch request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        images = r.json().get("images", [])
        check(failures, "unseeded batch returns 2 images", len(images) == 2, f"got {len(images)}")
        seeds = [img.get("seed") for img in images]
        check(failures, "unseeded batch does not return a seed value", all(s is None for s in seeds), f"got {seeds}")

    payload = {
        "profile": "product",
        "num_images": 3,
        "prompt": "a small wooden chest, closed, on a stone floor",
        "seed": 500,
        "spread": 5,
    }
    r = post_generate(payload)
    check(failures, "spread batch (population >= num_images) request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        images = r.json().get("images", [])
        check(failures, "spread batch returns 3 images", len(images) == 3, f"got {len(images)}")
        seeds = [img.get("seed") for img in images]
        in_range = all(s is not None and 495 <= s <= 505 for s in seeds)
        check(failures, "spread batch seeds fall within [seed-spread, seed+spread]", in_range, f"got {seeds}")
        check(failures, "spread batch seeds are unique (sample path)", len(set(seeds)) == len(seeds), f"got {seeds}")

    payload = {
        "profile": "product",
        "num_images": 5,
        "prompt": "a small wooden chest, closed, on a stone floor",
        "seed": 300,
        "spread": 1,
    }
    r = post_generate(payload)
    check(failures, "spread batch (population < num_images) request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        images = r.json().get("images", [])
        check(failures, "narrow-spread batch returns 5 images", len(images) == 5, f"got {len(images)}")
        seeds = [img.get("seed") for img in images]
        in_range = all(s is not None and 299 <= s <= 301 for s in seeds)
        check(failures, "narrow-spread batch seeds fall within [seed-spread, seed+spread]", in_range, f"got {seeds}")

    payload = {
        "profile": "product",
        "num_images": 3,
        "prompt": "a small wooden chest, closed, on a stone floor",
        "seed": 777,
        "spread": 0,
    }
    r = post_generate(payload)
    check(failures, "explicit spread=0 batch request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        images = r.json().get("images", [])
        check(failures, "explicit spread=0 batch returns 3 images", len(images) == 3, f"got {len(images)}")
        seeds = [img.get("seed") for img in images]
        check(failures, "explicit spread=0 gives identical seeds for all images", seeds == [777, 777, 777], f"got {seeds}")

    return summarize("test_batch_and_seeds", failures)


if __name__ == "__main__":
    run()
