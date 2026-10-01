from smoke_tests._client import check, post_generate, summarize


def run():
    failures = []

    payload = {
        "profile": "scene_frame",
        "num_images": 1,
        "prompt": "an empty misty mountain valley at dawn, no people, wide landscape",
        "seed": 41,
    }
    r = post_generate(payload)
    check(failures, "scene-only request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code != 200:
        return summarize("test_quality_gates", failures)

    quality = r.json()["images"][0].get("quality") or []
    by_gate = {g.get("gate"): g for g in quality}

    check(failures, "response has all 5 gates", len(quality) == 5, f"got {len(quality)}")

    hands = by_gate.get("hands", {})
    check(failures, "hands gate is NOT_APPLICABLE for a scene with no hands", hands.get("score") is None and hands.get("passed") is None, f"got {hands}")

    face = by_gate.get("face", {})
    check(failures, "face gate is NOT_APPLICABLE for a scene with no face", face.get("score") is None and face.get("passed") is None, f"got {face}")

    tiling = by_gate.get("tiling", {})
    check(failures, "tiling gate returns a real score", tiling.get("score") is not None, f"got {tiling}")

    return summarize("test_quality_gates", failures)


if __name__ == "__main__":
    run()
