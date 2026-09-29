from smoke_tests._client import check, post_generate, summarize


def run():
    failures = []

    payload = {
        "profile": "character",
        "num_images": 1,
        "prompt": "a knight standing in a ruined castle",
        "refine": True,
        "seed": 7,
    }
    r = post_generate(payload)
    check(failures, "refine=true request succeeds", r.status_code == 200, f"status={r.status_code}")
    if r.status_code != 200:
        return summarize("test_prompt_refinement", failures)

    body = r.json()
    refined = body.get("refined_prompt")

    check(failures, "refined_prompt is non-empty", bool(refined))
    check(failures, "refined_prompt differs from raw input prompt", refined != payload["prompt"], f"got {refined!r}")
    check(failures, "refined_prompt looks like a comma-separated tag list", isinstance(refined, str) and "," in refined, f"got {refined!r}")

    return summarize("test_prompt_refinement", failures)


if __name__ == "__main__":
    run()
