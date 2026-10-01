from smoke_tests import (
    test_health,
    test_validation,
    test_basic_generation,
    test_batch_and_seeds,
    test_prompt_refinement,
    test_controlnet,
    test_style_presets,
    test_upscaling,
    test_quality_gates,
)

SUITES = [
    test_health,
    test_validation,
    test_basic_generation,
    test_batch_and_seeds,
    test_prompt_refinement,
    test_controlnet,
    test_style_presets,
    test_upscaling,
    test_quality_gates,
]


def main():
    results = {}
    for suite in SUITES:
        results[suite.__name__] = suite.run()

    print("\n=== Overall summary ===")
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")

    if all(results.values()):
        print("\nAll suites passed.")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"\n{len(failed)} suite(s) failed: {failed}")


if __name__ == "__main__":
    main()
