import os
import requests

BASE_URL = os.getenv("SMOKE_TEST_BASE_URL", "http://localhost:5000")

def check(failures, name, condition, detail=""):
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        failures.append(name)


def post_generate(payload, timeout=300):
    return requests.post(f"{BASE_URL}/generate", json=payload, timeout=timeout)


def get(path, timeout=10):
    return requests.get(f"{BASE_URL}{path}", timeout=timeout)


def summarize(name, failures):
    print(f"\n=== {name}: {len(failures)} failed ===")
    if failures:
        print(failures)
    return len(failures) == 0
