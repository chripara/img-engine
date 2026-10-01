from smoke_tests._client import check, get, summarize


def run():
    failures = []

    r = get("/health")
    check(failures, "health endpoint returns 200", r.status_code == 200, f"status={r.status_code}")
    check(failures, "health endpoint reports ok", r.status_code == 200 and r.json().get("status") == "ok")

    r = get("/docs/swagger")
    check(failures, "swagger docs reachable", r.status_code == 200, f"status={r.status_code}")

    r = get("/docs/redoc")
    check(failures, "redoc docs reachable", r.status_code == 200, f"status={r.status_code}")

    r = get("/docs/openapi.json")
    ok = r.status_code == 200
    if ok:
        try:
            r.json()
        except ValueError:
            ok = False
    check(failures, "openapi.json reachable and valid JSON", ok, f"status={r.status_code}")

    return summarize("test_health", failures)


if __name__ == "__main__":
    run()
