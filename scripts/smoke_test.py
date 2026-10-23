"""Smoke-test a running API: the model must be loaded and /predict must score a payload.

Fails (non-zero exit) if /health never reports ``model_loaded: true`` within the timeout,
or if /predict does not return a valid probability. Used by CI, CD and docker-compose docs.

  python scripts/smoke_test.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PAYLOAD = {"features": {"loan_amnt": 15000.0, "int_rate": 11.5, "annual_inc": 62000.0,
                        "dti": 18.2, "emp_length": 5, "home_ownership": "RENT"},
           "request_id": "smoke-test"}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--timeout", type=float, default=60)
    args = p.parse_args()
    deadline, health = time.monotonic() + args.timeout, None
    while time.monotonic() < deadline:
        try:
            health = _get(f"{args.url}/health")
            if health.get("model_loaded"):
                break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    print("health:", json.dumps(health))
    if not health or not health.get("model_loaded"):
        print("FAIL: model not loaded" + (f" ({health.get('reason')})" if health else ""))
        return 1
    request = urllib.request.Request(f"{args.url}/predict", data=json.dumps(PAYLOAD).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"FAIL: /predict returned {exc.code}: {exc.read().decode()[:300]}")
        return 1
    print("predict:", json.dumps(body))
    pd_value = body.get("probability_of_default")
    if not isinstance(pd_value, float) or not 0 <= pd_value <= 1:
        print("FAIL: /predict did not return a probability")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
