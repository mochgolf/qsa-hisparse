"""Warm the production sampling path before the server reports ready."""

import time

import requests


def wait_for_http_listener(*, url, headers, verify, attempts=120):
    """Wait for the listener without changing the server's readiness state."""
    last_error = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(1)
        try:
            response = requests.get(
                url + "/model_info", headers=headers, timeout=5, verify=verify
            )
            response.raise_for_status()
            return
        except requests.exceptions.RequestException as exc:
            last_error = exc
    raise RuntimeError(
        "HTTP listener did not become available for sampling warmup"
    ) from last_error


def warmup_non_greedy_sampling(*, url, api_key=None, verify=True, timeout=600):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    wait_for_http_listener(url=url, headers=headers, verify=verify)
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": [10, 11, 12],
            "sampling_params": {
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.95,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        },
        headers=headers,
        timeout=timeout,
        verify=verify,
    )
    response.raise_for_status()
    output_ids = response.json().get("output_ids")
    if not isinstance(output_ids, list) or len(output_ids) != 1:
        raise RuntimeError("Non-greedy sampling warmup produced no token")
