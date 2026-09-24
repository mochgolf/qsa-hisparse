"""Warm the production sampling path before the server reports ready."""

import requests


def warmup_non_greedy_sampling(*, url, api_key=None, verify=True, timeout=600):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
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
