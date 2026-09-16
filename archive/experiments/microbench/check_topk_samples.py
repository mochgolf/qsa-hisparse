#!/usr/bin/env python3
"""CPU-only independent checker for QSA fast-top-k sentinel samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


SAMPLE_STEPS = (0, 1, 2, 3, 127, 255, 511, 766)
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
TOPK = 512


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("expected a qsa-topk-v1 payload with a samples list")
    return payload


def _row_check(sample: dict[str, Any]) -> dict[str, Any]:
    required = {"layer_id", "step", "compressed_length", "logits", "block_indices"}
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"sample is missing fields: {missing}")
    logits = sample["logits"]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    logits = logits.detach().cpu().reshape(-1)
    length = int(sample["compressed_length"])
    if length != logits.numel():
        raise ValueError(
            f"layer={sample['layer_id']} step={sample['step']}: "
            f"logits length {logits.numel()} != compressed_length {length}"
        )
    # Check the same values with both CPU torch and NumPy; no candidate kernel
    # or tolerance participates in this oracle.
    torch_finite = bool(torch.isfinite(logits).all().item())
    values = logits.numpy()
    numpy_finite = bool(np.isfinite(values).all())
    if not torch_finite or not numpy_finite:
        raise ValueError(
            f"layer={sample['layer_id']} step={sample['step']}: non-finite logits"
        )

    selected = sample["block_indices"]
    if isinstance(selected, torch.Tensor):
        selected = selected.detach().cpu().reshape(-1).numpy()
    else:
        selected = np.asarray(selected)
    if selected.shape != (TOPK,):
        raise ValueError(
            f"layer={sample['layer_id']} step={sample['step']}: "
            f"expected {TOPK} selected IDs, got {selected.shape}"
        )
    if not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected block IDs must be integer")
    selected = selected.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.any(selected >= length):
        raise ValueError(
            f"layer={sample['layer_id']} step={sample['step']}: selected ID out of range"
        )
    unique = np.unique(selected)
    if unique.size != TOPK:
        raise ValueError(
            f"layer={sample['layer_id']} step={sample['step']}: "
            f"selected IDs are not unique ({unique.size}/{TOPK})"
        )
    selected_values = values[selected]
    unselected_mask = np.ones(length, dtype=bool)
    unselected_mask[unique] = False
    unselected_values = values[unselected_mask]
    selected_min = float(selected_values.min())
    unselected_max = (
        None if unselected_values.size == 0 else float(unselected_values.max())
    )
    threshold_ok = unselected_max is None or selected_min >= unselected_max
    if not threshold_ok:
        raise ValueError(
            f"layer={sample['layer_id']} step={sample['step']}: "
            f"selected_min={selected_min} < unselected_max={unselected_max}"
        )
    return {
        "layer_id": int(sample["layer_id"]),
        "step": int(sample["step"]),
        "position": int(sample.get("position", -1)),
        "compressed_length": length,
        "selected_count": int(unique.size),
        "unselected_count": int(unselected_values.size),
        "selected_min": selected_min,
        "unselected_max": unselected_max,
        "finite": True,
        "threshold_ok": True,
    }


def check_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "qsa-topk-v1":
        raise ValueError(f"unexpected schema_version={payload.get('schema_version')!r}")
    if tuple(payload.get("sample_steps", ())) != SAMPLE_STEPS:
        raise ValueError("sample_steps do not match the fixed QSA sentinel")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    seen: set[tuple[int, int]] = set()
    for sample in payload["samples"]:
        if not isinstance(sample, dict):
            failures.append("sample is not an object")
            continue
        try:
            key = (int(sample.get("layer_id", -1)), int(sample.get("step", -1)))
        except (TypeError, ValueError):
            failures.append("sample has non-integer layer_id/step")
            continue
        if key in seen:
            failures.append(f"duplicate layer/step sample: {key}")
            continue
        seen.add(key)
        try:
            rows.append(_row_check(sample))
        except (TypeError, ValueError, RuntimeError) as exc:
            failures.append(str(exc))
    layer_ids = sorted({layer for layer, _step in seen})
    if tuple(layer_ids) != LAYERS:
        failures.append(f"expected QSA layers {LAYERS}, got {tuple(layer_ids)}")
    expected_keys = {(layer, step) for layer in LAYERS for step in SAMPLE_STEPS}
    missing = sorted(expected_keys - seen)
    if missing:
        failures.append(f"missing layer/step samples: {missing[:8]}")
    unexpected = sorted(seen - expected_keys)
    if unexpected:
        failures.append(f"unexpected layer/step samples: {unexpected[:8]}")
    return {
        "schema_version": "qsa-topk-check-v1",
        "source_schema_version": payload.get("schema_version"),
        "sample_steps": list(SAMPLE_STEPS),
        "layer_ids": layer_ids,
        "layer_count": len(layer_ids),
        "sample_count": len(payload["samples"]),
        "expected_sample_count": len(LAYERS) * len(SAMPLE_STEPS),
        "passed": (
            not failures
            and len(rows) == len(payload["samples"]) == len(expected_keys)
        ),
        "rows": rows,
        "failures": failures,
    }


def _self_test_payload() -> dict[str, Any]:
    samples = []
    logits = torch.arange(TOPK + 64, dtype=torch.float32)
    for layer_id in LAYERS:
        for step in SAMPLE_STEPS:
            samples.append(
                {
                    "layer_id": layer_id,
                    "step": step,
                    "position": 262000 + step,
                    "compressed_length": logits.numel(),
                    "logits": logits.clone(),
                    "block_indices": torch.arange(64, TOPK + 64, dtype=torch.int32),
                }
            )
    return {
        "schema_version": "qsa-topk-v1",
        "sample_steps": list(SAMPLE_STEPS),
        "samples": samples,
    }


def _self_test() -> dict[str, Any]:
    payload = _self_test_payload()
    result = check_payload(payload)
    assert result["passed"]

    tie_payload = _self_test_payload()
    for sample in tie_payload["samples"]:
        sample["logits"].zero_()
    tie_result = check_payload(tie_payload)
    assert tie_result["passed"]

    low_selected = _self_test_payload()
    low_selected["samples"][0]["logits"][511] = -1.0
    low_result = check_payload(low_selected)
    assert not low_result["passed"]

    duplicate = _self_test_payload()
    duplicate["samples"][1]["step"] = duplicate["samples"][0]["step"]
    duplicate_result = check_payload(duplicate)
    assert not duplicate_result["passed"]

    missing_layer = _self_test_payload()
    missing_layer["samples"] = [
        sample for sample in missing_layer["samples"] if sample["layer_id"] != LAYERS[-1]
    ]
    missing_result = check_payload(missing_layer)
    assert not missing_result["passed"]
    result["self_test_negative_checks"] = {
        "boundary_tie_passes": tie_result["passed"],
        "low_selected_fails": not low_result["passed"],
        "duplicate_fails": not duplicate_result["passed"],
        "missing_layer_fails": not missing_result["passed"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _self_test()
    elif args.input is not None:
        payload = _load(args.input)
        result = check_payload(payload)
    else:
        parser.error("provide input or --self-test")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)
    print(json.dumps({key: result[key] for key in ("passed", "layer_count", "sample_count")}))


if __name__ == "__main__":
    main()
