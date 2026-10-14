from __future__ import annotations
import math
import torch

def format_metrics(metrics: dict[str, float]) -> str:
    return f"cos={metrics['cosine']:.9f} rel_l2={metrics['relative_l2']:.6e} mse={metrics['mse']:.6e} max_abs={metrics['max_abs']:.6e} row_cos_p01/min={metrics['row_cosine_p01']:.9f}/{metrics['row_cosine_min']:.9f} row_rel_p99/max={metrics['row_relative_l2_p99']:.6e}/{metrics['row_relative_l2_max']:.6e} row_zero_both/single={metrics['row_both_zero_count']:.0f}/{metrics['row_single_zero_count']:.0f}"

def numeric_metrics(actual: torch.Tensor, expected: torch.Tensor, *, actual_dtype: torch.dtype=torch.float32) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f'shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}')
    if actual.device.type != 'cpu' or expected.device.type != 'cpu':
        raise AssertionError(f'comparison tensors must be CPU: actual={actual.device} expected={expected.device}')
    if actual.dtype != actual_dtype or expected.dtype != torch.float32:
        raise AssertionError(f'comparison dtype mismatch: actual={actual.dtype} (expected {actual_dtype}) reference={expected.dtype} (expected torch.float32)')
    if actual.numel() == 0:
        raise AssertionError('comparison tensors must be non-empty')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('comparison tensors contain non-finite values')
    actual64 = actual.reshape(-1).double()
    expected64 = expected.reshape(-1).double()
    error = actual64 - expected64
    actual_norm = actual64.norm()
    expected_norm = expected64.norm()
    if float(expected_norm) == 0.0:
        cosine = 1.0 if float(actual_norm) == 0.0 else 0.0
        relative_l2 = 0.0 if float(actual_norm) == 0.0 else float('inf')
    else:
        cosine = 0.0 if float(actual_norm) == 0.0 else float(actual64 @ expected64 / (actual_norm * expected_norm))
        relative_l2 = float(error.norm() / expected_norm)
    row_width = expected.shape[-1] if expected.ndim else 1
    actual_rows = actual64.reshape(-1, row_width)
    expected_rows = expected64.reshape(-1, row_width)
    error_rows = actual_rows - expected_rows
    actual_row_norm = actual_rows.norm(dim=1)
    expected_row_norm = expected_rows.norm(dim=1)
    actual_row_zero = actual_row_norm == 0
    expected_row_zero = expected_row_norm == 0
    both_zero = actual_row_zero & expected_row_zero
    single_zero = actual_row_zero ^ expected_row_zero
    regular = ~(both_zero | single_zero)
    row_cosine = torch.ones_like(actual_row_norm)
    row_cosine[single_zero] = 0.0
    row_cosine[regular] = (actual_rows[regular] * expected_rows[regular]).sum(dim=1) / (actual_row_norm[regular] * expected_row_norm[regular])
    row_relative_l2 = torch.zeros_like(expected_row_norm)
    expected_nonzero = ~expected_row_zero
    row_relative_l2[expected_nonzero] = error_rows[expected_nonzero].norm(dim=1) / expected_row_norm[expected_nonzero]
    row_relative_l2[single_zero & expected_row_zero] = float('inf')
    return {'cosine': cosine, 'relative_l2': relative_l2, 'mse': float(error.square().mean()), 'max_abs': float(error.abs().max()), 'row_cosine_p01': float(torch.quantile(row_cosine, 0.01)), 'row_cosine_min': float(row_cosine.min()), 'row_relative_l2_p99': float(torch.quantile(row_relative_l2, 0.99)), 'row_relative_l2_max': float(row_relative_l2.max()), 'row_both_zero_count': float(both_zero.sum()), 'row_single_zero_count': float(single_zero.sum())}

def within_budget(metrics: dict[str, float], *, cosine_min: float | None, relative_l2_max: float, row_relative_l2_max: float) -> bool:
    if cosine_min is not None and (not math.isfinite(cosine_min) or not -1.0 <= cosine_min <= 1.0):
        raise ValueError('cosine_min must be finite in [-1, 1]')
    if not math.isfinite(relative_l2_max) or relative_l2_max < 0.0:
        raise ValueError('relative_l2_max must be finite and non-negative')
    if not math.isfinite(row_relative_l2_max) or row_relative_l2_max < 0.0:
        raise ValueError('row_relative_l2_max must be finite and non-negative')
    return math.isfinite(metrics['cosine']) and math.isfinite(metrics['relative_l2']) and math.isfinite(metrics['row_relative_l2_max']) and (cosine_min is None or metrics['cosine'] >= cosine_min) and (metrics['relative_l2'] <= relative_l2_max) and (metrics['row_relative_l2_max'] <= row_relative_l2_max) and (metrics['row_single_zero_count'] == 0.0)

def require_same_semantic(actual: torch.Tensor, expected: torch.Tensor, *, cosine_min: float | None, relative_l2_max: float, row_relative_l2_max: float, label: str, actual_dtype: torch.dtype=torch.float32) -> dict[str, float]:
    metrics = numeric_metrics(actual, expected, actual_dtype=actual_dtype)
    if not within_budget(metrics, cosine_min=cosine_min, relative_l2_max=relative_l2_max, row_relative_l2_max=row_relative_l2_max):
        raise AssertionError(f"{label}: cosine={metrics['cosine']:.9f} (min={cosine_min}) relative_l2={metrics['relative_l2']:.6e} (max={relative_l2_max}) mse={metrics['mse']:.6e} max_abs={metrics['max_abs']:.6e} row_cos_p01/min={metrics['row_cosine_p01']:.9f}/{metrics['row_cosine_min']:.9f} row_rel_l2_p99/max={metrics['row_relative_l2_p99']:.6e}/{metrics['row_relative_l2_max']:.6e} (max={row_relative_l2_max}) row_single_zero_count={metrics['row_single_zero_count']:.0f}")
    return metrics
