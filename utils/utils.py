from typing import Dict


def format_metrics(metrics: Dict[str, float]) -> str:
    format_str = f'loss={metrics["loss"]:.05f}, F1={metrics["F1"]:.03f}, P={metrics["P"]:.03f}, R={metrics["R"]:.03f}'
    for k in metrics:
        if k == "loss" or k == "F1" or k == "P" or k == "R":
            continue
        format_str += f', {k}={metrics[k]:.03f}'
    return format_str
