from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge b into a (returns new dict)."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
