"""Reading SAE config fields across SAELens versions.

SAELens >= 6 moved `hook_name`, `hook_layer`, `context_size`, `dataset_path` and `prepend_bos` from `sae.cfg` onto
`sae.cfg.metadata`, so `sae.cfg.hook_name` raises an AttributeError there. Use `sae_cfg_attr` instead.
"""

import re
from typing import Any


def sae_cfg_attr(sae: Any, name: str) -> Any:
    """Returns `sae.cfg.<name>`, falling back to `sae.cfg.metadata.<name>` (SAELens >= 6)."""
    cfg = sae.cfg
    if hasattr(cfg, name):
        return getattr(cfg, name)

    metadata = getattr(cfg, "metadata", None)
    value = getattr(metadata, name, None)

    # SAELens >= 6 doesn't always record the layer; recover it from the hook name, e.g. "blocks.7.hook_resid_pre"
    if value is None and name == "hook_layer":
        match = re.search(r"blocks\.(\d+)\.", sae_cfg_attr(sae, "hook_name"))
        if match:
            return int(match.group(1))

    if not hasattr(metadata, name):
        raise AttributeError(
            f"SAE config has no field {name!r} (checked both `sae.cfg` and `sae.cfg.metadata`)"
        )
    return value
