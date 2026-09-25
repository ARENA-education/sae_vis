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


def sae_hook_name(model: Any, sae: Any, suffix: str = "hook_sae_acts_post") -> str:
    """Returns the name of one of the SAE's hook points (e.g. `hook_sae_acts_post`) in the model's activation cache.

    `HookedSAETransformer` names these `{sae.cfg.hook_name}.{suffix}`, e.g. "blocks.0.hook_mlp_out.hook_sae_acts_post".
    `SAETransformerBridge` (TransformerLens 4) attaches the SAE at the model's canonical hook name instead, e.g.
    "blocks.0.mlp.hook_out.hook_sae_acts_post", and the TransformerLens alias doesn't work for SAE hooks, so we ask the
    model when it can tell us.
    """
    if hasattr(model, "get_sae_hook_name"):
        return model.get_sae_hook_name(sae, suffix)
    return f"{sae_cfg_attr(sae, 'hook_name')}.{suffix}"
