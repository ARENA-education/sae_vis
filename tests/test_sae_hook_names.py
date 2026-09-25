from pathlib import Path
from types import SimpleNamespace

import torch
from sae_lens import SAETransformerBridge, StandardSAE, StandardSAEConfig

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

HOOK_NAME = "blocks.0.hook_mlp_out"
FEATURES = list(range(4))


def test_feature_centric_vis_with_sae_transformer_bridge(tmp_path: Path):
    """`SAETransformerBridge` caches SAE activations under its canonical hook name (e.g.
    `blocks.0.mlp.hook_out.hook_sae_acts_post`), not `{sae.cfg.hook_name}.hook_sae_acts_post`."""
    torch.manual_seed(0)
    model = SAETransformerBridge.boot_transformers(
        "roneneldan/TinyStories-1M", device="cpu"
    )
    model.enable_compatibility_mode()
    sae_cfg = StandardSAEConfig(d_in=model.cfg.d_model, d_sae=16, device="cpu")
    sae_cfg.metadata.hook_name = HOOK_NAME
    sae = StandardSAE(sae_cfg)
    torch.nn.init.normal_(sae.W_enc)

    tokens = torch.randint(0, model.cfg.d_vocab, (16, 32))
    data = SaeVisData.create(
        sae=sae, model=model, tokens=tokens, cfg=SaeVisConfig(features=FEATURES)
    )
    assert set(data.feature_data_dict.keys()) == set(FEATURES)

    save_path = tmp_path / "feature_centric_vis.html"
    data.save_feature_centric_vis(save_path)
    assert save_path.exists()


def test_sae_hook_name_falls_back_to_cfg_hook_name():
    """Models without `get_sae_hook_name` (e.g. `HookedSAETransformer`) keep the `{hook_name}.{suffix}` naming."""
    from sae_vis.sae_cfg import sae_hook_name

    sae_cfg = StandardSAEConfig(d_in=8, d_sae=16, device="cpu")
    sae_cfg.metadata.hook_name = HOOK_NAME
    sae = StandardSAE(sae_cfg)
    model = SimpleNamespace()
    assert sae_hook_name(model, sae) == f"{HOOK_NAME}.hook_sae_acts_post"
    assert sae_hook_name(model, sae, "hook_sae_input") == f"{HOOK_NAME}.hook_sae_input"
