import torch
from sae_lens import StandardSAE, StandardSAEConfig

from sae_vis._compat import HookedSAETransformer
from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_fetching_fns import get_prompt_data
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.sae_cfg import sae_hook_name
from sae_vis.utils_fns import get_device

# sae_vis moves tokens to `get_device()` (CUDA when available), so build the model and SAE there too
DEVICE = str(get_device())

PROMPT = "Once upon a time, there was a little girl named Lily."
HOOK_NAME = "blocks.0.hook_mlp_out"
FEATURES = list(range(4))


def load_model() -> HookedSAETransformer:
    if hasattr(HookedSAETransformer, "boot_transformers"):  # TransformerLens 4
        model = HookedSAETransformer.boot_transformers(
            "roneneldan/TinyStories-1M", device=DEVICE
        )
        model.enable_compatibility_mode()
        return model
    return HookedSAETransformer.from_pretrained("tiny-stories-1M", device=DEVICE)


def test_prompt_data_tokens_align_with_str_toks():
    """Each prompt token label must be paired with that token's own activations (no BOS shift)."""
    torch.manual_seed(0)
    model = load_model()
    sae_cfg = StandardSAEConfig(d_in=model.cfg.d_model, d_sae=16, device=DEVICE)
    sae_cfg.metadata.hook_name = HOOK_NAME
    sae = StandardSAE(sae_cfg)
    torch.nn.init.normal_(sae.W_enc)

    tokens = torch.randint(0, model.cfg.d_vocab, (16, 32))
    data = SaeVisData.create(
        sae=sae, model=model, tokens=tokens, cfg=SaeVisConfig(features=FEATURES)
    )
    get_prompt_data(data, PROMPT, num_top_features=len(FEATURES))

    str_toks = model.tokenizer.tokenize(PROMPT)
    expected_ids = model.to_tokens(PROMPT, prepend_bos=False)[0]
    assert len(expected_ids) == len(str_toks)
    _, cache = model.run_with_cache_with_saes(expected_ids, saes=[sae])
    expected_acts = cache[sae_hook_name(model, sae)][0]  # [seq d_sae]

    for feature in FEATURES:
        seq_data = data.feature_data_dict[feature]["prompt"][0].seq_data[0]
        assert (
            seq_data.token_ids == expected_ids.tolist()
        ), "prompt tokens don't match `str_toks` (was BOS prepended?)"
        torch.testing.assert_close(
            torch.tensor(seq_data.feat_acts),
            expected_acts[:, feature].cpu(),
            atol=1e-3,
            rtol=0,
        )
