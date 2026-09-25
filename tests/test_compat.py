import sae_lens
import transformer_lens

TL_HAS_HOOKED_TRANSFORMER = hasattr(transformer_lens, "HookedTransformer")


def test_import_sae_vis():
    """`import sae_vis` works on TransformerLens 2.x-4.x (TL 4 removed `HookedTransformer`)."""
    import sae_vis

    assert sae_vis.SaeVisData is not None


def test_compat_aliases():
    from sae_vis._compat import HookedSAETransformer, HookedTransformer

    if TL_HAS_HOOKED_TRANSFORMER:
        assert HookedTransformer is transformer_lens.HookedTransformer
        assert HookedSAETransformer is sae_lens.HookedSAETransformer
    else:
        from transformer_lens.model_bridge import TransformerBridge

        assert HookedTransformer is TransformerBridge
        assert HookedSAETransformer is sae_lens.SAETransformerBridge
    assert issubclass(HookedSAETransformer, HookedTransformer)


def test_utils_get_act_name():
    from sae_vis._compat import utils

    assert utils.get_act_name("resid_post", 3) == "blocks.3.hook_resid_post"
    assert utils.get_act_name("pattern", 0) == "blocks.0.attn.hook_pattern"
