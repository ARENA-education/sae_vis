"""Imports that differ between TransformerLens / SAELens versions.

TransformerLens 4 removed `HookedTransformer` (replaced by `TransformerBridge`) and renamed `transformer_lens.utils` to
`transformer_lens.utilities`. SAELens only provides `HookedSAETransformer` when `HookedTransformer` exists; on
TransformerLens 4 its replacement is `SAETransformerBridge`. We alias the new classes to the old names, so the type hints
and `isinstance` checks in this package accept either.
"""

try:  # TransformerLens < 4
    from transformer_lens import HookedTransformer
except (ImportError, AttributeError):
    from transformer_lens.model_bridge import TransformerBridge as HookedTransformer

try:  # TransformerLens < 4
    from transformer_lens import utils
except (ImportError, AttributeError):
    from transformer_lens import utilities as utils

try:
    from sae_lens import HookedSAETransformer
except (ImportError, AttributeError):
    from sae_lens import SAETransformerBridge as HookedSAETransformer

__all__ = ["HookedSAETransformer", "HookedTransformer", "utils"]
