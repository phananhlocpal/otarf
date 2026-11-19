"""Window-normalization switch for the mechanism test (paper, Sect. 5.1 and 6.2).

The official model classes are not edited. SOFTS and iTransformer expose the official flag ``use_norm``, which is set
to 0 before the model is built. TimeFilter (Normalize, always on) and CRIB (RevIN) have no flag, so their normalization
module is replaced on the constructed instance by an identity module with the same call signature ``module(x, mode)``.
Every other layer, argument and weight initialisation is unchanged.
"""
import torch


class IdentityNorm(torch.nn.Module):
    """Same interface as the official window-normalization modules; returns the input unchanged."""

    def forward(self, x, mode=None):
        return x


def disable_window_norm(name, official):
    """Applies the switch to an already constructed official model and returns a short description."""
    if name in ("SOFTS", "iTransformer"):
        if getattr(official, "use_norm", None) not in (0, False):
            raise ValueError(f"{name}: use_norm must already be 0 from the official argument")
        return "official flag use_norm=0"
    if name == "TimeFilter":
        if not hasattr(official, "norm"):
            raise ValueError("TimeFilter instance has no norm module")
        official.norm = IdentityNorm()
        return "instance attribute norm (Normalize) replaced by identity"
    if name == "CRIB":
        if not hasattr(official, "revinlayer"):
            raise ValueError("CRIB instance has no revinlayer")
        official.revinlayer = IdentityNorm()
        return "instance attribute revinlayer (RevIN) replaced by identity"
    raise ValueError("Unsupported model " + str(name))
