"""Model variants used in the paper (each trained from scratch with the same recipe).

otarf               OTARF as described in the paper (local base forecast + support-gated
                    attention correction).
no_anchor           history fed as the filled standardized series instead of R = M (Z - b);
                    the anchor value is removed from the base and from every feature.
no_cross            cross-variable correction fixed at zero.
mean_pool           uniform mean over the same valid source segments instead of attention.
state_blind         mask, age and anchor validity replaced by constants at the input.
state_blind_no_aug  as state_blind, trained on natural histories only.
"""

from .core import OTARFCore

ABLATIONS = ("no_anchor", "no_cross", "mean_pool", "state_blind", "state_blind_no_aug")
VARIANTS = ("otarf",) + ABLATIONS


def build_model(variant, channels, horizon, config):
    if variant == "otarf":
        return OTARFCore(channels, horizon, config, "core")
    if variant in ABLATIONS:
        return OTARFCore(channels, horizon, config, variant)
    raise ValueError("Unknown variant " + variant)
