"""Compatibility wrapper for the opcdimage actor."""

from opcdimage_recipe.training.actor import DataParallelPPOActor, compute_reverse_kl_loss

__all__ = ["DataParallelPPOActor", "compute_reverse_kl_loss"]
