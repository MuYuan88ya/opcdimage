"""Compatibility wrapper for the opcdimage trainer."""

from opcdimage_recipe.training.trainer import OPCDImageRayTrainer
from opcdimage_recipe.training.utils import compose_prompt_response_tensors

__all__ = ["OPCDImageRayTrainer", "compose_prompt_response_tensors"]
