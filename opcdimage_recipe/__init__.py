"""Trainer-native Vision-OPCD recipe for opcdimage."""

from opcdimage_recipe.dp_actor import DataParallelPPOActor, compute_reverse_kl_loss
from opcdimage_recipe.ray_trainer import OPCDImageRayTrainer, compose_prompt_response_tensors

__all__ = ["DataParallelPPOActor", "OPCDImageRayTrainer", "compose_prompt_response_tensors", "compute_reverse_kl_loss"]

