"""Trainer-native Vision-OPCD recipe for opcdimage."""

from opcdimage_recipe.training import (
    DataParallelPPOActor,
    OPCDImageRayTrainer,
    compose_prompt_response_tensors,
    compute_reverse_kl_loss,
)

__all__ = ["DataParallelPPOActor", "OPCDImageRayTrainer", "compose_prompt_response_tensors", "compute_reverse_kl_loss"]

