"""Training-specific implementation for the opcdimage recipe."""

from opcdimage_recipe.training.actor import DataParallelPPOActor, compute_reverse_kl_loss
from opcdimage_recipe.training.trainer import OPCDImageRayTrainer
from opcdimage_recipe.training.utils import compose_prompt_response_tensors

__all__ = ["DataParallelPPOActor", "OPCDImageRayTrainer", "compose_prompt_response_tensors", "compute_reverse_kl_loss"]
