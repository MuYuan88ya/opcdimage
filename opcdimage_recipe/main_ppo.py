"""Compatibility wrapper for the opcdimage PPO entrypoint."""

from opcdimage_recipe.training.main_ppo import OPCDImageTaskRunner, main

__all__ = ["OPCDImageTaskRunner", "main"]


if __name__ == "__main__":
    main()
