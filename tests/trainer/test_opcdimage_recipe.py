from pathlib import Path

import torch
from omegaconf import OmegaConf

from opcdimage_recipe.core import build_crop_messages_from_raw_prompt
from opcdimage_recipe.dp_actor import compute_reverse_kl_loss
from opcdimage_recipe.ray_trainer import compose_prompt_response_tensors
from opcdimage_recipe.training.utils import load_opcdimage_reward_manager
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.workers.reward_manager.naive import NaiveRewardManager


ROOT = Path(__file__).resolve().parents[2]


def test_reverse_kl_loss_matches_current_formula():
    log_prob = torch.log(
        torch.tensor(
            [
                [[0.60, 0.40], [0.30, 0.70]],
                [[0.25, 0.75], [0.80, 0.20]],
            ],
            dtype=torch.float32,
        )
    )
    exp_log_prob = torch.log(
        torch.tensor(
            [
                [[0.55, 0.45], [0.35, 0.65]],
                [[0.40, 0.60], [0.70, 0.30]],
            ],
            dtype=torch.float32,
        )
    )
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.int64)

    loss, kld = compute_reverse_kl_loss(
        log_prob=log_prob,
        exp_log_prob=exp_log_prob,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        on_policy_merge=True,
        kl_loss_type="full",
        kl_renorm_topk=True,
    )

    expected_kld = kl_penalty(
        logprob=log_prob,
        ref_logprob=exp_log_prob,
        kl_penalty="full",
        kl_renorm_topk=True,
    )
    expected_loss = agg_loss(loss_mat=expected_kld, loss_mask=response_mask, loss_agg_mode="token-mean")

    assert torch.allclose(kld, expected_kld)
    assert torch.allclose(loss, expected_loss)


def test_compose_prompt_response_tensors_reuses_same_response():
    prompt_input_ids = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
    prompt_attention_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    prompt_position_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    responses = torch.tensor([[30, 31, 32], [40, 2, 0]], dtype=torch.long)

    composed = compose_prompt_response_tensors(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        prompt_position_ids=prompt_position_ids,
        responses=responses,
        eos_token_id=2,
    )

    assert torch.equal(composed["prompts"], prompt_input_ids)
    assert torch.equal(composed["responses"], responses)
    assert torch.equal(composed["input_ids"], torch.tensor([[10, 11, 30, 31, 32], [20, 21, 40, 2, 0]]))
    assert torch.equal(composed["attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]))
    assert torch.equal(composed["position_ids"], torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]))


def test_build_crop_messages_replaces_single_image_only():
    raw_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "original.png"},
                {"type": "text", "text": "question"},
            ],
        }
    ]

    crop_messages = build_crop_messages_from_raw_prompt(raw_prompt, "crop.png")
    assert crop_messages[0]["content"][0]["image"] == "crop.png"
    assert raw_prompt[0]["content"][0]["image"] == "original.png"


def test_reward_manager_loader_honors_reward_config():
    class DummyTokenizer:
        pass

    config = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": str(ROOT / "opcdimage_recipe" / "reward_fn.py"),
                    "name": "compute_score",
                },
                "reward_manager": {
                    "source": "register",
                    "name": "naive",
                },
            },
            "data": {"reward_fn_key": "data_source"},
        }
    )

    reward_manager = load_opcdimage_reward_manager(config, DummyTokenizer(), num_examine=1)

    assert isinstance(reward_manager, NaiveRewardManager)
    assert reward_manager.num_examine == 1


def test_framework_files_match_upstream_vendor_copy():
    pairs = [
        (
            ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py",
            ROOT / "third_party" / "verl_upstream" / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        ),
        (
            ROOT / "verl" / "workers" / "actor" / "dp_actor.py",
            ROOT / "third_party" / "verl_upstream" / "verl" / "workers" / "actor" / "dp_actor.py",
        ),
        (
            ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml",
            ROOT / "third_party" / "verl_upstream" / "verl" / "trainer" / "config" / "ppo_trainer.yaml",
        ),
    ]
    for local_file, upstream_file in pairs:
        assert local_file.read_bytes() == upstream_file.read_bytes(), f"{local_file} drifted from upstream."
