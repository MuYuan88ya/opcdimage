from __future__ import annotations

import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from opcdimage_recipe.core import build_crop_messages_from_raw_prompt
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_extern_object
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import get_response_mask
from verl.utils.tracking import Tracking
from verl.workers.reward_manager import get_reward_manager_cls


def get_opcdimage_options(config) -> dict[str, object]:
    privileged_mode = OmegaConf.select(config, "opcdimage.privileged_mode") or "crop"
    on_policy_merge = OmegaConf.select(config, "opcdimage.on_policy_merge")
    if on_policy_merge is None:
        on_policy_merge = True
    return {"privileged_mode": privileged_mode, "on_policy_merge": bool(on_policy_merge)}


def load_opcdimage_reward_manager(config, tokenizer, num_examine: int = 0):
    reward_manager_source = OmegaConf.select(config, "reward.reward_manager.source") or "register"
    reward_manager_name = OmegaConf.select(config, "reward.reward_manager.name") or "naive"

    if reward_manager_source == "register":
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    elif reward_manager_source == "importlib":
        module_path = OmegaConf.select(config, "reward.reward_manager.module.path")
        module_name = OmegaConf.select(config, "reward.reward_manager.module.name") or reward_manager_name
        if not module_path:
            raise ValueError("reward.reward_manager.module.path must be set when source=importlib.")
        reward_manager_cls = load_extern_object(module_path, module_name)
    else:
        raise ValueError(f"Unsupported reward manager source: {reward_manager_source}")

    reward_fn_path = OmegaConf.select(config, "reward.custom_reward_function.path")
    reward_fn_name = OmegaConf.select(config, "reward.custom_reward_function.name") or "compute_score"
    if not reward_fn_path:
        raise ValueError("opcdimage recipe requires reward.custom_reward_function.path to be set.")
    compute_score = load_extern_object(reward_fn_path, reward_fn_name)

    reward_kwargs_cfg = OmegaConf.select(config, "reward.reward_kwargs")
    reward_kwargs = (
        dict(OmegaConf.to_container(reward_kwargs_cfg, resolve=True)) if reward_kwargs_cfg is not None else {}
    )

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compose_prompt_response_tensors(
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    responses: torch.Tensor,
    eos_token_id: int,
) -> dict[str, torch.Tensor]:
    response_attention_mask = get_response_mask(
        response_id=responses,
        eos_token=eos_token_id,
        dtype=prompt_attention_mask.dtype,
    )
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    sequence = torch.cat([prompt_input_ids, responses], dim=-1)

    response_length = responses.size(1)
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(prompt_input_ids.size(0), -1)
    if prompt_position_ids.dim() == 3:
        delta_position_id = delta_position_id.view(prompt_input_ids.size(0), 1, -1).expand(
            prompt_input_ids.size(0), 3, -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

    return {
        "prompts": prompt_input_ids,
        "responses": responses,
        "input_ids": sequence,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def to_python_list(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    return value


class OPCDImageRayTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opcdimage_options = get_opcdimage_options(self.config)
        self.reward_fn = load_opcdimage_reward_manager(self.config, self.tokenizer, num_examine=0)
        self.val_reward_fn = load_opcdimage_reward_manager(self.config, self.tokenizer, num_examine=1)
        if self.opcdimage_options["privileged_mode"] != "crop":
            raise NotImplementedError("opcdimage recipe currently only supports crop privileged training.")
        if not self.opcdimage_options["on_policy_merge"]:
            raise NotImplementedError("opcdimage recipe only supports on-policy response reuse.")
        if self.config.actor_rollout_ref.actor.kl_loss_type == "seqkd":
            raise NotImplementedError("opcdimage recipe does not support seqkd distillation.")

    def _build_privileged_gen_batch(self, gen_batch: DataProto) -> DataProto:
        privileged_gen_batch = gen_batch.select(deepcopy=True)
        updated_gen_inputs = []
        for idx in range(len(privileged_gen_batch)):
            crop_images = to_python_list(privileged_gen_batch.non_tensor_batch["crop_images"][idx])
            if not isinstance(crop_images, (list, tuple)) or len(crop_images) != 1:
                raise ValueError("Each sample must contain exactly one crop image.")
            messages = build_crop_messages_from_raw_prompt(
                privileged_gen_batch.non_tensor_batch["raw_prompt"][idx],
                crop_image=crop_images[0],
            )
            updated_gen_inputs.append(self.train_dataset.re_tokenize(messages))

        privileged_gen_batch.batch["input_ids"] = torch.stack([inp["input_ids"] for inp in updated_gen_inputs])
        privileged_gen_batch.batch["attention_mask"] = torch.stack([inp["attention_mask"] for inp in updated_gen_inputs])
        privileged_gen_batch.batch["position_ids"] = torch.stack([inp["position_ids"] for inp in updated_gen_inputs])

        updated_multi_modal_inputs = [inp.get("multi_modal_inputs") for inp in updated_gen_inputs]
        if any(mmi is not None for mmi in updated_multi_modal_inputs):
            privileged_gen_batch.non_tensor_batch["multi_modal_inputs"] = np.array(updated_multi_modal_inputs, dtype=object)
        else:
            privileged_gen_batch.non_tensor_batch.pop("multi_modal_inputs", None)
        return privileged_gen_batch

    def _build_training_batch(
        self,
        *,
        source_batch: DataProto,
        prompt_batch: DataProto,
        rollout_output: DataProto,
    ) -> DataProto:
        repeat_times = self.config.actor_rollout_ref.rollout.n
        train_batch = source_batch.select(deepcopy=True)
        train_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(train_batch))], dtype=object)
        train_batch = train_batch.repeat(repeat_times=repeat_times, interleave=True)
        repeated_prompt_batch = prompt_batch.repeat(repeat_times=repeat_times, interleave=True)

        composed = compose_prompt_response_tensors(
            prompt_input_ids=repeated_prompt_batch.batch["input_ids"],
            prompt_attention_mask=repeated_prompt_batch.batch["attention_mask"],
            prompt_position_ids=repeated_prompt_batch.batch["position_ids"],
            responses=rollout_output.batch["responses"],
            eos_token_id=source_batch.meta_info.get("eos_token_id", self.tokenizer.eos_token_id),
        )
        for key, value in composed.items():
            train_batch.batch[key] = value
        train_batch.batch["response_mask"] = compute_response_mask(train_batch)

        if "multi_modal_inputs" in repeated_prompt_batch.non_tensor_batch:
            train_batch.non_tensor_batch["multi_modal_inputs"] = np.array(
                repeated_prompt_batch.non_tensor_batch["multi_modal_inputs"],
                dtype=object,
            )
        else:
            train_batch.non_tensor_batch.pop("multi_modal_inputs", None)
        return train_batch

    def _compute_topk_support(self, student_batch: DataProto, privileged_batch: DataProto, timing_raw: dict) -> None:
        actor_cfg = self.config.actor_rollout_ref.actor
        if actor_cfg.kl_loss_type != "full" or actor_cfg.kl_topk <= 0:
            return

        student_batch.meta_info["return_all_logits"] = True
        privileged_batch.meta_info["return_all_logits"] = True
        with marked_timer("compute_topk_indices", timing_raw, color="purple"):
            log_prob_proto = self.actor_rollout_wg.compute_log_prob(student_batch)
            actor_topk_indices = log_prob_proto.batch["old_log_probs"].long()

            if actor_cfg.get("kl_merge_indice", False):
                privileged_batch.batch["first_kl_topk_indices"] = actor_topk_indices
                ref_log_prob_for_topk = self.ref_policy_wg.compute_ref_log_prob(privileged_batch)
                privileged_batch.batch["kl_topk_indices"] = ref_log_prob_for_topk.batch["ref_log_prob"].long()
                privileged_batch.batch.pop("first_kl_topk_indices")
            else:
                privileged_batch.batch["kl_topk_indices"] = actor_topk_indices

    def _compute_privileged_scores(
        self,
        *,
        student_batch: DataProto,
        privileged_batch: DataProto,
        timing_raw: dict,
    ) -> DataProto:
        with marked_timer("exp_log_prob", timing_raw, color="olive"):
            privileged_batch.meta_info["return_all_logits"] = self.config.actor_rollout_ref.actor.kl_loss_type == "full"
            exp_log_prob = self.ref_policy_wg.compute_ref_log_prob(privileged_batch)
            exp_log_prob.batch["exp_log_probs"] = exp_log_prob.batch.pop("ref_log_prob")
            if "kl_topk_indices" in privileged_batch.batch:
                exp_log_prob.batch["kl_topk_indices"] = privileged_batch.batch["kl_topk_indices"]
            return student_batch.union(exp_log_prob)

    def _compute_reward_metrics(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, list]]:
        reward_result = self.reward_fn(batch, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_info = reward_result.get("reward_extra_info", {})
        return reward_tensor, reward_extra_info

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_info = reward_result.get("reward_extra_info", {})

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if "timing" in gen_batch_output.meta_info:
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                    privileged_gen_batch = self._build_privileged_gen_batch(gen_batch)
                    student_batch = self._build_training_batch(
                        source_batch=batch,
                        prompt_batch=gen_batch,
                        rollout_output=gen_batch_output,
                    )
                    privileged_batch = self._build_training_batch(
                        source_batch=batch,
                        prompt_batch=privileged_gen_batch,
                        rollout_output=gen_batch_output,
                    )

                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, _ = self._compute_reward_metrics(student_batch)
                        scores = reward_tensor.sum(dim=-1).float()
                        metrics["actor/curr_acc"] = scores.mean().item()

                    self._compute_topk_support(student_batch, privileged_batch, timing_raw)
                    student_batch = self._compute_privileged_scores(
                        student_batch=student_batch,
                        privileged_batch=privileged_batch,
                        timing_raw=timing_raw,
                    )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(student_batch, metrics=metrics)

                    student_batch.meta_info["global_token_num"] = torch.sum(
                        student_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with marked_timer("update_actor", timing_raw, color="red"):
                        student_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        student_batch.meta_info["stage_merge"] = True
                        student_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        student_batch.meta_info["on_policy_merge"] = self.opcdimage_options["on_policy_merge"]
                        actor_output = self.actor_rollout_wg.update_actor(student_batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(self.global_steps)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        metrics.update(val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics

                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                metrics["timing/step"] = timing_raw.get("step", 0.0)
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    progress_bar.close()
                    if last_val_metrics is not None:
                        print(f"Final validation metrics: {last_val_metrics}")
                    return


__all__ = [
    "OPCDImageRayTrainer",
    "compose_prompt_response_tensors",
    "get_opcdimage_options",
    "load_opcdimage_reward_manager",
]
