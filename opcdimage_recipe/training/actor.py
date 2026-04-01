from __future__ import annotations

import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor.dp_actor import DataParallelPPOActor as UpstreamDataParallelPPOActor


def compute_reverse_kl_loss(
    *,
    log_prob: torch.Tensor,
    exp_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    on_policy_merge: bool,
    kl_loss_type: str,
    kl_renorm_topk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if on_policy_merge:
        kld = kl_penalty(
            logprob=log_prob,
            ref_logprob=exp_log_prob,
            kl_penalty=kl_loss_type,
            kl_renorm_topk=kl_renorm_topk,
        )
    else:
        kld = kl_penalty(
            logprob=exp_log_prob,
            ref_logprob=log_prob,
            kl_penalty=kl_loss_type,
            kl_renorm_topk=kl_renorm_topk,
        )
    return agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode), kld


class DataParallelPPOActor(UpstreamDataParallelPPOActor):
    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not return_all_logits:
            return super()._forward_micro_batch(
                micro_batch=micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    raise NotImplementedError("return_all_logits=True is not supported with fused kernels in opcdimage.")

                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)
                log_probs = torch.nn.functional.log_softmax(logits_rmpad, dim=-1)

                if calculate_entropy:
                    entropy_rmpad = (
                        self.compute_entropy_from_logits(logits_rmpad)
                        if not self.config.entropy_checkpointing
                        else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                    )

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs[:, -response_length - 1 : -1]
            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    raise NotImplementedError("return_all_logits=True is not supported with fused kernels in opcdimage.")

                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                if calculate_entropy:
                    if not self.config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if self.config.kl_topk > 0:
                if "kl_topk_indices" in micro_batch:
                    indices = micro_batch["kl_topk_indices"].to(log_probs.device)
                    valid_mask = indices != -1
                    safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
                    gathered_log_probs = torch.gather(log_probs, -1, safe_indices.long())
                    log_probs = torch.where(valid_mask, gathered_log_probs, torch.full_like(gathered_log_probs, -1e20))
                elif "first_kl_topk_indices" in micro_batch:
                    first_indices = micro_batch["first_kl_topk_indices"].to(log_probs.device)
                    target_k = 2 * self.config.kl_topk
                    _, current_indices = torch.topk(log_probs, k=self.config.kl_topk, dim=-1)

                    combined = torch.cat([first_indices, current_indices], dim=-1)
                    combined_sorted, _ = combined.sort(dim=-1)
                    shift = torch.full_like(combined_sorted[..., :1], -1)
                    mask = torch.cat(
                        [
                            torch.ones_like(shift, dtype=torch.bool),
                            combined_sorted[..., 1:] != combined_sorted[..., :-1],
                        ],
                        dim=-1,
                    )
                    max_val = combined_sorted.max()
                    filler = max_val + 1
                    unique_candidates = torch.where(mask, combined_sorted, filler)
                    final_sorted, _ = unique_candidates.sort(dim=-1)
                    merged_indices = final_sorted[..., :target_k]
                    merged_indices[merged_indices > max_val] = -1
                    log_probs = merged_indices.float()
                else:
                    _, indices = torch.topk(log_probs, k=self.config.kl_topk, dim=-1)
                    log_probs = indices.float()

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            return outputs

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        return_all_logits = data.meta_info.get("return_all_logits", False)
        has_precomputed_topk = "kl_topk_indices" in data.batch or "first_kl_topk_indices" in data.batch
        if not return_all_logits and not has_precomputed_topk:
            return super().compute_log_prob(data=data, calculate_entropy=calculate_entropy)

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "kl_topk_indices" in data.batch:
            select_keys.append("kl_topk_indices")
        if "first_kl_topk_indices" in data.batch:
            select_keys.append("first_kl_topk_indices")

        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=torch.distributed.group.WORLD,
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_all_logits=return_all_logits,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        return outputs

    def update_policy(self, data: DataProto):
        if not data.meta_info.get("stage_merge", False):
            return super().update_policy(data)

        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        multi_turn = data.meta_info.get("multi_turn", False)
        on_policy_merge = data.meta_info.get("on_policy_merge", True)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if self.config.kl_loss_type == "full" and self.config.kl_topk > 0:
            select_keys.append("kl_topk_indices")
        if self.config.kl_loss_type != "seqkd":
            select_keys.append("exp_log_probs")
        if multi_turn:
            select_keys.append("loss_mask")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        actual_ppo_mini_batch_size = min(self.config.ppo_mini_batch_size, len(data.batch))
        mini_batches = data.split(actual_ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                        dp_group=torch.distributed.group.WORLD,
                    )
                else:
                    self.gradient_accumulation = max(
                        1, actual_ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)

                    if multi_turn:
                        response_mask = model_inputs["loss_mask"][:, -response_length:]
                    else:
                        response_mask = model_inputs["attention_mask"][:, -response_length:]

                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=True,
                        return_all_logits=self.config.kl_loss_type == "full",
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"]
                    exp_log_prob = model_inputs.get("exp_log_probs", torch.zeros_like(log_prob))

                    policy_loss, _ = compute_reverse_kl_loss(
                        log_prob=log_prob,
                        exp_log_prob=exp_log_prob,
                        response_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                        on_policy_merge=on_policy_merge,
                        kl_loss_type=self.config.kl_loss_type,
                        kl_renorm_topk=self.config.kl_renorm_topk,
                    )
                    entropy_agg = agg_loss(
                        loss_mat=entropy,
                        loss_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / actual_ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    append_to_dict(
                        metrics,
                        {
                            "actor/policy_loss": policy_loss.detach().item(),
                            "actor/entropy": entropy_agg.detach().item(),
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
