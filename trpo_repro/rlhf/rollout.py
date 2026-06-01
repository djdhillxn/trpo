from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .lm_policy import FrozenCausalLM, TokenPolicyWithValue, response_label_mask_from_lengths, shifted_token_logprobs
from .ppo_buffer import LMRolloutBatch


@dataclass
class GenerationConfig:
    max_prompt_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


def _ensure_pad_token(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos if x is not None}
    return {int(eos)}


def _response_lengths(response_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """Number of generated tokens to keep per row, including first EOS if present."""
    eos_ids = _eos_token_ids(tokenizer)
    lengths: list[int] = []
    for row in response_ids.detach().cpu().tolist():
        keep = len(row)
        if eos_ids:
            for idx, token_id in enumerate(row):
                if int(token_id) in eos_ids:
                    keep = idx + 1
                    break
        lengths.append(max(0, keep))
    return torch.tensor(lengths, device=response_ids.device, dtype=torch.long)


def _build_full_attention(prompt_attention: torch.Tensor, generated: torch.Tensor, prompt_width: int, response_lengths: torch.Tensor) -> torch.Tensor:
    full_attention = torch.zeros_like(generated, dtype=torch.long)
    full_attention[:, :prompt_width] = prompt_attention.long()
    if generated.size(1) > prompt_width:
        pos = torch.arange(generated.size(1) - prompt_width, device=generated.device).unsqueeze(0)
        full_attention[:, prompt_width:] = (pos < response_lengths.unsqueeze(1)).long()
    return full_attention


@torch.no_grad()
def collect_lm_rollouts(
    policy: TokenPolicyWithValue,
    reference: FrozenCausalLM,
    reward_model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    generation: GenerationConfig,
    kl_coef: float,
    device: torch.device | str,
    metadata: list[dict[str, Any]] | None = None,
) -> LMRolloutBatch:
    """Generate responses and build an on-policy token-level PPO batch."""
    device = torch.device(device)
    pad_id = _ensure_pad_token(tokenizer)
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(generation.max_prompt_length),
    )
    tokenizer.padding_side = old_padding_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_width = int(input_ids.size(1))

    generated = policy.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(generation.max_new_tokens),
        do_sample=bool(generation.do_sample),
        temperature=float(generation.temperature),
        top_p=float(generation.top_p),
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response_ids = generated[:, prompt_width:]
    response_lengths = _response_lengths(response_ids, tokenizer)
    full_attention = _build_full_attention(attention_mask, generated, prompt_width, response_lengths)

    policy_out = policy(generated, full_attention)
    old_logprobs = shifted_token_logprobs(policy_out.logits, generated).detach()
    values = policy_out.values[:, :-1].detach()
    ref_logprobs = reference.token_logprobs(generated, full_attention).detach()
    resp_mask = response_label_mask_from_lengths(generated, prompt_width, response_lengths)

    expected_shape = resp_mask.shape
    for name, tensor in {"old_logprobs": old_logprobs, "values": values, "ref_logprobs": ref_logprobs}.items():
        if tensor.shape != expected_shape:
            raise RuntimeError(f"{name} shape {tuple(tensor.shape)} does not match response_mask {tuple(expected_shape)}")

    scores = reward_model(generated, full_attention).detach().float()
    kl_per_token = old_logprobs.float() - ref_logprobs.float()
    rewards = (-float(kl_coef) * kl_per_token) * resp_mask.float()

    # Add terminal reward-model score to the final generated token of each sequence.
    for i in range(generated.size(0)):
        positions = torch.nonzero(resp_mask[i], as_tuple=False).flatten()
        if positions.numel() > 0:
            rewards[i, positions[-1]] += scores[i]

    decoded_responses: list[str] = []
    for ids, keep in zip(response_ids, response_lengths.tolist()):
        decoded_responses.append(tokenizer.decode(ids[: int(keep)], skip_special_tokens=True).strip())

    return LMRolloutBatch(
        input_ids=generated.detach(),
        attention_mask=full_attention.detach(),
        response_mask=resp_mask.detach(),
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        values=values.detach(),
        rewards=rewards.detach(),
        scores=scores.detach(),
        prompts=list(prompts),
        responses=decoded_responses,
        metadata=metadata,
    )
