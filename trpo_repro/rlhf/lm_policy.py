from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .reward_model import ModelLoadConfig, build_lora_config, load_causal_lm


@dataclass
class LMForwardOutput:
    logits: torch.Tensor
    values: torch.Tensor | None = None


def shifted_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return log p(input_ids[:, 1:] | input_ids[:, :-1])."""
    shifted_logits = logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def response_label_mask_from_lengths(input_ids: torch.Tensor, prompt_width: int, response_lengths: torch.Tensor) -> torch.Tensor:
    """Mask shifted labels corresponding to generated response tokens.

    input_ids has shape [B, L]. Token at absolute sequence position p is the
    label predicted by logits at p-1, so its shifted-label index is p-1.
    response_lengths gives the number of generated tokens to include per row.
    This avoids relying on token_id != pad_id, which is unsafe when pad_token_id
    is the same as eos_token_id, as it commonly is for decoder-only LMs.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2, got {tuple(input_ids.shape)}")
    response_lengths = response_lengths.to(device=input_ids.device, dtype=torch.long)
    label_width = max(input_ids.size(1) - 1, 0)
    label_positions = torch.arange(label_width, device=input_ids.device).unsqueeze(0)
    start = max(int(prompt_width) - 1, 0)
    end = start + response_lengths.unsqueeze(1)
    return (label_positions >= start) & (label_positions < end)


class TokenPolicyWithValue(nn.Module):
    """Causal LM policy with a token-value head for PPO."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        *,
        torch_dtype: str = "auto",
        device_map: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        lora: dict[str, Any] | None = None,
        gradient_checkpointing: bool = True,
        trust_remote_code: bool = False,
    ) -> "TokenPolicyWithValue":
        backbone = load_causal_lm(
            ModelLoadConfig(
                model_name=model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                trust_remote_code=trust_remote_code,
            )
        )
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            if hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
        if load_in_4bit or load_in_8bit:
            from peft import prepare_model_for_kbit_training

            backbone = prepare_model_for_kbit_training(backbone)
        lora_cfg = build_lora_config(lora)
        if lora_cfg is not None:
            from peft import get_peft_model

            backbone = get_peft_model(backbone, lora_cfg)
        hidden_size = int(getattr(backbone.config, "hidden_size", getattr(backbone.config, "n_embd", 0)))
        if hidden_size <= 0:
            raise ValueError("Could not infer model hidden size.")
        return cls(backbone=backbone, hidden_size=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> LMForwardOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        values = self.value_head(hidden).squeeze(-1)
        return LMForwardOutput(logits=outputs.logits, values=values)

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.Tensor:
        return self.backbone.generate(**kwargs)

    def save_rlhf_pretrained(self, output_dir: str | Path, tokenizer: Any | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(output_dir / "adapter_or_model")
        torch.save(self.value_head.state_dict(), output_dir / "value_head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir / "tokenizer")

    @classmethod
    def load_rlhf_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        base_model_name: str,
        torch_dtype: str = "auto",
        device_map: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ) -> "TokenPolicyWithValue":
        from peft import PeftModel

        checkpoint_dir = Path(checkpoint_dir)
        model = cls.from_model_name(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            lora=None,
            gradient_checkpointing=False,
            trust_remote_code=trust_remote_code,
        )
        adapter_dir = checkpoint_dir / "adapter_or_model"
        if adapter_dir.exists():
            model.backbone = PeftModel.from_pretrained(model.backbone, adapter_dir, is_trainable=True)
        value_head_path = checkpoint_dir / "value_head.pt"
        if value_head_path.exists():
            model.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))
        return model

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class FrozenCausalLM(nn.Module):
    """Frozen reference scorer used for KL-to-reference."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        *,
        torch_dtype: str = "auto",
        device_map: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ) -> "FrozenCausalLM":
        return cls(
            load_causal_lm(
                ModelLoadConfig(
                    model_name=model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    trust_remote_code=trust_remote_code,
                )
            )
        )

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits

    @torch.no_grad()
    def token_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return shifted_token_logprobs(self.logits(input_ids, attention_mask), input_ids)
