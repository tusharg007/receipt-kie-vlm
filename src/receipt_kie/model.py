"""SmolVLM processor/model loading and validated PEFT LoRA attachment."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from receipt_kie.utils import project_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeDecision:
    device: str
    dtype_name: str
    attention: str
    bf16_supported: bool
    flash_attention_2_installed: bool

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.dtype_name]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_runtime(model_config: dict[str, Any]) -> RuntimeDecision:
    """Detect CUDA, precision, and attention implementation instead of assuming them."""
    cuda = torch.cuda.is_available()
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    requested_precision = str(model_config.get("precision", "auto")).lower()
    if requested_precision == "auto":
        dtype_name = "bf16" if bf16 else "fp16" if cuda else "fp32"
    else:
        dtype_name = requested_precision
    flash_installed = importlib.util.find_spec("flash_attn") is not None
    requested_attention = str(model_config.get("attention", "auto")).lower()
    if requested_attention == "auto":
        attention = (
            "flash_attention_2"
            if flash_installed and cuda and torch.cuda.get_device_capability(0)[0] >= 8
            else "sdpa"
            if hasattr(torch.nn.functional, "scaled_dot_product_attention")
            else "eager"
        )
    else:
        attention = requested_attention
    return RuntimeDecision(
        device="cuda" if cuda else "cpu",
        dtype_name=dtype_name,
        attention=attention,
        bf16_supported=bf16,
        flash_attention_2_installed=flash_installed,
    )


def load_processor(model_config: dict[str, Any], hf_cache: str) -> Any:
    """Load the Idefics3 processor and apply its supported longest-edge option."""
    processor = AutoProcessor.from_pretrained(
        model_config["model_id"],
        cache_dir=str(project_path(hf_cache)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    longest_edge = int(model_config.get("image_longest_edge", 512))
    current_size = dict(processor.image_processor.size)
    current_size["longest_edge"] = longest_edge
    processor.image_processor.size = current_size
    processor.tokenizer.padding_side = "right"
    LOGGER.info(
        "Loaded processor=%s image_size=%s",
        type(processor).__name__,
        processor.image_processor.size,
    )
    return processor


def load_base_model(
    model_config: dict[str, Any],
    hf_cache: str,
    training: bool = False,
) -> tuple[Any, RuntimeDecision]:
    """Load SmolVLM with detected SDPA/eager attention and mixed precision."""
    runtime = choose_runtime(model_config)
    kwargs = {
        "cache_dir": str(project_path(hf_cache)),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "torch_dtype": runtime.torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": runtime.attention,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_config["model_id"], **kwargs)
    except (ImportError, RuntimeError, ValueError) as exc:
        if runtime.attention == "eager":
            raise
        LOGGER.warning(
            "Attention implementation %s failed (%s); retrying with eager",
            runtime.attention,
            exc,
        )
        fallback = RuntimeDecision(
            device=runtime.device,
            dtype_name=runtime.dtype_name,
            attention="eager",
            bf16_supported=runtime.bf16_supported,
            flash_attention_2_installed=runtime.flash_attention_2_installed,
        )
        kwargs["attn_implementation"] = "eager"
        model = AutoModelForImageTextToText.from_pretrained(model_config["model_id"], **kwargs)
        runtime = fallback
    if runtime.device == "cuda":
        model.to("cuda")
    if training:
        model.config.use_cache = False
    else:
        model.eval()
    LOGGER.info("Loaded model runtime=%s", runtime.to_dict())
    return model, runtime


def attach_lora(model: Any, lora_config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Validate requested module suffixes against the model before attaching LoRA."""
    requested = [str(name) for name in lora_config["target_modules"]]
    candidate_names = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
    )
    selected = [name for name in requested if name in candidate_names]
    missing = [name for name in requested if name not in candidate_names]
    if missing:
        LOGGER.warning("Requested LoRA targets not present: %s", missing)
    if not selected:
        raise ValueError(
            f"No valid LoRA targets selected. Requested={requested}; candidates={candidate_names}"
        )
    peft_config = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        bias=str(lora_config.get("bias", "none")),
        target_modules=selected,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)
    stats = parameter_statistics(peft_model)
    stats.update(
        {
            "target_modules": selected,
            "candidate_linear_module_suffixes": candidate_names,
            "estimated_adapter_size_mib_fp32": round(stats["trainable_parameters"] * 4 / 2**20, 2),
        }
    )
    LOGGER.info("Attached LoRA: %s", stats)
    return peft_model, stats


def parameter_statistics(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total,
    }
