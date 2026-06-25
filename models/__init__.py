"""Package dei modelli e degli adapter (LoRA) per il progetto TabPFN."""

from .lora import (
    LoRAConfig,
    LoRALinear,
    count_trainable_parameters,
    inject_lora_adapters,
    load_lora_adapters,
    lora_state_dict,
    mark_only_lora_as_trainable,
    merge_lora_adapters,
    save_lora_adapters,
)
from .tabpfn_lora import (
    create_lora_classifier,
    finetune_lora_on_dataset,
    train_lora,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "count_trainable_parameters",
    "create_lora_classifier",
    "finetune_lora_on_dataset",
    "inject_lora_adapters",
    "load_lora_adapters",
    "lora_state_dict",
    "mark_only_lora_as_trainable",
    "merge_lora_adapters",
    "save_lora_adapters",
    "train_lora",
]
