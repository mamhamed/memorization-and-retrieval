from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    dataset_sizes: List[int]
    model_sizes: List[str]
    num_variations: List[int]
    vocab_sizes: List[int]
    batch_size: int
    max_length: int
    learning_rate_cpt: float
    learning_rate_ift: float
    num_epochs_cpt: int
    num_epochs_ift: int
    output_dir: str
    use_gpu: bool
    num_gpus: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    max_eval_samples: int
    fsdp_config: Optional[str]=None # path to fsdp config
