"""Configuration management for experiments."""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None  # None = MHA, < n_heads = GQA
    d_ff: Optional[int] = None  # None = 4 * d_model (or 8/3 * d_model for SwiGLU)
    max_seq_len: int = 1024
    dropout: float = 0.0
    pos_encoding: str = "rope"  # "rope", "alibi", "learned"
    activation: str = "swiglu"  # "swiglu", "gelu"
    norm_type: str = "rmsnorm"  # "rmsnorm", "layernorm"
    norm_position: str = "pre"  # "pre", "post"
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads  # default: MHA
        if self.d_ff is None:
            # SwiGLU uses 8/3 * d_model (rounded to multiple of 256)
            if self.activation == "swiglu":
                raw = int(8 / 3 * self.d_model)
                self.d_ff = ((raw + 255) // 256) * 256
            else:
                self.d_ff = 4 * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.d_model
        attn_q = self.d_model * self.d_model
        attn_kv = 2 * self.d_model * (self.n_kv_heads * self.head_dim)
        attn_o = self.d_model * self.d_model
        attn = attn_q + attn_kv + attn_o
        if self.activation == "swiglu":
            ffn = 3 * self.d_model * self.d_ff  # gate + up + down
        else:
            ffn = 2 * self.d_model * self.d_ff
        norms = 2 * self.d_model  # per layer
        layer_total = attn + ffn + norms
        total = emb + self.n_layers * layer_total
        if not self.tie_embeddings:
            total += emb  # separate LM head
        return total


@dataclass
class DataConfig:
    data_dir: str = "./data/processed"
    seq_len: int = 1024
    batch_size: int = 32
    num_workers: int = 4
    tokenizer_path: str = "./data/tokenizer"
    val_split: float = 0.01


@dataclass
class TrainingConfig:
    max_steps: int = 50000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    grad_accum_steps: int = 4
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    compile: bool = True  # torch.compile


@dataclass
class ExperimentConfig:
    name: str = "base"
    output_dir: str = "./experiments"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def exp_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def save(self, path: Optional[str] = None):
        path = path or str(self.exp_dir / "config.yaml")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        model_cfg = ModelConfig(**raw.pop("model", {}))
        data_cfg = DataConfig(**raw.pop("data", {}))
        train_cfg = TrainingConfig(**raw.pop("training", {}))
        return cls(model=model_cfg, data=data_cfg, training=train_cfg, **raw)

    @classmethod
    def from_overrides(cls, base_path: str, **overrides) -> "ExperimentConfig":
        """Load base config and apply nested overrides like model.pos_encoding='alibi'."""
        cfg = cls.from_yaml(base_path)
        for key, val in overrides.items():
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)
        # Re-run post_init for model if model params changed
        if any(k.startswith("model.") for k in overrides):
            cfg.model.__post_init__()
        return cfg
