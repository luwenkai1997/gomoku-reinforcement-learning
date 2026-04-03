"""
config.py — Centralized hyperparameter configuration for Gomoku RL.

All modules import Config from here. Override defaults via command-line
arguments in train.py / play.py, or by editing this file directly.
"""

import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # ─── Board ───────────────────────────────────────────────────────────────
    board_size: int = 15        # Board width = height
    n_in_row:   int = 5         # Consecutive stones needed to win

    # ─── Neural Network ──────────────────────────────────────────────────────
    n_res_blocks: int = 10      # Number of residual blocks (depth)
    n_filters:    int = 256     # Conv filters per layer

    # ─── MCTS ────────────────────────────────────────────────────────────────
    c_puct:         float = 5.0   # PUCT exploration constant
    n_playout:      int   = 400   # MCTS simulations per move during self-play
    n_playout_eval: int   = 800   # MCTS simulations per move during evaluation

    # ─── Exploration ─────────────────────────────────────────────────────────
    temperature:    float = 1.0   # Softmax temperature for action sampling
    temp_threshold: int   = 30    # Switch to near-greedy after this many moves
    noise_eps:      float = 0.25  # Weight of Dirichlet noise at root
    dirichlet_alpha: float = 0.3  # Dirichlet concentration parameter

    # ─── Training ────────────────────────────────────────────────────────────
    learn_rate:       float = 2e-3
    lr_multiplier:    float = 1.0  # Adaptive LR scaler (adjusted by KL divergence)
    train_batch_size: int   = 512
    update_epochs:    int   = 5    # SGD passes per training step
    kl_coeff:         float = 0.02 # KL threshold coefficient for early stopping / LR adapt
    buffer_size:      int   = 20000  # Replay buffer capacity (in augmented samples)

    # ─── Pipeline ────────────────────────────────────────────────────────────
    total_iterations: int = 5000   # Total training iterations to run
    games_per_iter:   int = 1      # Self-play games collected before each training step
    check_freq:       int = 50     # Evaluate & checkpoint every N iterations
    n_eval_games:     int = 20     # Games played against pure MCTS during evaluation
    pure_mcts_playout: int = 1000  # Pure-MCTS opponent simulation count

    # ─── Parallel Self-Play ──────────────────────────────────────────────────
    # n_workers > 1 enables multiprocessing; each worker uses a CPU copy of model.
    # Recommended: set to (cpu_count - 1), e.g. 3 on Kaggle (4 cores).
    n_workers: int = 1  # 1 = single-process (GPU inference); >1 = parallel CPU workers

    # ─── Paths ───────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    restore_model:  str = ""      # Path to .pt file to resume from (empty = fresh start)
    log_dir:        str = "./logs"

    # ─── Device (auto-detected, do not set manually) ─────────────────────────
    # Populated by __post_init__
    device: object = field(default=None, init=False, repr=False)
    n_gpus: int    = field(default=0,    init=False, repr=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.n_gpus = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_gpus = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def n_actions(self) -> int:
        return self.board_size * self.board_size

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  Gomoku RL Configuration",
            "=" * 55,
            f"  Board:       {self.board_size}×{self.board_size}, win={self.n_in_row}",
            f"  Network:     ResNet-{self.n_res_blocks} × {self.n_filters}ch",
            f"  MCTS:        n_playout={self.n_playout} (eval={self.n_playout_eval})",
            f"  Training:    batch={self.train_batch_size}, epochs={self.update_epochs}, lr={self.learn_rate}",
            f"  Buffer:      {self.buffer_size} samples",
            f"  Device:      {self.device} ({self.n_gpus} GPU(s))",
            f"  Workers:     {self.n_workers}",
            f"  Checkpoints: {self.checkpoint_dir}",
            "=" * 55,
        ]
        return "\n".join(lines)
