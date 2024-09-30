import os
from typing import Literal

import torch
from tap import Tap

from src.kd_trainer import KDTrainer
# from src.sft_trainer import SFTTrainer
# from src.trainer import GFNTrainer
# from src.ar_trainer import ARTrainer
from src.utils import seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Argument(Tap):
    baseline: bool = False
    mode: Literal["gflownet", "sft", "gflownet-mistral", "kd", "ar"] = "gflownet"
    model_name: str = "gpt2"
    sft_ckpt: str = "save/gpt2-sft/latest"
    save_dir: str = "./save"

    prompt_file: str = "data/sft_dataset.json"
    seed_file: str = None

    lr: float = 1e-4
    max_norm: float = 1.0
    weight_decay: float = 0.1

    num_warmup_steps: int = 10
    train_steps: int = 5000
    batch_size: int = 256
    epochs: int = 3
    grad_acc_steps: int = 8

    len_norm: bool = False
    max_len: int = 20
    min_len: int = 5

    load_buffer: bool = False
    buffer_size: int = 1000
    sim_tolerance: float = 0.25
    prioritization: Literal["c_reward", "reward", "uniform"] = "c_reward"
    buffer_ckpt: str = ""
    compare: str = "c_reward"
    metric: Literal["edit", "cosine"] = "cosine"

    seed: int = 42

    eval_period: int = 500
    eval_batch_size: int = 1024
    # lora hparams
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # reward scaling
    beta: float = 0.1
    lm_sched_end: float = 1.0
    lm_sched_start: float = 1.0
    lm_sched_horizon: int = 2000

    # reward temperature
    reward_sched_start: float = 2.0
    reward_sched_end: float = 1.0
    reward_sched_horizon: int = 500

    # sampling temperature
    temp_low: float = 0.5
    temp_high: float = 2.0

    # kd
    kd_model: str = None
    kd_file: str = None
    max_length: int = 1024
    student_ckpt: str = None
    # wandb
    exp_name: str = "debug"
    wandb_project: str = "llama-guard-distillation-neo"
    wandb_entity: str = "seanie14"
    wandb_group: str = None
    num_threads: int = 16


if __name__ == "__main__":
    args = Argument(explicit_bool=True).parse_args()
    torch.set_num_threads(args.num_threads)
    seed(args.seed)
    if args.mode == "sft":
        print("sft")
        # trainer = SFTTrainer(args)
    elif args.mode == "kd":
        print("kd")
        trainer = KDTrainer(args)
    elif args.mode == "gflownet":
        print("GFlowNet")
        # trainer = GFNTrainer(args)
    elif args.mode == "ar":
        print("ar")
        # trainer = ARTrainer(args)
    trainer.train()
