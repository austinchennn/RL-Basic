"""超参数与路径配置。"""
from dataclasses import dataclass


@dataclass
class Config:
    # 环境（数据）
    map_name: str = "4x4"
    is_slippery: bool = True
    seed: int = 0

    # Q-learning 公式参数
    alpha: float = 0.1          # 学习率
    gamma: float = 0.99         # 折扣率

    # ε-greedy 探索
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995

    # 训练 / 评估
    num_episodes: int = 20000
    max_steps: int = 100
    eval_episodes: int = 1000
    log_every: int = 2000
    smooth_window: int = 500

    output_dir: str = "outputs"
