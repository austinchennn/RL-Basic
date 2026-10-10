"""入口：组装各层 → 训练 → 评估 → 可视化。"""
import argparse
import os

from config import Config
from src.data import FrozenLakeData
from src.formula import EpsilonGreedy, QLearningFormula
from src.model import QTable
from src.predictor import Predictor
from src.trainer import Trainer
from src.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-slippery", action="store_true", help="使用确定性（不打滑）环境")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.no_slippery:
        cfg.is_slippery = False
    if args.episodes:
        cfg.num_episodes = args.episodes
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 数据 / 模型 / 公式
    data = FrozenLakeData(cfg.map_name, cfg.is_slippery, cfg.seed)
    q_table = QTable(data.num_states, data.num_actions)
    formula = QLearningFormula(cfg.alpha, cfg.gamma)
    explorer = EpsilonGreedy(cfg.epsilon_start, cfg.epsilon_min, cfg.epsilon_decay,
                             data.num_actions, cfg.seed)

    # 训练
    trainer = Trainer(data, q_table, formula, explorer, cfg.max_steps)
    history = trainer.train(cfg.num_episodes, cfg.log_every)
    q_table.save(f"{cfg.output_dir}/q_table.npy")

    # 预测 / 评估（新开一个环境，避免和训练共用随机状态）
    eval_data = FrozenLakeData(cfg.map_name, cfg.is_slippery, cfg.seed + 1)
    predictor = Predictor(q_table, cfg.seed)
    success = predictor.evaluate(eval_data, cfg.eval_episodes, cfg.max_steps)
    print(f"\ngreedy policy success rate over {cfg.eval_episodes} episodes: {success:.3f}")
    path, actions = predictor.rollout(eval_data, cfg.max_steps)
    print("example path:", path)

    # 可视化
    vis = Visualizer(cfg.output_dir)
    print("saved:", vis.plot_training(history, cfg.smooth_window))
    print("saved:", vis.plot_policy(q_table, data.grid))
    print("saved:", vis.plot_task(data.grid, cfg.is_slippery))
    print("saved:", vis.animate_episode(data.grid, path, actions))

    data.close()
    eval_data.close()


if __name__ == "__main__":
    main()
