"""预测层：用训练好的 Q 表做贪心决策，并评估成功率。"""
import numpy as np

from .formula import greedy


class Predictor:
    def __init__(self, q_table, seed=0):
        self.q = q_table
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        """π(s) = argmax_a Q(s, a)，不再探索。"""
        return greedy(self.q.values(state), self.rng)

    def evaluate(self, data, num_episodes, max_steps):
        """返回贪心策略的成功率（到达终点的回合比例）。"""
        successes = 0
        for _ in range(num_episodes):
            state = data.reset()
            for _ in range(max_steps):
                state, reward, terminated, truncated = data.step(self.act(state))
                if terminated or truncated:
                    successes += reward > 0
                    break
        return successes / num_episodes

    def rollout(self, data, max_steps):
        """跑一局，返回 (状态序列, 动作序列)，用于展示或动画。"""
        state = data.reset()
        path, actions = [state], []
        for _ in range(max_steps):
            action = self.act(state)
            state, _, terminated, truncated = data.step(action)
            path.append(state)
            actions.append(action)
            if terminated or truncated:
                break
        return path, actions
