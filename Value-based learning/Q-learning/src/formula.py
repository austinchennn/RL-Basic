"""公式层：basic.md 中的公式，纯函数，不依赖环境和训练流程。"""
import numpy as np


class QLearningFormula:
    """贝尔曼最优方程的样本形式：TD 目标 / TD 误差 / TD 更新。"""

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def td_target(self, reward, next_q_values, terminated):
        """y_t = r_t + γ · max_a Q(s_{t+1}, a)；终止状态时 y_t = r_t。"""
        if terminated:
            return reward
        return reward + self.gamma * np.max(next_q_values)

    def td_error(self, target, q_value):
        """δ_t = y_t - Q(s_t, a_t)。"""
        return target - q_value

    def update(self, q_value, td_error):
        """Q(s_t, a_t) ← Q(s_t, a_t) + α · δ_t。"""
        return q_value + self.alpha * td_error


class EpsilonGreedy:
    """ε-greedy 探索策略与 ε 指数衰减。"""

    def __init__(self, epsilon_start, epsilon_min, epsilon_decay, num_actions, seed=0):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def select(self, q_values):
        """以概率 ε 随机选动作，否则选 argmax（并列最大时随机打破平局）。"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return greedy(q_values, self.rng)

    def decay(self):
        """ε_k = max(ε_min, ε_0 · d^k)，每回合结束调用一次。"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def greedy(q_values, rng):
    """argmax_a Q(s, a)。初始 Q 全为 0 时，直接 argmax 会一直选动作 0，所以随机打破平局。"""
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))
