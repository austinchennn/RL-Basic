"""模型层：Q 表，只负责存取 Q(s, a)，不含任何学习逻辑。"""
import numpy as np


class QTable:
    def __init__(self, num_states, num_actions):
        self.table = np.zeros((num_states, num_actions))

    def get(self, state, action):
        return self.table[state, action]

    def set(self, state, action, value):
        self.table[state, action] = value

    def values(self, state):
        """返回 Q(s, ·)，形状 (num_actions,)。"""
        return self.table[state]

    def state_values(self):
        """V(s) = max_a Q(s, a)，形状 (num_states,)。"""
        return self.table.max(axis=1)

    def greedy_actions(self):
        """π(s) = argmax_a Q(s, a)，形状 (num_states,)。"""
        return self.table.argmax(axis=1)

    def save(self, path):
        np.save(path, self.table)

    @classmethod
    def load(cls, path):
        table = np.load(path)
        q = cls(*table.shape)
        q.table = table
        return q
