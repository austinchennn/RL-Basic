"""数据层：封装环境，负责产生转移样本 (s, a, r, s', done)。"""
import gymnasium as gym


class FrozenLakeData:
    def __init__(self, map_name="4x4", is_slippery=True, seed=0, render_mode=None):
        self.env = gym.make(
            "FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode
        )
        self.seed = seed
        self._first_reset = True

    @property
    def num_states(self):
        return self.env.observation_space.n

    @property
    def num_actions(self):
        return self.env.action_space.n

    @property
    def grid(self):
        """地图字符矩阵，例如 [['S','F',...], ...]。"""
        return [[c.decode() for c in row] for row in self.env.unwrapped.desc]

    def reset(self):
        # 只在第一次 reset 时设种子，保证整次训练可复现
        if self._first_reset:
            self.env.action_space.seed(self.seed)
            state, _ = self.env.reset(seed=self.seed)
            self._first_reset = False
        else:
            state, _ = self.env.reset()
        return state

    def step(self, action):
        """返回 (s', r, terminated, truncated)。

        terminated：掉洞或到终点（真正的终止状态，之后没有回报）；
        truncated：步数用完（不是终止状态，只是被截断）。
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return next_state, float(reward), terminated, truncated

    def close(self):
        self.env.close()
