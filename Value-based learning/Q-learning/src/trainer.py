"""训练层：把数据、模型、公式组合成 Q-learning 训练循环。"""
import numpy as np


class Trainer:
    def __init__(self, data, q_table, formula, explorer, max_steps):
        self.data = data
        self.q = q_table
        self.formula = formula
        self.explorer = explorer
        self.max_steps = max_steps
        self.history = {"reward": [], "epsilon": [], "td_error": []}

    def run_episode(self):
        state = self.data.reset()
        total_reward, abs_errors = 0.0, []

        for _ in range(self.max_steps):
            # 1. ε-greedy 选动作
            action = self.explorer.select(self.q.values(state))
            # 2. 与环境交互，得到样本 (s, a, r, s')
            next_state, reward, terminated, truncated = self.data.step(action)
            # 3. TD 目标 → TD 误差 → TD 更新
            target = self.formula.td_target(reward, self.q.values(next_state), terminated)
            error = self.formula.td_error(target, self.q.get(state, action))
            self.q.set(state, action, self.formula.update(self.q.get(state, action), error))

            total_reward += reward
            abs_errors.append(abs(error))
            state = next_state
            if terminated or truncated:
                break

        return total_reward, float(np.mean(abs_errors))

    def train(self, num_episodes, log_every=0):
        for ep in range(1, num_episodes + 1):
            reward, td_error = self.run_episode()
            self.history["reward"].append(reward)
            self.history["epsilon"].append(self.explorer.epsilon)
            self.history["td_error"].append(td_error)
            self.explorer.decay()

            if log_every and ep % log_every == 0:
                recent = np.mean(self.history["reward"][-log_every:])
                print(f"episode {ep:6d} | success rate (last {log_every}) {recent:.3f} "
                      f"| epsilon {self.explorer.epsilon:.3f}")
        return self.history
