# 05 评估强化学习：OpenAI Gym

- **定义**：Gym 是一个用于**开发和比较强化学习算法**的工具包，提供统一接口的标准环境。
- **含义**：Gym 把"环境"封装好了，算法只需要实现 agent。不同算法在同一个环境上跑，就可以直接比较效果。

## 常见环境

| 类别 | 环境 | 动作空间 |
|---|---|---|
| 经典控制（Classical control） | CartPole、Pendulum | CartPole 离散；Pendulum 连续 |
| Atari 游戏 | Pong、Space Invaders、Breakout | 离散 |
| MuJoCo（连续控制） | Ant、Humanoid、Half Cheetah | 连续 |

## CartPole 示例（课件原代码，旧版 Gym API）

```python
import gym
env = gym.make('CartPole-v0')     # 从 Gym 获取 CartPole 环境；env 负责提供状态和奖励

state = env.reset()               # 初始状态 s_1

for t in range(100):
    env.render()                  # 弹出窗口渲染画面
    print(state)

    action = env.action_space.sample()                 # 随机选一个动作
    state, reward, done, info = env.step(action)       # 环境返回 s_{t+1}, r_t

    if done:                      # done=1 表示游戏结束（赢或输）
        print('Finished')
        break

env.close()
```

## 代码与概念的对应

| 代码 | 对应概念 |
|---|---|
| `env` | 环境 Environment |
| `env.reset()` | 回合开始，得到初始状态 $s_1$ |
| `env.action_space.sample()` | 一个**均匀随机策略** $\pi$：从动作空间里随机抽一个 $a_t$ |
| `env.step(action)` | 环境执行状态转移 $s_{t+1} \sim p(\cdot \mid s_t, a_t)$，并给出奖励 $r_t$ |
| `done` | 回合（episode）是否结束 |
| 整个 `for` 循环 | 一次 agent 与环境的交互，产生一条轨迹 $s_1, a_1, r_1, s_2, \cdots$ |

> **版本提示**：Gym 现在由 Farama 基金会以 **Gymnasium** 的名字维护，API 有变化：`reset()` 返回 `(obs, info)`；`step()` 返回 5 元组，用 `terminated`（到达终止状态）和 `truncated`（超时截断）取代了 `done`；渲染在 `make` 时指定。新写法：
>
> ```python
> import gymnasium as gym
>
> env = gym.make("CartPole-v1", render_mode="human")
> state, info = env.reset()
>
> for t in range(100):
>     action = env.action_space.sample()
>     state, reward, terminated, truncated, info = env.step(action)
>     if terminated or truncated:
>         print("Finished")
>         break
>
> env.close()
> ```
