# Q-learning 模型讲解（Model）

前置概念见 [basic.md](basic.md)，任务见 [task.md](task.md)。本文把 basic 里的概念串成一个完整的算法，并对应到代码。

---

## 1. 一句话概括

> **Q-learning = 用 ε-greedy 和环境交互收集样本，再用 TD 更新让 Q 表逐步满足贝尔曼最优方程，最终 $Q \to Q^\star$ ，然后用 $\arg\max$ 选动作。**

- **类别**：Value-based（学价值函数，不直接学策略）、model-free（不需要知道 $p$ ）、off-policy（见第 5 节）、表格型（tabular）。

---

## 2. 模型是什么：一张 Q 表

FrozenLake 有 16 个状态、4 个动作，所以"模型"就是一个 $16 \times 4$ 的矩阵，第 $s$ 行第 $a$ 列存的是 $Q(s, a)$ ，即对 $Q^\star(s, a)$ 的估计。初始全为 0。

- 预测时用的策略： $\pi(s) = \arg\max_{a} Q(s, a)$
- 状态价值： $V(s) = \max_{a} Q(s, a)$ （`policy.png` 里的热力图）

代码：[src/model.py](src/model.py) 中的 `QTable`，只负责存取，不含学习逻辑。

---

## 3. 从贝尔曼最优方程推出更新规则

**第 1 步：目标方程。** $Q^\star$ 满足贝尔曼最优方程（basic 3.3）：

$$
Q^\star(s_t, a_t) = \mathbb E_{S_{t+1}} \left[ R_t + \gamma \max_{a'} Q^\star(S_{t+1}, a') \right]
$$

**第 2 步：期望没法算，用一个样本代替。** 和环境交互一步，得到 $(s_t, a_t, r_t, s_{t+1})$ ，用它代替期望，再把未知的 $Q^\star$ 换成当前的估计 $Q$ ，得到 **TD 目标**：

$$
y_t = r_t + \gamma \max_{a} Q(s_{t+1}, a)
$$

**第 3 步：让 $Q(s_t, a_t)$ 向 $y_t$ 靠近一小步。** 只靠一个样本有噪声（冰面打滑， $s_{t+1}$ 是随机的），所以不直接令 $Q = y_t$ ，而是用学习率 $\alpha$ 做加权平均：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \underbrace{r_t + \gamma \max_{a} Q(s_{t+1}, a)}_{\text{TD 目标 } y_t} - Q(s_t, a_t) \right]
$$

方括号里就是 **TD 误差** $\delta_t$ 。多次访问同一个 $(s, a)$ 后，这个加权平均会逼近 $y_t$ 的期望，也就是贝尔曼方程右边，于是 $Q$ 逐渐满足贝尔曼最优方程。

**终止状态**：掉洞或到终点后没有未来回报，所以 $y_t = r_t$ 。注意要区分 `terminated`（真终止）和 `truncated`（步数用完被截断）：截断时 agent 并没有结束，仍然要加上 $\gamma \max Q(s_{t+1}, \cdot)$ 。

代码：[src/formula.py](src/formula.py) 中的 `QLearningFormula.td_target / td_error / update`，三行公式一一对应。

---

## 4. 算法流程

```
初始化 Q(s, a) = 0，ε = 1
for 每个回合 episode:
    s ← 环境 reset
    for 每一步:
        a ← ε-greedy(Q(s, ·))                       # 探索 or 利用
        执行 a，观测 r, s', terminated
        y ← r                       若 terminated
            r + γ · max_a' Q(s', a')  否则           # TD 目标
        δ ← y − Q(s, a)                              # TD 误差
        Q(s, a) ← Q(s, a) + α · δ                    # TD 更新
        s ← s'
        若回合结束则 break
    ε ← max(ε_min, ε · d)                            # ε 衰减
```

代码：[src/trainer.py](src/trainer.py) 的 `Trainer.run_episode` 就是内层循环。

---

## 5. 为什么是 off-policy

Q-learning 里有**两个策略**：

| 策略 | 作用 | 在 Q-learning 中 |
|---|---|---|
| 行为策略（behavior policy） | 实际用来和环境交互、产生样本 | ε-greedy |
| 目标策略（target policy） | 要学习、评估的策略 | 贪心 $\arg\max_a Q$ |

TD 目标里用的是 $\max_{a} Q(s_{t+1}, a)$ ，也就是假设"下一步按**贪心**走"，而不管下一步实际按 ε-greedy 选了什么动作。两个策略不同，所以叫 **off-policy**。

对比 **SARSA**（on-policy）：它的 TD 目标是 $r_t + \gamma Q(s_{t+1}, a_{t+1})$ ，其中 $a_{t+1}$ 是 ε-greedy 实际选的动作，学到的是 ε-greedy 策略本身的价值。

off-policy 的好处：样本可以来自任何行为策略，这也是后面 DQN 能用**经验回放**反复利用旧样本的原因。

---

## 6. ε-greedy 在这个任务里为什么必不可少

FrozenLake 的奖励**极其稀疏**：只有到达 G 才有 1。初始 Q 表全为 0：

- 如果不探索，agent 每一步都选同一个动作，基本不可能走到 G， $r$ 永远是 0，TD 误差永远是 0，Q 表永远不会变。
- 有了随机探索，agent 偶尔会走到 G，奖励 1 先更新到 $Q(14, \cdot)$ 、 $Q(13, \cdot)$ 这些靠近终点的格子，然后通过 TD 目标里的 $\max Q(s', \cdot)$ 一步一步**往回传播**到起点。这就是训练曲线"先平后升"的原因。

代码里还有一个细节：`greedy()` 在多个动作 Q 值相同时**随机打破平局**。否则初始全 0 时 `argmax` 永远返回动作 0（向左），探索效率会很低。

---

## 7. 读懂训练结果

运行 `python main.py`（打滑环境）的一次结果：

```
episode   2000 | success rate 0.059 | epsilon 0.368
episode   6000 | success rate 0.412 | epsilon 0.050
episode  12000 | success rate 0.705 | epsilon 0.010
greedy policy success rate over 1000 episodes: 0.730
```

- **前期成功率低**：ε 大，基本在乱走，同时奖励还没传回起点。
- **中期上升**：ε 变小，Q 表也变准，开始利用学到的东西。
- **后期稳定在 ~0.7**：打滑环境下理论最优约 0.74，已经接近上限。

**`policy.png` 里一些"反直觉"的箭头**：比如起点 0 的最优动作是**向左**（撞墙），格子 4 也是向左。原因是冰面打滑：向左时实际会以 1/3 概率往下、1/3 往上、1/3 撞墙原地不动，**永远不会掉进右边的洞**。agent 学会了"背对危险走，借打滑慢慢挪过去"。这说明 Q-learning 学到的是**期望回报最大**的策略，而不是几何上最短的路径。

确定性环境（`python main.py --no-slippery`）下，贪心路径就是一条最短路：`[0, 1, 2, 6, 10, 14, 15]`。

---

## 8. 代码结构（解耦）

| 层 | 文件 | 类 | 职责 |
|---|---|---|---|
| 配置 | [config.py](config.py) | `Config` | 所有超参数 |
| 数据 | [src/data.py](src/data.py) | `FrozenLakeData` | 封装环境，产生 $(s, a, r, s')$ |
| 模型 | [src/model.py](src/model.py) | `QTable` | 存取 $Q(s, a)$ 、保存 / 加载 |
| 公式 | [src/formula.py](src/formula.py) | `QLearningFormula`、`EpsilonGreedy` | TD 目标 / 误差 / 更新；ε-greedy 与衰减 |
| 训练 | [src/trainer.py](src/trainer.py) | `Trainer` | 组合以上各层，跑训练循环、记录历史 |
| 预测 | [src/predictor.py](src/predictor.py) | `Predictor` | 贪心决策、评估成功率、rollout |
| 可视化 | [src/visualizer.py](src/visualizer.py) | `Visualizer` | 训练曲线、策略箭头 + 价值热力图 |
| 入口 | [main.py](main.py) | — | 组装 → 训练 → 评估 → 画图 |

解耦的好处：想换成 SARSA，只改 `formula.py` 的 TD 目标；想换成 CliffWalking，只改 `data.py`；想换成 DQN，把 `QTable` 换成神经网络，其他层基本不动。

---

## 9. 超参数

| 参数 | 值 | 作用 |
|---|---|---|
| $\alpha$ | 0.1 | 学习率。太大 → 被打滑带来的噪声带偏；太小 → 收敛慢 |
| $\gamma$ | 0.99 | 折扣率。接近 1，鼓励走到终点（虽然慢，但别掉洞） |
| $\epsilon_0, \epsilon_{\min}, d$ | 1.0, 0.01, 0.9995 | 约 9000 回合后降到最低 |
| 回合数 | 20000 | |

## 10. 局限 → 引出 DQN

Q 表要给**每个** $(s, a)$ 存一个数。FrozenLake 只有 64 个，但 CartPole 的状态是连续的，Atari 的状态是图像，根本存不下。解决办法是用神经网络 $Q(s, a; w)$ 近似 Q 表，更新规则不变（还是 TD 目标），这就是下一个模型 [DQN](../DQN/)。
