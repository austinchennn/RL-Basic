# RL-Basic

强化学习基础：学习笔记 + 模型复现。

> **来源说明**：本项目中的概念与定义来自 Shusen Wang（王树森）的《深度强化学习》课程：<https://github.com/wangshusen/DRL>

## 目录

```
RL-Basic/
├── Note/                      # 学习笔记
│   ├── 基础概念/               # Lecture 1：术语、随机性、回报、价值函数、几种学习方式
│   └── RL中的数值计算/         # 采样估计 / 蒙特卡罗
├── Value-based learning/
│   └── DQN/                   # Deep Q-Network
├── Policy-based learning/
│   └── Policy Network/        # 策略网络
└── Actor-Critic Methods/      # Actor-Critic 方法
```

- [Note/基础概念/](Note/基础概念/)：Lecture 1（`1_Basics_1.pdf`）中的术语、随机性、回报、价值函数，以及几种学习方式
- [Note/RL中的数值计算/](Note/RL中的数值计算/)：RL 里常用的数值方法（采样估计 / 蒙特卡罗）
- [Value-based learning/DQN/](Value-based%20learning/DQN/)：学最优动作价值函数 $Q^\star$，用 $\operatorname{argmax}$ 选动作
- [Policy-based learning/Policy Network/](Policy-based%20learning/Policy%20Network/)：学策略 $\pi(a \mid s)$，用采样选动作
- [Actor-Critic Methods/](Actor-Critic%20Methods/)：同时学策略网络（actor）和价值网络（critic）
