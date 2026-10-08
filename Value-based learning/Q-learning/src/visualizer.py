"""可视化层：训练曲线、策略箭头 + 状态价值热力图。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

# 动作编号 → 箭头 (dx, dy)：0 左、1 下、2 右、3 上（图像坐标 y 向下）
ARROWS = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}
ACTION_NAMES = {0: "left", 1: "down", 2: "right", 3: "up"}
CELL_COLORS = {"S": "#c9b6e4", "F": "#dff1fb", "H": "#2b3a55", "G": "#8fd694"}


class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def _moving_average(x, window):
        x = np.asarray(x, dtype=float)
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    def plot_training(self, history, window, filename="training_curve.png"):
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        curves = [
            ("reward", f"success rate (moving avg {window})", True),
            ("td_error", f"mean |TD error| (moving avg {window})", True),
            ("epsilon", "epsilon", False),
        ]
        for ax, (key, label, smooth) in zip(axes, curves):
            y = self._moving_average(history[key], window) if smooth else history[key]
            ax.plot(np.arange(len(y)) + (window if smooth else 0), y)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("episode")
        fig.tight_layout()
        path = f"{self.output_dir}/{filename}"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def plot_policy(self, q_table, grid, filename="policy.png"):
        n_rows, n_cols = len(grid), len(grid[0])
        values = q_table.state_values().reshape(n_rows, n_cols)
        actions = q_table.greedy_actions().reshape(n_rows, n_cols)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(values, cmap="Blues")
        fig.colorbar(im, ax=ax, label="V(s) = max_a Q(s, a)")

        for r in range(n_rows):
            for c in range(n_cols):
                cell = grid[r][c]
                if cell in "HG":
                    ax.text(c, r, cell, ha="center", va="center", fontsize=18,
                            color="red" if cell == "H" else "green", weight="bold")
                    continue
                dx, dy = ARROWS[int(actions[r, c])]
                ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.15, color="black")
                ax.text(c - 0.4, r - 0.35, f"{values[r, c]:.2f}", fontsize=8)
                if cell == "S":
                    ax.text(c + 0.3, r + 0.4, "S", fontsize=10, color="purple")

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_title("Greedy policy and state values")
        fig.tight_layout()
        path = f"{self.output_dir}/{filename}"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ---------- 任务可视化 ----------

    @staticmethod
    def _draw_grid(ax, grid, show_index=True):
        n_rows, n_cols = len(grid), len(grid[0])
        for r in range(n_rows):
            for c in range(n_cols):
                cell = grid[r][c]
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=CELL_COLORS[cell],
                                       edgecolor="white", linewidth=2))
                color = "white" if cell == "H" else "black"
                ax.text(c, r, cell, ha="center", va="center", fontsize=16, weight="bold", color=color)
                if show_index:
                    ax.text(c - 0.42, r - 0.42, str(r * n_cols + c), fontsize=8,
                            va="top", color=color)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_task(self, grid, is_slippery, filename="task.png"):
        """左：地图（状态编号、格子类型、奖励）；中：动作编号；右：打滑转移概率。"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

        # 1. 地图
        ax = axes[0]
        self._draw_grid(ax, grid)
        n_rows, n_cols = len(grid), len(grid[0])
        for r in range(n_rows):
            for c in range(n_cols):
                if grid[r][c] == "G":
                    ax.text(c, r + 0.3, "r = 1", ha="center", fontsize=9)
                elif grid[r][c] == "H":
                    ax.text(c, r + 0.3, "end, r = 0", ha="center", fontsize=8, color="white")
        ax.set_title("Map: 16 states (index = row * 4 + col)")

        # 2. 动作空间
        ax = axes[1]
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(1.6, -1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor=CELL_COLORS["F"], edgecolor="gray"))
        ax.text(0, 0, "s", ha="center", va="center", fontsize=16)
        for a, (dx, dy) in ARROWS.items():
            ax.arrow(dx * 1.5, dy * 1.5, dx * 1.8, dy * 1.8, head_width=0.12, color="black")
            ax.text(dx * 4.6, dy * 4.6, f"{a}: {ACTION_NAMES[a]}", ha="center", va="center", fontsize=11)
        ax.set_title("Actions: 4 discrete actions")

        # 3. 转移概率（以“向右”为例）
        ax = axes[2]
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(1.6, -1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor=CELL_COLORS["F"], edgecolor="gray"))
        ax.text(0, 0, "a = right", ha="center", va="center", fontsize=9)
        moves = {(1, 0): 1 / 3, (0, -1): 1 / 3, (0, 1): 1 / 3, (-1, 0): 0} if is_slippery \
            else {(1, 0): 1, (0, -1): 0, (0, 1): 0, (-1, 0): 0}
        for (dx, dy), prob in moves.items():
            if prob == 0:
                continue
            ax.arrow(dx * 0.5, dy * 0.5, dx * 0.5, dy * 0.5, head_width=0.1,
                     color="tab:red" if (dx, dy) == (1, 0) else "tab:orange")
            ax.text(dx * 1.3, dy * 1.3, f"{prob:.2f}", ha="center", va="center", fontsize=12)
        title = "Slippery: p(s' | s, right)" if is_slippery else "Deterministic: p(s' | s, right)"
        ax.set_title(title)
        ax.text(0, 1.5, "intended direction 1/3, each perpendicular 1/3, never backwards"
                if is_slippery else "always moves in the intended direction",
                ha="center", fontsize=8)

        fig.suptitle("FrozenLake-v1 task: reach G from S without falling into H", fontsize=13)
        fig.tight_layout()
        path = f"{self.output_dir}/{filename}"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def animate_episode(self, grid, path_states, actions, filename="episode.gif", fps=4):
        """把一局的状态序列做成 GIF：agent 位置、已走过的轨迹、意图动作、步数。"""
        n_cols = len(grid[0])
        coords = [(s // n_cols, s % n_cols) for s in path_states]

        fig, ax = plt.subplots(figsize=(5, 5.4))
        self._draw_grid(ax, grid)
        (trail,) = ax.plot([], [], "-", color="tab:orange", linewidth=2, alpha=0.6)
        (agent,) = ax.plot([], [], "o", color="tab:red", markersize=22)
        title = ax.set_title("")

        def update(i):
            rows = [r for r, _ in coords[: i + 1]]
            cols = [c for _, c in coords[: i + 1]]
            trail.set_data(cols, rows)
            agent.set_data([cols[-1]], [rows[-1]])
            r, c = coords[i]
            if i < len(actions):
                text = f"step {i}  state {path_states[i]}  action: {ACTION_NAMES[actions[i]]}"
            else:
                outcome = {"G": "reached Goal!", "H": "fell into Hole"}.get(grid[r][c], "truncated")
                text = f"step {i}  state {path_states[i]}  {outcome}"
            title.set_text(text)
            return trail, agent, title

        anim = FuncAnimation(fig, update, frames=len(coords), interval=1000 // fps)
        out = f"{self.output_dir}/{filename}"
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return out
