import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import torch.distributed as dist


class RangeTracker():
    def __init__(self, layer_idx, gla_cfg, max_batches=100000, log_interval=1000):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.rank != 0:
            return

        try:
            # print(f"Get gla config:\n{type(gla_cfg)}")
            task_cfg = gla_cfg.task_cfg
            # print(f"Get task config:\n{type(task_cfg)}")

            expt_name = task_cfg.get('expt_name')
            expt_time = task_cfg.get('time')
            # print(f"Experiment name: {expt_name}, time: {expt_time}")

            if task_cfg.get('debug').get('flag'):
                range_experiment_dir = os.path.join(
                    'range_experiment', f'{expt_name}', 'debug', f'{expt_time}'
                )
            else:
                range_experiment_dir = os.path.join(
                    'range_experiment', f'{expt_name}', f'{expt_time}'
                )

        except Exception as e:
            print(f"Get Hydra config failed:\n{e}")

        self.layer_idx = layer_idx
        self.max_batches = max_batches
        self.log_interval = log_interval

        self.global_q_min = float("inf")
        self.global_q_max = float("-inf")
        self.global_k_min = float("inf")
        self.global_k_max = float("-inf")
        self.count = 0

        # 存储历史数据
        self.q_min_history = deque(maxlen=max_batches)
        self.q_max_history = deque(maxlen=max_batches)
        self.k_min_history = deque(maxlen=max_batches)
        self.k_max_history = deque(maxlen=max_batches)
        self.steps = deque(maxlen=max_batches)

        self.log_text_dir = os.path.join(range_experiment_dir, 'text')
        os.makedirs(self.log_text_dir, exist_ok=True)

        self.log_pic_dir = os.path.join(range_experiment_dir, 'pic', f'layer_{self.layer_idx}')
        os.makedirs(self.log_pic_dir, exist_ok=True)

        self._init_log_file()

    def _init_log_file(self):
        self.text_file_name = os.path.join(self.log_text_dir, f"qk_range_log_layer_{self.layer_idx}.txt")

        """初始化日志文件，写入头部信息"""
        with open(self.text_file_name, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("QK Range Tracking Log\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max batches: {self.max_batches}\n")
            f.write(f"Layer index: {self.layer_idx}\n")
            f.write("=" * 80 + "\n\n")
    
    def write_stats(self):
        """将统计信息写入文件"""
        try:
            with open(self.text_file_name, 'a', encoding='utf-8') as f:
                f.write(f"Batch {self.count} - {datetime.now().strftime('%H:%M:%S')} - layer {self.layer_idx}\n")
                f.write(f"  Global Q range: [{self.global_q_min:.4f}, {self.global_q_max:.4f}]\n")
                f.write(f"  Global K range: [{self.global_k_min:.4f}, {self.global_k_max:.4f}]\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"写入日志文件失败: {e}")
    
    def update(self, q, k):
        if self.rank != 0:
            return
        
        current_q_min = q.min().item()
        current_q_max = q.max().item()
        current_k_min = k.min().item()
        current_k_max = k.max().item()
        
        self.global_q_min = min(self.global_q_min, current_q_min)
        self.global_q_max = max(self.global_q_max, current_q_max)
        self.global_k_min = min(self.global_k_min, current_k_min)
        self.global_k_max = max(self.global_k_max, current_k_max)
        
        # 记录历史数据
        self.q_min_history.append(current_q_min)
        self.q_max_history.append(current_q_max)
        self.k_min_history.append(current_k_min)
        self.k_max_history.append(current_k_max)
        self.steps.append(self.count)
        
        self.count += 1

        # 动态调整的绘图频率
        if self.count % self.log_interval == 0:
            self.draw_range()
            self.write_stats()

    def draw_range(self):
        """绘制qk值的历史范围"""
        pic_file_name = os.path.join(self.log_pic_dir, f"qk_range_history_batch_{self.count}.png")

        if len(self.steps) < 2:
            return
            
        try:
            plt.figure(figsize=(12, 8))
            
            # 转换为numpy数组便于绘图
            steps = np.array(list(self.steps))
            q_min_hist = np.array(list(self.q_min_history))
            q_max_hist = np.array(list(self.q_max_history))
            k_min_hist = np.array(list(self.k_min_history))
            k_max_hist = np.array(list(self.k_max_history))
            
            # 子图1: Q值范围
            plt.subplot(2, 2, 1)
            plt.plot(steps, q_min_hist, 'b-', label='Q min', alpha=0.7)
            plt.plot(steps, q_max_hist, 'r-', label='Q max', alpha=0.7)
            plt.axhline(y=self.global_q_min, color='b', linestyle='--', alpha=0.5, label=f'Global Q min: {self.global_q_min:.3f}')
            plt.axhline(y=self.global_q_max, color='r', linestyle='--', alpha=0.5, label=f'Global Q max: {self.global_q_max:.3f}')
            plt.fill_between(steps, q_min_hist, q_max_hist, alpha=0.2, color='gray')
            plt.title('Q Values Range Over Time')
            plt.xlabel('Steps')
            plt.ylabel('Q Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: K值范围
            plt.subplot(2, 2, 2)
            plt.plot(steps, k_min_hist, 'g-', label='K min', alpha=0.7)
            plt.plot(steps, k_max_hist, 'm-', label='K max', alpha=0.7)
            plt.axhline(y=self.global_k_min, color='g', linestyle='--', alpha=0.5, label=f'Global K min: {self.global_k_min:.3f}')
            plt.axhline(y=self.global_k_max, color='m', linestyle='--', alpha=0.5, label=f'Global K max: {self.global_k_max:.3f}')
            plt.fill_between(steps, k_min_hist, k_max_hist, alpha=0.2, color='gray')
            plt.title('K Values Range Over Time')
            plt.xlabel('Steps')
            plt.ylabel('K Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图3: 组合范围对比
            plt.subplot(2, 2, 3)
            plt.plot(steps, q_min_hist, 'b-', label='Q min', alpha=0.7)
            plt.plot(steps, q_max_hist, 'b--', label='Q max', alpha=0.7)
            plt.plot(steps, k_min_hist, 'r-', label='K min', alpha=0.7)
            plt.plot(steps, k_max_hist, 'r--', label='K max', alpha=0.7)
            plt.title('Q vs K Range Comparison')
            plt.xlabel('Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图4: 范围宽度变化
            plt.subplot(2, 2, 4)
            q_range_width = q_max_hist - q_min_hist
            k_range_width = k_max_hist - k_min_hist
            plt.plot(steps, q_range_width, 'b-', label='Q range width', alpha=0.7)
            plt.plot(steps, k_range_width, 'r-', label='K range width', alpha=0.7)
            plt.title('Range Width Over Time')
            plt.xlabel('Steps')
            plt.ylabel('Range Width')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(str(pic_file_name), dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"绘图失败: {e}")


class KANActivation(torch.nn.Module):
    def __init__(
        self,
        grid_size=8,
        spline_order=3,
        grid_range=[-2, 2],
    ):
        super(KANActivation, self).__init__()
        self.grid_size = grid_size  # g
        self.spline_order = spline_order  # k

        # 创建B样条网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]  # (g+2k+1,)
        self.register_buffer("grid", grid)

        # 可学习的样条权重
        self.spline_weight = torch.nn.Parameter(
            self._init_linear_spline_weights(grid_range)
        )

    # def _apply(self, fn):
    #     """确保Lightning的自动类型转换正确应用"""
    #     super()._apply(fn)
    #     with torch.no_grad():
    #         self.spline_weight = fn(self.spline_weight)
    #         self.grid = fn(self.grid)
    #     return self

    def _init_linear_spline_weights(self, grid_range):
        """简化的线性映射初始化"""
        num_coeffs = self.grid_size + self.spline_order
        
        # 近似方法：让权重在边界处体现线性关系
        weights = torch.linspace(grid_range[0], grid_range[1], num_coeffs)
        
        # 可能需要调整边界权重以更好地近似线性函数
        return weights  # (g+k,)

    def b_splines(self, x: torch.Tensor):  # BLD
        """计算B样条基函数"""
        assert x.dim() == 3, "Input tensor must have exactly 3 dimensions (BLD)."

        x = x.reshape(-1, x.shape[-1]).unsqueeze(-1)  #  ND1
        # print(f"x: \n{x}")

        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, g+2k+1)
        # print(f"grid: \n{grid}")

        # 初始化指示函数 (k=0)
        bases = ((x >= grid[:,:,:-1]) & (x < grid[:,:,1:])).to(x.dtype)  # ND(g+2k+1)
        # print(f"bases: \n{bases}")

        # 递归计算B样条 (Cox-de Boor 公式)
        for k in range(1, self.spline_order + 1):
            # print(f"------------------ k: {k} ------------------")
            left_denom = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            right_denom = grid[:, :, (k + 1):] - grid[:, :, 1:(-k)]
            # print(f"left_denom(shape: {left_denom.shape}): \n{left_denom}")
            # print(f"right_denom(shape: {right_denom.shape}): \n{right_denom}")

            # 避免除零
            # left_denom = torch.where(left_denom == 0, torch.ones_like(left_denom), left_denom)
            # right_denom = torch.where(right_denom == 0, torch.ones_like(right_denom), right_denom)
            
            bases = (
                (x - grid[:, :, :-(k + 1)]) / left_denom * bases[:, :, :-1] +
                (grid[:, :, (k + 1):] - x) / right_denom * bases[:, :, 1:]
            )

            # print(f"--------------------------------------------\n")
        
        assert bases.shape == (x.shape[0], x.shape[1], self.grid_size + self.spline_order), "B-spline bases shape mismatch."
        return bases  # ND(g+k)

    def forward(self, x: torch.Tensor):
        """前向传播，保持输入输出形状一致"""
        original_shape = x.shape

        bases = self.b_splines(x)  # ND(g+k)

        output = torch.sum(bases * self.spline_weight, dim=-1)  # ND

        return output.view(original_shape)


if __name__ == "__main__":
    pass