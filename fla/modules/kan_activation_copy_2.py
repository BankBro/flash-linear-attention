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


class TensorQuantileStats:
    """单个张量的分位数统计类"""
    def __init__(self, name, max_batches=100000, quantiles=[0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]):
        self.name = name
        
        # 强制包含0.0和1.0分位数，并过滤无效值
        quantiles_set = set(q for q in quantiles if 0.0 <= q <= 1.0)
        quantiles_set.add(0.0)
        quantiles_set.add(1.0)
        self.quantiles = sorted(list(quantiles_set))
        self.num_quantiles = len(self.quantiles)
        
        # 全局最大最小值统计
        self.global_min = float("inf")
        self.global_max = float("-inf")
        
        # 每个batch的分位数历史
        self.quantile_histories = {q: deque(maxlen=max_batches) for q in quantiles}
    
    def update(self, tensor):
        """更新分位数统计信息"""

        # 确保张量是float类型（quantile要求float或double类型）
        if tensor.dtype not in [torch.float32, torch.float64]:
            tensor = tensor.float()

        # 使用torch.quantile计算分位数（GPU加速，如果tensor在GPU上）
        quantile_tensor = torch.tensor(self.quantiles, device=tensor.device, dtype=tensor.dtype)
        current_quantiles_tensor = torch.quantile(tensor.flatten(), quantile_tensor)

        # 只在最后转换为numpy（一次性转换）
        current_quantiles = current_quantiles_tensor.detach().cpu().numpy()
        
        # 从分位数结果中提取最大最小值（排序后0.0在第一个，1.0在最后一个）
        current_min = current_quantiles[0]   # 0.0分位数 = 最小值
        current_max = current_quantiles[-1]  # 1.0分位数 = 最大值
        self.global_min = min(self.global_min, current_min)
        self.global_max = max(self.global_max, current_max)
        
        # 存储到对应的历史记录中
        for i, q in enumerate(self.quantiles):
            self.quantile_histories[q].append(current_quantiles[i])
        
        return current_quantiles
    
    def get_history_arrays(self):
        """获取历史数据的numpy数组 - 基于实际配置的分位数动态生成"""
        result = {}
        
        # 动态生成分位数映射，基于实际配置的 self.quantiles
        for q_val in self.quantiles:
            p_key = f'p{int(q_val*100):02d}'  # 例如: 0.05 -> 'p05', 0.5 -> 'p50'
            
            if q_val in self.quantile_histories:
                result[p_key] = np.array(list(self.quantile_histories[q_val]))
            else:
                # 如果分位数不存在，返回空数组
                result[p_key] = np.array([])
        
        return result
    
    def get_quantile_history(self, quantile):
        """获取指定分位数的历史数据"""
        if quantile in self.quantile_histories:
            return np.array(list(self.quantile_histories[quantile]))
        return np.array([])

    def get_latest_quantiles(self):
        """获取最新的分位数值"""
        latest = {}
        for q_val in self.quantiles:
            if self.quantile_histories[q_val]:
                latest[q_val] = self.quantile_histories[q_val][-1]
            else:
                latest[q_val] = float('nan')
        return latest


class RangeTracker():
    def __init__(self, layer_idx, gla_cfg, max_batches=100000, log_interval=100):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.rank != 0:
            return

        # 解析实验配置
        range_experiment_dir = _parse_experiment_config(gla_cfg)

        self.layer_idx = layer_idx
        self.max_batches = max_batches
        self.log_interval = log_interval
        self.count = 0

        # 使用分位数统计类
        self.q_stats = TensorQuantileStats('Q', max_batches)
        self.k_stats = TensorQuantileStats('K', max_batches)
        self.steps = deque(maxlen=max_batches)

        self.log_text_dir = os.path.join(range_experiment_dir, 'text')
        os.makedirs(self.log_text_dir, exist_ok=True)

        self.log_pic_dir = os.path.join(range_experiment_dir, 'pic', f'layer_{self.layer_idx}')
        os.makedirs(self.log_pic_dir, exist_ok=True)

        self._init_log_file()

    def _init_log_file(self):
        self.text_file_name = os.path.join(self.log_text_dir, f"qk_quantiles_log_layer_{self.layer_idx}.txt")

        """初始化日志文件，写入头部信息"""
        with open(self.text_file_name, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("QK Quantile Distribution Tracking Log\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max batches: {self.max_batches}\n")
            f.write(f"Layer index: {self.layer_idx}\n")
            f.write(f"Quantiles tracked: {self.q_stats.quantiles}\n")
            f.write("=" * 80 + "\n\n")

    def _format_statistics_table(self, f, name, stats_obj, latest_values):
        """生成单个张量的统计表格"""
        f.write(f"  {name} Statistics Table:\n")
        
        # 表头 - 使用固定宽度格式化
        header = "    Batch"
        for q in stats_obj.quantiles:
            header += f"{int(q*100):>8d}%"
        f.write(header + "\n")

        # 分隔线
        separator_length = 9 + len(stats_obj.quantiles) * 9  # Batch列9字符 + 每个分位数列9字符
        f.write("    " + "-" * separator_length + "\n")
        
        # 数据行 - 使用固定宽度格式化
        data_line = f"    {self.count:5d}"
        for q in stats_obj.quantiles:
            value = latest_values.get(q, float('nan'))
            if not np.isnan(value):
                data_line += f"{value:>9.4f}"
            else:
                data_line += f"{'N/A':>9}"
        f.write(data_line + "\n")

    def write_stats(self):
        """将分位数统计信息写入文件"""
        try:
            with open(self.text_file_name, 'a', encoding='utf-8') as f:
                f.write(f"Batch {self.count} - {datetime.now().strftime('%H:%M:%S')} - layer {self.layer_idx}\n")
                
                # 获取最新的分位数值
                q_latest = self.q_stats.get_latest_quantiles()
                k_latest = self.k_stats.get_latest_quantiles()
                
                # 生成统计表格
                self._format_statistics_table(f, "Q", self.q_stats, q_latest)
                self._format_statistics_table(f, "K", self.k_stats, k_latest)
                
                # 显示全局最大最小值
                f.write(f"  Global Ranges:\n")
                f.write(f"    Q: Min={self.q_stats.global_min:.4f}, Max={self.q_stats.global_max:.4f}\n")
                f.write(f"    K: Min={self.k_stats.global_min:.4f}, Max={self.k_stats.global_max:.4f}\n")
                
                f.write("\n\n")

        except Exception as e:
            print(f"写入日志文件失败: {e}")
    
    def update(self, q, k):
        if self.rank != 0:
            return

        with torch.no_grad():
            self.q_stats.update(q)
            self.k_stats.update(k)
        
        self.steps.append(self.count)
        self.count += 1

        if self.count % self.log_interval == 0:
            self.draw_range()
            self.write_stats()

    def _plot_single_quantiles(self, ax, steps, quantile_data, name, stats_obj, quantiles_list, color_scheme='blue'):
        """绘制单个张量的分位数图 - 使用同色系颜色插值"""
        # 定义同色系的两个颜色端点（加深颜色便于查看）
        if color_scheme == 'blues':
            color_start = np.array([0.3, 0.6, 1.0])  # 中等蓝色
            color_end = np.array([0.0, 0.0, 0.6])    # 深蓝色
        else:  # reds
            color_start = np.array([1.0, 0.4, 0.4])  # 中等红色
            color_end = np.array([0.6, 0.0, 0.0])    # 深红色
        
        # 绘制所有分位数线
        for i, q_val in enumerate(quantiles_list):
            p_key = f'p{int(q_val*100):02d}'
            if p_key in quantile_data and len(quantile_data[p_key]) > 0:
                # 计算当前分位数的颜色插值
                ratio = i / (len(quantiles_list) - 1) if len(quantiles_list) > 1 else 0
                
                ax.plot(steps, quantile_data[p_key], 
                       color=color_start + ratio * (color_end - color_start), 
                       linewidth=2.0 if q_val == 0.5 else 1.5,
                       alpha=1.0 if q_val == 0.5 else 0.8)
                
                # 在线的末端添加标签
                if len(quantile_data[p_key]) > 0:
                    ax.annotate(f'{int(q_val*100)}%{" (median)" if q_val == 0.5 else ""}', 
                               xy=(steps[-1], quantile_data[p_key][-1]), 
                               xytext=(5, 0), 
                               textcoords='offset points',
                               fontsize=8, 
                               color=color_start + ratio * (color_end - color_start),
                               va='center')
        
        # 在最大值和最小值之间填充灰色区域
        if len(quantile_data['p00']) > 0 and len(quantile_data['p100']) > 0:
            ax.fill_between(steps, quantile_data['p00'], quantile_data['p100'], 
                           color='lightgray', alpha=0.3)
        
        # 绘制全局最大最小值参考线并标注数值
        if stats_obj.global_min != float("inf"):
            ax.axhline(y=stats_obj.global_min, color='gray', linestyle='--', 
                      alpha=0.7, linewidth=1)
            ax.text(0.02, stats_obj.global_min, f'Global Min: {stats_obj.global_min:.3f}', 
                   transform=ax.get_yaxis_transform(), 
                   fontsize=8, color='gray', va='bottom')
            
        if stats_obj.global_max != float("-inf"):
            ax.axhline(y=stats_obj.global_max, color='gray', linestyle='--', 
                      alpha=0.7, linewidth=1)
            ax.text(0.02, stats_obj.global_max, f'Global Max: {stats_obj.global_max:.3f}', 
                   transform=ax.get_yaxis_transform(), 
                   fontsize=8, color='gray', va='top')
        
        ax.set_title(f'{name} Values Quantile Distribution Over Time')
        ax.set_xlabel('Batches')
        ax.set_ylabel(f'{name} Values')
        ax.grid(True, alpha=0.3)

    def draw_range(self):
        """绘制qk值的分位数分布"""
        pic_file_name = os.path.join(self.log_pic_dir, f"qk_quantiles_history_batch_{self.count}.png")

        if len(self.steps) < 2:
            return
            
        try:
            fig = plt.figure(figsize=(12, 6))  # 调整为1x2布局
            
            # 获取历史数据
            steps = np.array(list(self.steps))
            q_quantiles = self.q_stats.get_history_arrays()
            k_quantiles = self.k_stats.get_history_arrays()
            
            # 子图1: Q值分位数分布
            ax1 = plt.subplot(1, 2, 1)
            self._plot_single_quantiles(ax1, steps, q_quantiles, 'Q', self.q_stats, self.q_stats.quantiles, 'blues')
            
            # 子图2: K值分位数分布
            ax2 = plt.subplot(1, 2, 2)
            self._plot_single_quantiles(ax2, steps, k_quantiles, 'K', self.k_stats, self.k_stats.quantiles, 'reds')
            
            plt.tight_layout()
            plt.savefig(str(pic_file_name), dpi=150, bbox_inches='tight')
            plt.close(fig)  # 显式关闭图形，释放内存
            
        except Exception as e:
            print(f"绘图失败: {e}")
            # 确保即使出错也关闭图形
            plt.close('all')


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
        # print(f"KAN spline_weight: {self.spline_weight.dtype}")

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

        grid = self.grid.unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, g+2k+1)
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

        # print(f"KAN forward input: {x.dtype}, shape: {x.shape}")
        bases = self.b_splines(x).detach()  # ND(g+k)

        # print(f"KAN forward spline_weight: {self.spline_weight.dtype}, bases: {bases.dtype}")
        output = torch.sum(bases * self.spline_weight, dim=-1)  # ND
        # print(f"KAN forward output: {output.dtype}")

        return output.view(original_shape).to(x.dtype)  # 保持原始形状和数据类型


def _parse_experiment_config(gla_cfg):
    """解析实验配置，生成range_experiment目录路径"""
    task_cfg = gla_cfg.task_cfg
    expt_name = task_cfg.get('expt_name')
    expt_time = task_cfg.get('time')
    
    if task_cfg.get('debug').get('flag'):
        range_experiment_dir = os.path.join(
            'range_experiment', f'{expt_name}', 'debug', f'{expt_time}'
        )
    else:
        range_experiment_dir = os.path.join(
            'range_experiment', f'{expt_name}', f'{expt_time}'
        )
    
    return range_experiment_dir


if __name__ == "__main__":
    pass