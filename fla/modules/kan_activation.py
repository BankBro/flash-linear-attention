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
from train.utils.logger_utils import logger


MIN_THRESHOLD = 0.03


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
    def __init__(self, layer_idx, gla_cfg, max_batches=100000, log_interval=1000):  # MY TODO: log_interval在debug模式下调成100
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.rank != 0:
            return

        # 解析实验配置
        range_experiment_dir = _parse_experiment_config(gla_cfg)

        self.layer_idx = layer_idx
        self.max_batches = max_batches
        self.log_interval = log_interval
        self.count = 0

        if gla_cfg.task_cfg.get('debug', {}).get('flag', False):
            self.log_interval = 100

        # 使用分位数统计类
        self.q_stats = TensorQuantileStats('Q', max_batches)
        self.k_stats = TensorQuantileStats('K', max_batches)
        self.steps = deque(maxlen=max_batches)

        self.log_text_dir = os.path.join(range_experiment_dir, 'text')
        os.makedirs(self.log_text_dir, exist_ok=True)

        self.log_pic_dir = os.path.join(range_experiment_dir, 'pic', f'layer_{self.layer_idx}')
        os.makedirs(self.log_pic_dir, exist_ok=True)
        
        # 为KAN激活函数创建单独目录
        self.kan_pic_dir = os.path.join(range_experiment_dir, 'activation', f'layer_{self.layer_idx}')
        os.makedirs(self.kan_pic_dir, exist_ok=True)

        self.split_activation_dir = os.path.join(range_experiment_dir, 'split_activation', f'layer_{self.layer_idx}')
        os.makedirs(self.split_activation_dir, exist_ok=True)

        # KAN激活函数存储
        self.kan_activations = {'q': None, 'k': None}

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
                self.draw_kan_activations()  # 绘制KAN激活函数

                device = self.kan_activations['q'].spline_weight.device
                self.plot_activation('Q', self.kan_activations['q'], device)
                self.plot_activation('K', self.kan_activations['k'], device)

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

    def set_kan_activations(self, kan_q=None, kan_k=None):
        """设置当前层的KAN激活函数"""
        if kan_q is not None:
            self.kan_activations['q'] = kan_q
        if kan_k is not None:
            self.kan_activations['k'] = kan_k
    
    def draw_kan_activations(self):
        """绘制Q和K的KAN激活函数到同一张图，包含spline_weight"""
        if not (self.kan_activations['q'] or self.kan_activations['k']):
            return  # 没有KAN激活函数，不绘制
            
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 绘制Q的KAN激活函数
            if self.kan_activations['q'] is not None:
                self._plot_single_kan_activation(ax1, self.kan_activations['q'], 'Q', 'blue')
                self._plot_spline_weights(ax3, self.kan_activations['q'], 'Q', 'blue')
            else:
                ax1.text(0.5, 0.5, 'No Q KAN\nActivation', 
                        transform=ax1.transAxes, ha='center', va='center', fontsize=12)
                ax1.set_title('Q KAN Activation (Not Available)')
                ax3.text(0.5, 0.5, 'No Q KAN\nWeights', 
                        transform=ax3.transAxes, ha='center', va='center', fontsize=12)
                ax3.set_title('Q Spline Weights (Not Available)')
            
            # 绘制K的KAN激活函数
            if self.kan_activations['k'] is not None:
                self._plot_single_kan_activation(ax2, self.kan_activations['k'], 'K', 'red')
                self._plot_spline_weights(ax4, self.kan_activations['k'], 'K', 'red')
            else:
                ax2.text(0.5, 0.5, 'No K KAN\nActivation', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
                ax2.set_title('K KAN Activation (Not Available)')
                ax4.text(0.5, 0.5, 'No K KAN\nWeights', 
                        transform=ax4.transAxes, ha='center', va='center', fontsize=12)
                ax4.set_title('K Spline Weights (Not Available)')
            
            plt.suptitle(f'Layer {self.layer_idx} - KAN Activations & Weights (Batch {self.count})', fontsize=14)
            plt.tight_layout()
            
            # 保存图片
            pic_file_name = os.path.join(self.kan_pic_dir, f"kan_activations_batch_{self.count}.png")
            plt.savefig(pic_file_name, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"绘制Layer {self.layer_idx} KAN激活函数失败: {e}")
            plt.close('all')
    
    def _plot_spline_weights(self, ax, kan_activation, name, color):
        """绘制spline_weight参数"""
        try:
            weights = kan_activation.spline_weight.detach().cpu().numpy()
            indices = np.arange(len(weights))
            
            ax.bar(indices, weights, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(f'{name} Spline Weights')
            ax.set_xlabel('Weight Index')
            ax.set_ylabel('Weight Value')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, w in enumerate(weights):
                ax.text(i, w + 0.01 * (max(weights) - min(weights)), f'{w:.3f}', 
                       ha='center', va='bottom' if w >= 0 else 'top', fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{name} Weights\nError: {str(e)[:20]}...', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(f'{name} Spline Weights (Error)')

    def _plot_single_kan_activation(self, ax, kan_activation, name, color):
        """绘制单个KAN激活函数"""
        try:
            # 使用grid_range作为采样区间
            if name == 'Q':
                x_min, x_max = self.q_stats.global_min, self.q_stats.global_max
            elif name == 'K':
                x_min, x_max = self.k_stats.global_min, self.k_stats.global_max

            x_plot = torch.linspace(x_min, x_max, 100, device=kan_activation.grid.device)
            x_plot = x_plot.unsqueeze(0).unsqueeze(-1)  # 1, 100, 1
            
            with torch.no_grad():
                y_plot = kan_activation(x_plot).cpu().numpy().squeeze()
            x_plot_np = x_plot.cpu().numpy().squeeze()

            ax.plot(x_plot_np, y_plot, color=color, linewidth=2)
            ax.set_title(f'{name} KAN Activation')
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{name} KAN\nError: {str(e)[:20]}...', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(f'{name} KAN Activation (Error)')
    
    # 可视化激活函数
    def plot_activation(self, name, kan_layer, device):
        B, L, D = 1, 100, 1
        num_points = B * L * D

        if name == 'Q':
            x_min, x_max = self.q_stats.global_min, self.q_stats.global_max
        elif name == 'K':
            x_min, x_max = self.k_stats.global_min, self.k_stats.global_max

        x = (
            torch.linspace(x_min, x_max, num_points)
            .unsqueeze(0)
            .unsqueeze(-1)
            .reshape(B, L, D)
            .to(device)
        )

        with torch.no_grad():
            y1 = kan_layer.forward_1(x)
            y2 = kan_layer.forward_2(y1)
            y3 = kan_layer.forward_3(y2)

        x_np = x.flatten().cpu().numpy()
        y1_np = y1.flatten().cpu().numpy()
        y2_np = y2.flatten().cpu().numpy()
        y3_np = y3.flatten().cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 第一个子图 - Stage 1
        axes[0, 0].plot(x_np, y1_np, 'b-', linewidth=2)
        axes[0, 0].set_title("KAN Activation - Stage 1")
        axes[0, 0].set_xlabel("Input x")
        axes[0, 0].set_ylabel("Output y1")
        axes[0, 0].grid(True)
        
        # 第二个子图 - Stage 2
        axes[0, 1].plot(y1_np, y2_np, 'r-', linewidth=2)
        axes[0, 1].set_title("KAN Activation - Stage 2")
        axes[0, 1].set_xlabel("Input y1")
        axes[0, 1].set_ylabel("Output y2")
        axes[0, 1].grid(True)
        
        # 第三个子图 - Stage 3
        axes[1, 0].plot(y2_np, y3_np, 'g-', linewidth=2)
        axes[1, 0].set_title("KAN Activation - Stage 3")
        axes[1, 0].set_xlabel("Input y2")
        axes[1, 0].set_ylabel("Output y3")
        axes[1, 0].grid(True)
        
        # 第四个子图 - 完整映射 (x -> y3)
        axes[1, 1].plot(x_np, y3_np, 'm-', linewidth=2)
        axes[1, 1].set_title("Complete KAN Mapping (x → y3)")
        axes[1, 1].set_xlabel("Input x")
        axes[1, 1].set_ylabel("Final Output y3")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        file_name = os.path.join(self.split_activation_dir, f"{name}_split_activations_batch_{self.count}.png")
        plt.savefig(file_name)
        plt.close()


class KANActivation(torch.nn.Module):
    def __init__(
        self,
        gla_cfg,
        grid_size=8,
        spline_order=3,
        grid_range=[-1.0, 1.0],  # 有效区域, 不包括两端扩展区域
        scale_factor=1.0,  # 映射范围因子
        # gradient_amplify_factor=5.0,  # 梯度放大倍数
        gradient_amplify_factor=1.0,  # 梯度放大倍数
        init_strategy='pure_linear',  # 初始化策略
        noise_scale=0.05,  # 噪声比例
    ):
        super(KANActivation, self).__init__()
        self.task_cfg = gla_cfg.task_cfg

        self.grid_size = grid_size  # g
        self.spline_order = spline_order  # k
        self.grid_range = grid_range  # 保存输入值域范围

        self.gradient_amplify_factor = self.task_cfg.expt_params.get('grad_amplifier', gradient_amplify_factor)
        self.init_strategy = self.task_cfg.expt_params.get('init_strategy', init_strategy)
        self.noise_scale = noise_scale

        self.identity_loss_weight = 1.0
        self.range_loss_weight = 10.0
        self.non_linear_loss_weight = 0.001
        self.current_loss = 0.0
        self.aux_loss = 0.0
        self.convex_concave_loss = 0.0

        self.spline_basis_mode = self.task_cfg.expt_params.get('spline_basis_mode', 'original')
        self.kan_loss_flag = self.task_cfg.expt_params.get('kan_loss_flag', False)
        self.aux_loss_flag = self.task_cfg.expt_params.get('aux_loss_flag', False)
        self.convex_concave_mode = self.task_cfg.expt_params.get('convex_concave_mode', None)
        # logger.info(f"spline_basis_mode: {self.spline_basis_mode}, kan_loss_flag: {self.kan_loss_flag}, aux_loss_flag: {self.aux_loss_flag}, convex_concave_mode: {self.convex_concave_mode}")

        # logger.info(f"init_strategy: {self.init_strategy}")
        # logger.info(f"grad_amplifier: {self.gradient_amplify_factor}")
        # logger.info(f"re_calculate: {self.re_calculate}")
        # logger.info(f"kan_loss_flag: {self.kan_loss_flag}")
        # logger.info(f"aux_loss_flag: {self.aux_loss_flag}")

        # 创建B样条网格
        # grid_range 是有效定义域，网格间距基于这个有效域
        h = (grid_range[1] - grid_range[0]) / grid_size
        
        # 构建完整网格：左扩展 + 有效域 + 右扩展
        # 从 (grid_range[0] - spline_order * h) 开始，到 (grid_range[1] + spline_order * h) 结束
        grid_start = grid_range[0] - spline_order * h
        grid = torch.arange(grid_size + 2 * spline_order + 1) * h + grid_start  # (g+2k+1,)
        self.register_buffer("grid", grid)

        # 可学习的样条权重
        self.spline_weight = torch.nn.Parameter(
            self._init_spline_weights()
        )

        # 可学习的映射范围因子
        # self.scale_factor = torch.nn.Parameter(torch.tensor(scale_factor))
        self.scale_factor = scale_factor
        
        # 注册梯度放大钩子
        self._register_gradient_hook()  # MY TODO: 分组成功后考虑删掉
        # print(f"KAN spline_weight: {self.spline_weight.dtype}")

    def _init_spline_weights(self):
        """根据策略初始化样条权重"""
        if self.init_strategy == 'linear_with_noise':
            return self._init_linear_with_noise()
        elif self.init_strategy == 'pure_linear':
            return self._init_pure_linear()
        else:
            # 默认使用带噪声的线性初始化
            return self._init_linear_with_noise()
    
    def _init_linear_with_noise(self):
        """改进的线性映射初始化，添加噪声干扰"""
        num_coeffs = self.grid_size + self.spline_order
        
        # 基础线性权重 - 直接调用纯线性初始化方法
        weights = self._init_pure_linear()
        
        # 添加噪声干扰，鼓励学习非线性变换
        range_width = self.grid_range[1] - self.grid_range[0]
        noise_amplitude = range_width * self.noise_scale  # 噪声幅度
        noise = torch.randn(num_coeffs) * noise_amplitude
        weights = weights + noise
        
        return weights
    
    def _init_pure_linear(self):
        num_coeffs = self.grid_size + self.spline_order
        
        control_points = torch.linspace(
            (self.grid[0] + self.grid[self.spline_order + 1]) / 2,
            (self.grid[-self.spline_order - 2] + self.grid[- 1]) / 2,
            num_coeffs
        )
        
        return control_points

    def b_spline_basis_re_calculate(self, x: torch.Tensor):  # BLD
        """计算B样条基函数"""
        assert x.dim() == 3, "Input tensor must have exactly 3 dimensions (BLD)."
        x = x.reshape(-1, x.shape[-1]).unsqueeze(-1)  #  ND1

        grid = self.grid.unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, g+2k+1)
        bases = ((x >= grid[:,:,:-1]) & (x < grid[:,:,1:])).to(x.dtype)  # ND(g+2k+1)

        # 递归计算B样条 (Cox-de Boor 公式)
        for k in range(1, self.spline_order + 1):
            def _forward(bases, x, grid, k):
                left_denom = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
                right_denom = grid[:, :, (k + 1):] - grid[:, :, 1:(-k)]
                
                bases = (
                    (x - grid[:, :, :-(k + 1)]) / left_denom * bases[:, :, :-1] +
                    (grid[:, :, (k + 1):] - x) / right_denom * bases[:, :, 1:]
                )

                return bases

            # 使用checkpointing来节省内存
            bases = torch.utils.checkpoint.checkpoint(_forward, bases, x, grid, k, use_reentrant=False)
        
        assert bases.shape == (x.shape[0], x.shape[1], self.grid_size + self.spline_order), "B-spline bases shape mismatch."
        return bases  # ND(g+k)

    def b_spline_basis_detach(self, x: torch.Tensor):  # BLD
        """计算B样条基函数"""
        assert x.dim() == 3, "Input tensor must have exactly 3 dimensions (BLD)."
        x = x.reshape(-1, x.shape[-1]).unsqueeze(-1)  #  ND1

        grid = self.grid.unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, g+2k+1)
        bases = ((x >= grid[:,:,:-1]) & (x < grid[:,:,1:])).to(x.dtype)  # ND(g+2k+1)

        # 递归计算B样条 (Cox-de Boor 公式)
        for k in range(1, self.spline_order + 1):
            left_denom = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            right_denom = grid[:, :, (k + 1):] - grid[:, :, 1:(-k)]
            
            bases = (
                (x - grid[:, :, :-(k + 1)]) / left_denom * bases[:, :, :-1] +
                (grid[:, :, (k + 1):] - x) / right_denom * bases[:, :, 1:]
            )
        
        assert bases.shape == (x.shape[0], x.shape[1], self.grid_size + self.spline_order), "B-spline bases shape mismatch."
        return bases  # ND(g+k)
    
    def b_spline_basis(self, x: torch.Tensor):
        if self.spline_basis_mode == "re_calculate":
            return self.b_spline_basis_re_calculate(x)
        
        elif self.spline_basis_mode == "detach":
            return self.b_spline_basis_detach(x).detach()
        
        elif self.spline_basis_mode == "original":
            return self.b_spline_basis_detach(x)
        
        else:
            raise ValueError(f"spline_basis_mode {self.spline_basis_mode} is not supported")

    def b_spline(self, x: torch.Tensor):
        original_shape = x.shape

        basis = self.b_spline_basis(x)#.detach()  # ND(g+k)
        output = torch.sum(basis * self.spline_weight, dim=-1)  # ND

        # 超出有效范围的值映射到边界(-1, 1)
        # output = torch.clamp(
        #     output,
        #     min=self.grid_range[0] + MIN_THRESHOLD,
        #     max=self.grid_range[1] - MIN_THRESHOLD
        # )

        return output.view(original_shape).to(x.dtype)
    
    def forward_1(self, x: torch.Tensor):
        # 1. 反正切映射到 (-1, 1)
        x = torch.arctan(x) * (2 / torch.pi)
        return x
    
    def forward_2(self, x: torch.Tensor):
        # 2. B样条非线性映射
        x = self.b_spline(x)
        return x
    
    def forward_3(self, x: torch.Tensor):
        # 3. 正切映射回原始范围
        x = torch.tan(x * torch.pi / 2)
        return x

    def forward(self, x: torch.Tensor):
        """前向传播：反正切 -> B样条 -> 正切"""
        original_dtype = x.dtype

        # 1. 反正切映射到 (-A, A)
        tmp = self.forward_1(x)

        # 2. B样条非线性映射
        tmp = self.forward_2(tmp)

        if self.kan_loss_flag:
            self.current_loss = self.b_spline_loss(x, tmp.view(x.shape))

        # logger.info(f"KAN FORWARD.")

        if self.aux_loss_flag:
            # logger.info("auxiliary loss flag is True")
            self.aux_loss = self.auxiliary_loss(x.device)  # 计算辅助损失
        
        if self.convex_concave_mode == 'convex':
            # logger.info("convex mode is True")
            self.convex_concave_loss = self.convex_loss(x.device)
        elif self.convex_concave_mode == 'concave':
            # logger.info("concave mode is True")
            self.convex_concave_loss = self.concave_loss(x.device)
        else:
            # logger.info("convex mode and concave mode are False")
            self.convex_concave_loss = 0.0
        
        # logger.info("KAN forward completed.")

        # 3. 正切映射回原始范围
        output = self.forward_3(tmp)

        return output.to(original_dtype)
    
    def identity_loss(self, x: torch.Tensor, output: torch.Tensor):
        """
        计算恒等映射损失，鼓励样条曲线在靠近 -1 和 1 的位置接近恒等映射。
        """
        # identity_loss = torch.mean(((x - output) ** 2) * (x ** 2))
        weight = torch.tan(x * torch.pi / 2 * 0.98) * x * 0.1
        identity_loss = torch.mean(((x - output) ** 2) * weight)
        return identity_loss

    def range_loss(self, x: torch.Tensor, output: torch.Tensor):
        """
        计算范围损失，鼓励样条曲线保持在 (-1, 1) 的范围。
        """
        range_loss = torch.mean(
            torch.where(
                (output <= -1) | (output >= 1),
                (output - x) ** 2,#* (output - x).abs(),  # 超出范围的平方差
                torch.zeros_like(output)  # 在范围内的部分不计算损失
            )
        )
        return range_loss

    def smooth_loss(self, output: torch.Tensor):
        """
        计算平滑损失，鼓励样条曲线的平滑性。
        """
        # 计算二阶差分
        second_derivative = torch.diff(output, n=2, dim=-1)
        # print(f"Second Derivative: {second_derivative}, shape: {second_derivative.shape}, output shape: {output.shape}")
        smooth_loss = torch.mean(second_derivative ** 2)
        return smooth_loss

    def non_linear_loss(self, x: torch.Tensor, output: torch.Tensor):
        """
        鼓励输出具有曲率（非线性）, 通过二阶差分衡量, 希望输出曲线有弯曲（即二阶导数不为零）。
        """
        return -self.smooth_loss(output)
    
    def b_spline_loss(self, x: torch.Tensor, output: torch.Tensor):
        """
        计算B样条损失，鼓励样条曲线在有效范围内的平滑性和非线性。
        """
        identity_loss = self.identity_loss(x, output)
        range_loss = self.range_loss(x, output)
        # non_linear_loss = self.non_linear_loss(x, output)

        total_loss = (
            identity_loss * self.identity_loss_weight +
            range_loss * self.range_loss_weight
            # non_linear_loss * self.non_linear_loss_weight
        )

        # print(f"Identity Loss: {identity_loss:.4f}, Range Loss: {range_loss:.4f}, Non-Linear Loss: {non_linear_loss:.4f}")

        return total_loss

    def auxiliary_loss(self, device):
        """
        Compute the supplemental loss for the given input and output tensors.
        """
        # Compute the loss
        x = torch.linspace(-0.9, 0.9, 1024, device=device, requires_grad=True).unsqueeze(0).unsqueeze(-1)  # BLD
        output = self.b_spline(x)

        aux_loss = self.b_spline_loss(x, output)
        return aux_loss
    
    def convex_loss(self, device, adaptive_weight=True):
        # 凸函数, 如y=x^2, 二阶导>=0
        sample_num = 1024

        x = torch.linspace(-0.9, 0.9, sample_num, device=device, requires_grad=True).unsqueeze(0).unsqueeze(-1)
        output = self.b_spline(x)

        h = (0.9 - (-0.9)) / (sample_num - 1)
        diff2 = torch.diff(output, n=2, dim=1) / (h * h)
        
        if adaptive_weight:
            # 自适应权重：当已经足够凸时，减小损失权重
            current_convexity = torch.mean(diff2)
            weight = torch.exp(-torch.abs(current_convexity) / 0.1)  # 越凸权重越小
            convex_loss = weight * torch.mean(-diff2)
        else:
            convex_loss = torch.mean(-diff2)
        
        return convex_loss
    
    def concave_loss(self, device, adaptive_weight=True):
        # 凹函数, 如y=-x^2, 二阶导<=0
        sample_num = 1024
        
        x = torch.linspace(-0.9, 0.9, sample_num, device=device, requires_grad=True).unsqueeze(0).unsqueeze(-1)
        output = self.b_spline(x)

        h = (0.9 - (-0.9)) / (sample_num - 1)
        diff2 = torch.diff(output, n=2, dim=1) / (h * h)
        
        if adaptive_weight:
            # 自适应权重：当已经足够凹时，减小损失权重
            current_concavity = torch.mean(diff2)
            weight = torch.exp(-torch.abs(current_concavity) / 0.1)  # 越凹权重越小
            concave_loss = weight * torch.mean(diff2)
        else:
            concave_loss = torch.mean(diff2)
        
        return concave_loss

    def _register_gradient_hook(self):
        """为spline_weight注册梯度放大钩子"""
        def gradient_amplify_hook(grad):
            """
            梯度放大钩子函数, 在反向传播时自动调用, 对spline_weight的梯度进行放大
            """
            if grad is not None:
                amplified_grad = grad * self.gradient_amplify_factor
                return amplified_grad
            return grad
        
        # 注册钩子 - Lightning框架会自动管理这个钩子的调用
        self.gradient_hook_handle = self.spline_weight.register_hook(gradient_amplify_hook)


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