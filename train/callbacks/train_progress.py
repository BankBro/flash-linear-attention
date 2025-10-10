import time
from lightning.pytorch import Callback
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.core.module import LightningModule
from train.utils.logger_utils import logger


class TrainProgressCallback(Callback):
    """自定义回调，用于在训练时按照指定频率输出进度信息"""
    
    def __init__(self, log_frequency_ratio: float = 0.1):
        """
        Args:
            log_frequency_ratio: 输出频率比例，相对于总batch数量 (默认 10%)
        """
        super().__init__()
        self.log_frequency_ratio = log_frequency_ratio
        self.log_interval = None
        self.last_log_step = 0
        # 直接保存训练参数
        self.batches_per_epoch = None
        self.total_batches = None
        self.max_steps = None
        # 跟踪当前epoch的batch计数
        self.current_epoch_batch_count = 0
        self.last_epoch = -1
        self.last_epoch_start_time = None
        self.start_time = None  # 添加训练开始时间
        
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """训练开始时初始化"""
        self.start_time = time.time()  # 初始化训练开始时间
        
        # 直接从trainer获取参数并保存为类属性
        self.batches_per_epoch = trainer.num_training_batches if hasattr(trainer, 'num_training_batches') else None
        self.total_batches = trainer.max_steps * trainer.accumulate_grad_batches
        self.max_steps = trainer.max_steps
        
        # 计算日志输出间隔（基于每个epoch的batch数量）
        self.log_interval = max(1, int(self.batches_per_epoch * self.log_frequency_ratio))

        logger.info(f"Training progress callback initialized:")
        logger.info(f"  Log frequency ratio: {self.log_frequency_ratio:.1%}")
        logger.info(f"  Log interval: every {self.log_interval} batches per epoch")
        logger.info(f"  Batches per epoch: {self.batches_per_epoch}")
        logger.info(f"  Total batches: {self.total_batches}")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Accumulate grad batches: {trainer.accumulate_grad_batches}")

    def on_train_epoch_start(self, trainer, pl_module):
        logger.info(f"{'-'*20} Starting Epoch {trainer.current_epoch + 1} {'-'*20}")
        self.last_epoch_start_time = time.time()  # 记录当前epoch开始时间
        self.current_epoch_batch_count = 0  # 重置当前epoch的batch计数
        self.last_epoch = trainer.current_epoch  # 保存当前epoch索引

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        """训练batch结束时检查是否需要输出进度"""
        # 增加当前epoch的batch计数
        self.current_epoch_batch_count += 1

        # print(f"output: {outputs}")
        # print(f"logged_metrics: {trainer.logged_metrics}")
        
        # 检查是否到了输出时间（基于每个epoch的batch数量）
        if (self.current_epoch_batch_count % self.log_interval == 0
            or self.current_epoch_batch_count == 1
            or self.current_epoch_batch_count == trainer.num_training_batches
        ):
            self._train_log(trainer, trainer.global_step, outputs, batch_idx)
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """训练结束，输出总体统计信息"""
        end_time = time.time()
        total_training_time = end_time - self.start_time
        training_time_str = self._format_time(total_training_time)

        total_epochs = trainer.current_epoch + 1
        total_steps = trainer.global_step
        total_batches_processed = total_steps * trainer.accumulate_grad_batches
        
        logger.info("Training completed! Summary:")
        logger.info(f"  Total epochs: {total_epochs}/{trainer.max_epochs}")
        logger.info(f"  Total steps: {total_steps}/{self.max_steps}")
        logger.info(f"  Total batches processed: {total_batches_processed}/{self.total_batches}")
        logger.info(f"  Total training time: {training_time_str}")
        
        if total_steps > 0:
            avg_time_per_step = total_training_time / total_steps
            logger.info(f"  Average time per step: {avg_time_per_step:.2f}s")
    
    def _train_log(self, trainer: Trainer, current_step: int, outputs, batch_idx: int):
        """输出训练进度信息"""
        if outputs is None:
            return

        # 计算当前epoch的剩余时间
        current_time = time.time()
        elapsed_time_epoch = current_time - self.last_epoch_start_time
        total_time_epoch = elapsed_time_epoch * (self.batches_per_epoch / self.current_epoch_batch_count)
        remaining_time_epoch = total_time_epoch - elapsed_time_epoch
        # 格式化时间
        elapsed_str = self._format_time(elapsed_time_epoch)
        remaining_str = self._format_time(remaining_time_epoch)

        # 当前epoch的进度
        epoch_progress = self.current_epoch_batch_count / self.batches_per_epoch

        # 获取当前损失和困惑度
        total_loss = outputs.get('total_loss', None)
        kan_loss = outputs.get('kan_loss', None)
        main_loss = outputs.get('main_loss', None)
        ppl = outputs.get('ppl', None)
        # print(f"print, Loss: {loss}, Kan Loss: {kan_loss}, Main Loss: {main_loss}, PPL: {ppl}")
        
        # 输出进度信息 - 只显示当前epoch内的进度
        progress_info = (
            f"Epoch {trainer.current_epoch + 1}|"
            f"Batch[{self.current_epoch_batch_count}/{self.batches_per_epoch}]"
            f"({epoch_progress:.0%})|"
            f"Step[{current_step}/{self.max_steps}]|"
            f"Loss:{total_loss:.4f} "
            # f"MainLoss:{main_loss:.4f} "
            f"KanLoss:{kan_loss:.4f}|"
            f"PPL:{ppl:.2f}|"
            f"{remaining_str}>{elapsed_str}"
        )
        
        logger.info(f"{progress_info}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m{secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:.0f}h{minutes:.0f}m{secs:.0f}s"
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """验证开始"""
        logger.info("Validation started...")
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """验证结束，输出验证结果"""
        # 获取最新的验证指标
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val/loss', None)
        val_ppl = metrics.get('val/ppl', None)
        
        if val_loss is not None and val_ppl is not None:
            logger.info(f"Validation completed - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
        else:
            logger.info("Validation completed")
    
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """测试开始"""
        logger.info(f"{'-'*20} Testing started... {'-'*20}")
    
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """测试结束，输出测试结果"""
        # 获取最新的测试指标
        metrics = trainer.callback_metrics
        test_loss = metrics.get('test/loss', None)
        test_ppl = metrics.get('test/ppl', None)
        
        if test_loss is not None and test_ppl is not None:
            logger.info(f"Testing completed - Loss: {test_loss:.4f}, PPL: {test_ppl:.2f}")
        
        logger.info(f"{'-'*20} Testing completed {'-'*20}")
