import sys
import os
import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl

from lightning.pytorch import (
    Trainer,
    LightningModule,
    LightningDataModule,
    Callback
)
from lightning.pytorch.loggers import WandbLogger
from typing import List
import torch
import time

from rich import print as rich_print
from rich.tree import Tree
from rich.console import Console

from train.utils.logger_utils import logger
from train.utils.train_time import TIME_FORMAT


DEFAULT_GLOBAL_SEED = 42


def insert_into_path(origin_path: str, insert_text: str, insert_position=0, insert_last=False):
    """
    在路径的指定位置插入字符串
    
    Args:
        path: 原始路径
        insert_text: 要插入的字符串
        insert_position: 要插入的位置, 位置从0开始计算
        insert_last: 是否在最后插入, 如果为True, 则忽略insert_position参数
    
    Returns:
        修改后的路径
    """
    assert isinstance(origin_path, str), "Path must be a string."

    path_parts = origin_path.split('/')
    
    if insert_last:
        path_parts.append(insert_text)
    else:
        path_parts.insert(insert_position, insert_text)

    return '/'.join(path_parts)

def process_config(config: DictConfig, datamodule: LightningDataModule = None) -> DictConfig:
    """
    Process the config to add additional information.
    """
    # Convert DictConfig to a regular dictionary for easier manipulation
    config = OmegaConf.to_container(config, resolve=True)

    # Process debug parameters
    if config['debug'].get('flag', False):
        debug_cfg = config['debug']
        logger.info("Debug mode enabled, applying debug configurations")

        config['trainer']['accumulate_grad_batches'] = debug_cfg['accumulate_grad_batches']
        config['trainer']['val_check_interval'] = debug_cfg['val_check_interval']

        if 'model_checkpoint' in config['callbacks']:
            config['callbacks']['model_checkpoint']['dirpath'] = insert_into_path(
                config['callbacks']['model_checkpoint']['dirpath'], 'debug', insert_position=-1
            )
        else:
            logger.warning(f"Config doesn't have model_checkpoint callback, ignoring.")

        if 'WandbLogger' in config['logger']['_target_']:
            config['logger']['save_dir'] = insert_into_path(
                config['logger']['save_dir'], 'debug', insert_position=-1
            )
        
        elif 'SwanLabLogger' in config['logger']['_target_']:
            config['logger']['logdir'] = insert_into_path(
                config['logger']['logdir'], 'debug', insert_position=-1
            )

        config['datamodule']['ids_num_train'] = debug_cfg['ids_num_train']
        config['datamodule']['ids_num_val'] = debug_cfg['ids_num_val']
        config['datamodule']['ids_num_test'] = debug_cfg['ids_num_test']
        
        logger.info(f"Debug dataset sizes - Train: {debug_cfg['ids_num_train']}, "
                   f"Val: {debug_cfg['ids_num_val']}, Test: {debug_cfg['ids_num_test']}")
    
    else:
        config['datamodule']['ids_num_train'] = datamodule.ids_num_train
        config['datamodule']['ids_num_val'] = datamodule.ids_num_val
        config['datamodule']['ids_num_test'] = datamodule.ids_num_test

    assert datamodule.dataset_train is None, "Dataset should not be loaded."
    datamodule.setup()
    dataset_train_sample_num = len(datamodule.dataset_train)
    datamodule.dataset_train = None
    datamodule.dataset_val = None
    datamodule.dataset_test = None

    # Update the strategy
    logger.info(f"Trainer devices: {config['trainer']['devices']}")
    n_device = len(config['trainer']['devices'])
    if n_device > 1:
        config['trainer']['strategy'] = 'ddp'
        logger.info(f"Multi-device training detected, using DDP strategy with {n_device} devices")

    # Calculate global batch size and other parameters
    global_batch_size = (
        config['datamodule']['datamodule_cfg']['batch_size_per_gpu']
        * n_device
        * config['trainer']['num_nodes']
    )

    batches_per_epoch = dataset_train_sample_num // global_batch_size
    total_batches = batches_per_epoch * config['trainer']['max_epochs']
    max_steps = total_batches // config['trainer']['accumulate_grad_batches']

    logger.info(f"Training parameters calculated:")
    logger.info(f"  Dataset size: {dataset_train_sample_num}")
    logger.info(f"  Global batch size: {global_batch_size}")
    logger.info(f"  Batches per epoch: {batches_per_epoch}")
    logger.info(f"  Total batches: {total_batches}")
    logger.info(f"  Max steps: {max_steps}")

    calculate_params_cfg = config['calculate_params']
    calculate_params_cfg['dataset_train_sample_num'] = dataset_train_sample_num
    calculate_params_cfg['global_batch_size'] = global_batch_size  # TODO: 计算错误？
    calculate_params_cfg['batches_per_epoch'] = batches_per_epoch
    calculate_params_cfg['total_batches'] = total_batches
    calculate_params_cfg['max_steps'] = max_steps
    config['calculate_params'] = calculate_params_cfg
    config['trainer']['max_steps'] = max_steps

    # Update the learning rate scheduler configuration
    warmup_t = int(config['lr_scheduler']['timm_cosine']['warmup_t'] * max_steps)
    t_initial = max_steps - warmup_t
    config['lr_scheduler']['timm_cosine']['warmup_t'] = warmup_t
    config['lr_scheduler']['timm_cosine']['t_initial'] = t_initial
    
    logger.info(f"Learning rate scheduler updated:")
    logger.info(f"  Warmup steps: {warmup_t}")
    logger.info(f"  Initial steps: {t_initial}")

    # Update the val_check_interval and checkpoint every_n_train_steps
    if 'model_checkpoint' in config['callbacks']:
        val_check_interval = int(
            config['trainer']['val_check_interval'] * batches_per_epoch
        )
        every_n_train_steps = int(
            val_check_interval / config['trainer']['accumulate_grad_batches']
        )
        config['callbacks']['model_checkpoint']['every_n_train_steps'] = every_n_train_steps
        # TODO: 改为每个val结束后保存一次ckpt

        logger.info(f"Validation and checkpoint intervals:")
        logger.info(f"  Validation check interval: {val_check_interval} batches")
        logger.info(f"  Checkpoint every: {every_n_train_steps} steps")

    # Convert back to DictConfig
    return OmegaConf.create(config)

def dict_to_tree(d, tree=None):
    if tree is None:
        tree = Tree("🌟 [bold magenta]Config")

    for key, value in d.items():
        if isinstance(value, (dict, DictConfig)):
            branch = tree.add(f"[bold blue]{key}")
            dict_to_tree(value, branch)
        else:
            # 根据值的类型使用不同的颜色
            if isinstance(value, (int, float)):
                tree.add(f"[bold green]{key}:[/bold green] [yellow]{value}")
            elif value is None:
                tree.add(f"[bold green]{key}:[/bold green] [dim]None")
            else:
                tree.add(f"[bold green]{key}:[/bold green] [white]{value}")
    return tree

def print_config(config: DictConfig):
    """使用 rich 显示配置，同时记录到日志"""
    logger.info("Displaying configuration tree")
    
    # 使用 rich 显示配置树
    rich_print(f"\n{'='*50}")
    console = Console()
    tree = dict_to_tree(OmegaConf.to_container(config, resolve=True))
    console.print(tree)
    rich_print(f"{'='*50}\n")
    
    # 同时记录主要配置到日志
    logger.info("Key configuration parameters:")
    logger.info(f"  Experiment name: {config.get('expt_name', 'default_expt')}")
    logger.info(f"  Max epochs: {config.get('trainer', {}).get('max_epochs', 'N/A')}")
    logger.info(f"  Max steps: {config.get('trainer', {}).get('max_steps', 'N/A')}")
    logger.info(f"  Devices: {config.get('trainer', {}).get('devices', 'N/A')}")

def save_config(config: DictConfig):
    """保存配置文件"""
    expt_name = config.get('expt_name', 'default_expt')
    expt_time = config.get('time', 'default_time')
    debug_flag = config.get('debug', {}).get('flag', False)

    if debug_flag:
        base_dir = os.path.join("logs", "configs", expt_name, "debug", expt_time)
    else:
        base_dir = os.path.join("logs", "configs", expt_name, expt_time)

    os.makedirs(base_dir, exist_ok=True)

    config_path = os.path.join(base_dir, "config.yaml")
    OmegaConf.save(config, config_path)
    
    logger.info(f"Configuration saved to: {config_path}")

def set_project_root():
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(project_root, "..")
    project_root = os.path.abspath(project_root)

    if project_root not in sys.path:
        sys.path.append(project_root)

def check_gpu_availability():
    """检查环境并设置必要的配置"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        raise RuntimeError("CUDA is not available.")
    
    logger.info("CUDA is available, environment check passed")

def check_and_set_env():
    os.environ["HYDRA_FULL_ERROR"] = "1"

    set_project_root()
    check_gpu_availability()

def get_generator_seed(config: DictConfig):
    """获取并设置随机种子"""
    if config.get('seed'):
        global_seed = pl.seed_everything(config.seed, workers=True)
        logger.info(f"Set random seed from config: {config.seed}")
    else:
        global_seed = pl.seed_everything(DEFAULT_GLOBAL_SEED, workers=True)
        logger.info(f"Set default random seed: {DEFAULT_GLOBAL_SEED}")
    
    logger.info(f"Global seed: {global_seed}")
    return global_seed

def training(
        trainer: Trainer,
        task_module: LightningModule,
        datamodule: LightningDataModule
    ):
    """执行训练过程"""
    start_time = time.strftime(TIME_FORMAT, time.localtime())
    logger.info(f"Training started at: {start_time}")

    trainer.fit(model=task_module, datamodule=datamodule)
    logger.info("Training completed successfully")

    end_time = time.strftime(TIME_FORMAT, time.localtime())
    logger.info(f"Training finished at: {end_time}")

    start_timestamp = time.mktime(time.strptime(start_time, TIME_FORMAT))
    end_timestamp = time.mktime(time.strptime(end_time, TIME_FORMAT))
    duration = end_timestamp - start_timestamp

    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")

@hydra.main(version_base="1.3.2", config_path="./configs/", config_name="config.yaml")
def main(config: DictConfig):
    logger.info("Starting main training process")
    
    hydra_cfg = HydraConfig.get()
    logger.info(f"config name: {hydra_cfg.job.config_name}, config: {config}")
    
    global_seed = get_generator_seed(config)

    logger.info(f"Instantiating datamodule <{config.datamodule.datamodule_cfg._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule.datamodule_cfg,
        cfg = config,
        global_seed = global_seed,
        _recursive_ = False
    )

    config = process_config(config, datamodule)
    datamodule.cfg = config
    
    print_config(config)
    save_config(config)

    logger.info(f"Instantiating task_module <{config.task_module._target_}>")
    task_module: LightningModule = hydra.utils.instantiate(
        config.task_module,
        cfg = config,
        vocab_size = datamodule.vocab_size,
        _recursive_ = False
    )

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger.info(f"Instantiating logger <{config.logger._target_}>")
    metrics_logger = hydra.utils.instantiate(config.logger)

    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=metrics_logger,
        callbacks=callbacks,
    )

    for k, v in sys.modules.items():
        if 'logger_utils' in k or 'train_time' in k:
            print(f"{k}, {v}")

    logger.info("Starting training process")
    training(trainer=trainer, task_module=task_module, datamodule=datamodule)

    logger.info("Starting testing process")
    trainer.test(model=task_module, datamodule=datamodule, ckpt_path="best")
    logger.info("Testing completed successfully")
        
    logger.info("All processes completed successfully")


if __name__ == "__main__":
    # # print(f"sys.modules = {sys.modules}")
    # # /home/lyj/project/flash-linear-attention/train/main.py
    check_and_set_env()
    main()