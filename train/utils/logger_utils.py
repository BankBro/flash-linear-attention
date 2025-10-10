"""
训练专用日志工具模块
创建独立的训练日志记录器, 避免与Lightning框架冲突
"""
import logging
import sys
import os
from typing import Optional
from train.utils.train_time import START_TIME_STR

LOGGER_NAME = "train_logger"
LOG_PATH = f"./logs/training"
LOG_FILE = f"{LOG_PATH}/{START_TIME_STR}.log"

logger = None  # 训练专用日志记录器


def setup_train_logging(log_level: str = "INFO", log_format: Optional[str] = None, log_file: Optional[str] = None):
    """
    设置训练专用日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式字符串
        log_file: 日志文件路径
    """
    global logger
    
    if log_format is None:
        log_format = '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s'
    
    # 创建训练专用的日志记录器
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器（避免重复）
    logger.handlers.clear()
    
    # 使用标准的控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加控制台处理器
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        # 添加文件处理器到 train_logger
        logger.addHandler(file_handler)
    
    # 禁用向上传播到根logger，避免重复记录
    logger.propagate = False
    
    return logger


if logger is None:
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH, exist_ok=True)

    logger = setup_train_logging(log_file=LOG_FILE)