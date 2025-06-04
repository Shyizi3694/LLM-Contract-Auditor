# src/common/logging_setup.py
import logging
import os
import colorlog
# from .config import LOG_PATH as DEFAULT_LOG_PATH # 或者直接用 config.LOG_PATH
from . import config # 导入整个config模块，以便在函数内部访问最新的LOG_PATH

LOG_FILE_NAME = "contract_auditor.log" # 可以考虑也将其移到config.py中

def setup_logging(default_level=logging.INFO, log_path_override=None): # <--- 新增参数
    """配置全局日志记录系统，控制台输出带颜色"""
    
    # 决定实际使用的日志路径
    # 如果测试时传入了 log_path_override，则使用它；否则使用 config.py 中定义的 LOG_PATH
    effective_log_path = log_path_override if log_path_override is not None else config.LOG_PATH
    
    log_file_path = os.path.join(effective_log_path, LOG_FILE_NAME)
    
    if not os.path.exists(effective_log_path):
        try:
            os.makedirs(effective_log_path)
            # print(f"信息：日志目录 {effective_log_path} 已创建。") # 测试时不建议print
        except OSError as e:
            # print(f"错误：创建日志目录 {effective_log_path} 失败: {e}。")
            log_file_path = LOG_FILE_NAME # Fallback, or raise error
    
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level)

    # 清除已存在的处理器 (保持不变)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close() # 关闭处理器是个好习惯

    # 文件处理器 (使用 effective_log_path)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(default_level)
    root_logger.addHandler(file_handler)

    # 控制台处理器 (保持不变)
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(levelname)-8s]%(reset)s %(message)s (%(filename)s:%(lineno)d)",
        # ... (其他 colorlog 配置保持不变)
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'purple', # 您之前考虑的颜色
        },
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(default_level)
    root_logger.addHandler(console_handler)