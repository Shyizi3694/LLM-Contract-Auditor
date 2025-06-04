# 建议创建一个新文件 src/common/logging_setup.py
import logging
import os
# 假设 LOG_PATH 从 config.py 导入
# from src.common.config import LOG_PATH # 如果 LOG_PATH 在 config.py 中

# 如果您在 config.py 中已经定义了 LOG_PATH:
from .config import LOG_PATH # 使用相对导入，因为 logging_setup.py 和 config.py 都在 common 包下

LOG_FILE_NAME = "contract_auditor.log"

def setup_logging(default_level=logging.INFO):
    """配置全局日志记录系统"""
    # 确保日志目录存在
    if not os.path.exists(LOG_PATH):
        try:
            os.makedirs(LOG_PATH)
            print(f"信息：日志目录 {LOG_PATH} 已创建。")
        except OSError as e:
            print(f"错误：创建日志目录 {LOG_PATH} 失败: {e}。日志将尝试写入当前工作目录。")
            log_file_path = LOG_FILE_NAME # 回退方案
    else:
        log_file_path = os.path.join(LOG_PATH, LOG_FILE_NAME)

    # 定义日志格式
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S" # 时间格式
    )

    # 获取根logger，并设置级别
    # 或者，您可以为不同的logger（模块）设置不同的级别和处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level)
    
    # 清除可能存在的旧处理器，避免重复添加 (特别是在多次调用setup_logging时)
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # mode='a' for append
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # logging.info("日志系统初始化完成。") # 这条消息会通过新配置的logger发出


if __name__ == "__main__":
    logger = logging.getLogger(__name__) # 获取当前模块的logger，用于测试输出
    setup_logging() # 如果直接运行此模块，则初始化日志系统
    # 可以添加一些测试日志输出，以验证配置
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")