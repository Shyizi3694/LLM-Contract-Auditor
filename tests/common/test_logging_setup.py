from src.common.logging_setup import setup_logging  # 根据实际路径调整导
# tests/common/test_logging_setup.py
import os
import logging
import pytest # pytest 库本身，以及它提供的 fixture
from pathlib import Path # 用于更方便地处理路径

# 从您的项目中导入被测试的函数和必要的常量
from src.common.logging_setup import setup_logging, LOG_FILE_NAME
# 注意：LOG_FILE_NAME 如果定义在 logging_setup.py 中，可以直接导入
# 如果 LOG_FILE_NAME 也定义在 config.py 中，则 from src.common.config import LOG_FILE_NAME





# 清理函数，确保每次测试都在干净的环境中运行（或者使用pytest的fixture机制）
def cleanup_log_handlers():
    """移除并关闭根logger上的所有处理器"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close() # 关键：关闭文件句柄等资源
        root_logger.removeHandler(handler)

def test_setup_logging_creates_directory_and_file(tmp_path: Path):
    """
    测试 setup_logging 是否在指定的临时路径下创建日志目录和文件。
    tmp_path 是 pytest 提供的一个fixture，会自动创建一个临时目录并在测试结束后清理。
    """
    cleanup_log_handlers() # 确保之前的处理器被清理

    test_specific_log_dir = tmp_path / "test_project_logs" # 在临时目录下创建一个子目录用作日志路径

    # 调用被测试的函数，并传入临时日志路径
    setup_logging(default_level=logging.DEBUG, log_path_override=str(test_specific_log_dir))

    # 1. 验证日志目录是否被创建
    assert test_specific_log_dir.exists(), "日志目录应该被 setup_logging 创建"
    assert test_specific_log_dir.is_dir(), "指定的日志路径应该是一个目录"

    # 2. 验证日志文件是否在指定目录下被创建
    expected_log_file = test_specific_log_dir / LOG_FILE_NAME
    assert expected_log_file.exists(), "日志文件应该在指定的日志目录中被创建"
    assert expected_log_file.is_file(), "创建的日志文件应该是一个文件"

def test_setup_logging_logs_messages_to_file(tmp_path: Path, caplog):
    """
    测试 setup_logging 配置后，日志消息是否能成功写入文件。
    caplog 是 pytest 提供的一个fixture，用于捕获日志输出。
    """
    cleanup_log_handlers()

    test_specific_log_dir = tmp_path / "test_project_logs_content"
    
    # 使用 DEBUG 级别，以便我们可以记录和检查 DEBUG 消息
    setup_logging(default_level=logging.DEBUG, log_path_override=str(test_specific_log_dir))
    
    expected_log_file = test_specific_log_dir / LOG_FILE_NAME

    # 获取一个logger实例并发送一条测试消息
    test_logger_name = "my_test_app.file_test"
    logger = logging.getLogger(test_logger_name)
    
    test_message_info = "This is an INFO message for file logging."
    test_message_debug = "This is a DEBUG message for file logging."
    
    logger.info(test_message_info)
    logger.debug(test_message_debug)

    # 强制刷新所有处理器，确保内容写入文件
    # (通常在处理器关闭时会自动刷新，但显式调用更保险，尤其是在测试结束前)
    for handler in logging.getLogger().handlers:
        handler.flush()

    # 验证文件内容
    assert expected_log_file.exists(), "日志文件应已创建"
    log_content = expected_log_file.read_text(encoding="utf-8")
    
    assert test_message_info in log_content, "INFO 消息应该在日志文件中"
    assert test_message_debug in log_content, "DEBUG 消息应该在日志文件中 (因为级别设为DEBUG)"
    assert test_logger_name in log_content, "Logger的名称应该在日志内容中"

    # pytest的caplog fixture可以捕获发送到logging系统的日志，主要用于控制台或其他非文件处理器
    # 这里我们主要测试文件写入，但也可以用caplog检查是否有消息被logger处理
    # logger.info("Another message for caplog.")
    # assert "Another message for caplog." in caplog.text
    # assert caplog.records[N].levelname == "INFO" # 检查特定记录的级别等
