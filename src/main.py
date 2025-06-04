
# src/main.py
import logging
import os
import sys

# --- 尝试以更健壮的方式处理模块导入路径 ---
# 获取当前文件 (main.py) 的目录，即 src/
_main_py_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (src/ 的上一级)
_project_root_dir = os.path.dirname(_main_py_dir)

# 将项目根目录和src目录添加到sys.path，确保模块能被找到
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)
if _main_py_dir not in sys.path: # src 目录
    sys.path.insert(0, _main_py_dir)

try:
    from common.logging_setup import setup_logging
    # from common.config import API_KEY, MODEL_NAME # 暂时不用，但可以保留导入
    from data_handling.data_preprocess import preprocess_all_data
except ImportError as e:
    print(f"发生导入错误，请确保您的PYTHONPATH配置正确，或者您是从项目根目录运行此脚本。错误: {e}")
    # 如果在IDE中运行，确保IDE的项目根目录设置正确。
    # 也可以考虑使用更复杂的相对导入或绝对导入（如果项目作为包安装）。
    sys.exit(1)


# 获取模块级 logger，在 setup_logging 之后获取，以确保配置生效
# logger = logging.getLogger(__name__) # 这需要在 setup_logging 之后

def run_pipeline():
    """
    主函数，运行合同审查流程。
    """
    # 日志应该在 setup_logging 之后获取和使用
    main_logger = logging.getLogger(__name__) # 在函数内部获取logger

    main_logger.info("=================================================")
    main_logger.info("启动 LLM 合同风险审查流程")
    main_logger.info("=================================================")

    # --- 阶段 1 (隐式): 配置和日志已由 setup_logging 处理 ---

    # --- 阶段 2 (部分): 数据预处理 ---
    main_logger.info("--- 开始阶段: 数据预处理 ---")
    try:
        preprocess_all_data() # 调用预处理函数
        main_logger.info("数据预处理阶段成功完成。")
    except Exception as e:
        main_logger.error(f"数据预处理阶段发生错误: {e}", exc_info=True)
        main_logger.critical("因数据预处理错误，流程中止。")
        return # 如果预处理失败，则停止

    # --- 后续阶段占位符 ---
    main_logger.info("--- 后续阶段 (数据加载, 知识检索等) 待实现 ---")
    
    # 示例：接下来应该是加载处理后的数据
    # from data_handling.data_loader import DataLoader
    # main_logger.info("尝试加载处理后的数据...")
    # try:
    #     # DataLoader 需要能够从 PROCESSED_DATA_PATH 加载
    #     # (DataLoader 的 __init__ 可能需要修改以接受 processed_path)
    #     # 或者创建一个新的 ProcessedDataLoader 类
    #     # from common.config import PROCESSED_DATA_PATH
    #     # loader = DataLoader(raw_data_path_override=PROCESSED_DATA_PATH) # 假设可以这样用
    #     # contracts = loader.get_contracts_df(file_name="processed_contract.csv") # 假设可以指定文件名
    #     # if contracts is not None and not contracts.empty:
    #     #     main_logger.info(f"成功加载处理后的合同数据: {len(contracts)} 条。")
    #     # else:
    #     #     main_logger.warning("未能加载处理后的合同数据或文件为空。")
    # except Exception as e:
    #     main_logger.error(f"加载处理后数据时出错: {e}", exc_info=True)


    main_logger.info("=================================================")
    main_logger.info("LLM 合同风险审查流程结束。")
    main_logger.info("=================================================")


if __name__ == '__main__':
    # 1. 首先配置日志系统
    try:
        # 假设日志级别等配置可以在 setup_logging 中调整或通过config传入
        setup_logging(default_level=logging.INFO) 
    except NameError: # 如果 setup_logging 因导入问题未定义
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.critical("自定义日志系统未能初始化！将使用基础日志配置。请检查导入路径。")
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"初始化自定义日志系统时发生错误: {e}。将使用基础日志配置。", exc_info=True)
        logging.warning("请检查 LOG_PATH 配置和目录权限。")

    # 2. 运行主流程
    run_pipeline()