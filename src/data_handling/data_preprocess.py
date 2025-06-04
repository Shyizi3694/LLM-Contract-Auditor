# src/data_handling/data_preprocess.py

import pandas as pd
import os
import logging

from src.common import config as app_config
# 尝试从公共配置导入路径，如果失败则使用回退（主要用于直接运行此脚本时的便利性）
# try:
#    from src.common.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
# except ImportError:
#     logger_temp = logging.getLogger(__name__) # 临时logger
#     logger_temp.warning("无法从 src.common.config 导入路径配置。将使用基于此文件位置的相对回退路径。")
#     _actuelle_file_dir = os.path.dirname(os.path.abspath(__file__)) # src/data_handling
#     _src_dir = os.path.dirname(_actuelle_file_dir) # src/
#     _project_root = os.path.dirname(_src_dir) # project root
#     RAW_DATA_PATH = os.path.join(_project_root, "data", "raw")
#     PROCESSED_DATA_PATH = os.path.join(_project_root, "data", "processed")

logger = logging.getLogger(__name__) # 获取模块级 logger

def _clean_and_normalize_df(df: pd.DataFrame, 
                           text_cols_to_strip_fillna: list[str] | None = None, 
                           key_cols_to_lower_strip_fillna: list[str] | None = None) -> pd.DataFrame:
    """
    对DataFrame应用通用的清洗和规范化操作。
    - 对指定的文本列：先用空字符串填充NaN，然后转为字符串，最后去除首尾空格。
    - 对指定的键列：先用空字符串填充NaN，然后转为字符串，转为小写，最后去除首尾空格。
    """
    processed_df = df.copy()

    if text_cols_to_strip_fillna:
        for col in text_cols_to_strip_fillna:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna('').astype(str).str.strip()
            else:
                logger.warning(f"文本列 '{col}' 在DataFrame中未找到。跳过处理。")
    
    if key_cols_to_lower_strip_fillna:
        for col in key_cols_to_lower_strip_fillna:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna('').astype(str).str.lower().str.strip()
            else:
                logger.warning(f"键列 '{col}' 在DataFrame中未找到。跳过处理。")
    return processed_df

def _preprocess_single_csv(input_base_path: str, 
                           output_base_path: str, 
                           file_name: str, 
                           text_cols: list[str] | None = None, 
                           key_cols: list[str] | None = None,
                           specific_column_cleaning: dict | None = None):
    """
    预处理单个CSV文件的通用函数。
    specific_column_cleaning: 一个字典，键为列名，值为一个函数，用于对该列进行特定清洗。
    """
    input_file_path = os.path.join(input_base_path, file_name)
    output_file_path = os.path.join(output_base_path, f"processed_{file_name}")

    try:
        if not os.path.exists(input_file_path):
            logger.error(f"原始数据文件未找到: {input_file_path}。跳过此文件的预处理。")
            return

        df = pd.read_csv(input_file_path)
        logger.info(f"开始预处理文件: {file_name}。原始形状: {df.shape}")
        
        processed_df = _clean_and_normalize_df(
            df,
            text_cols_to_strip_fillna=text_cols,
            key_cols_to_lower_strip_fillna=key_cols
        )

        if specific_column_cleaning:
            for col, clean_func in specific_column_cleaning.items():
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].apply(clean_func)
                    logger.info(f"对文件 '{file_name}' 的 '{col}' 列应用了特定的清洗函数。")
                else:
                    logger.warning(f"尝试对文件 '{file_name}' 的 '{col}' 列应用特定清洗，但该列不存在。")
        
        processed_df.to_csv(output_file_path, index=False)
        logger.info(f"已保存预处理后的数据到: {output_file_path}")

    except pd.errors.EmptyDataError:
        logger.warning(f"原始数据文件 '{input_file_path}' 为空。已创建空的预处理文件：{output_file_path}")
        pd.DataFrame().to_csv(output_file_path, index=False) # 创建一个空的processed文件
    except Exception as e:
        logger.error(f"预处理文件 '{file_name}' 时发生错误: {e}", exc_info=True)


def preprocess_all_data():
    """
    加载所有原始CSV文件，进行预处理，并将它们保存到 PROCESSED_DATA_PATH 目录。
    """
    # --- 通过 app_config 访问路径 ---
    current_raw_path = app_config.RAW_DATA_PATH
    current_processed_path = app_config.PROCESSED_DATA_PATH

    logger.info(f"开始数据预处理流程。原始数据来源: {current_raw_path}, 处理后数据目标: {current_processed_path}")

    if current_processed_path is None: # 检查通过 app_config 获取的值
        logger.critical("PROCESSED_DATA_PATH 未配置或为None，无法进行预处理。")
        return

    if not os.path.exists(current_processed_path): # 使用 current_processed_path
        try:
            os.makedirs(current_processed_path) # 使用 current_processed_path
            logger.info(f"已创建处理后数据目录: {current_processed_path}")
        except OSError as e:
            logger.error(f"无法创建处理后数据目录 {current_processed_path}: {e}", exc_info=True)
            return # 如果无法创建目标目录，则停止

    # 定义每个文件的预处理规则 (保持不变)
    files_to_preprocess = {
        "contract.csv": {"text_cols": ['clause'], "key_cols": ['checkpoint']},
        "checklist.csv": {"text_cols": ['Identified_items'],"key_cols": ['Risk_feature']},
        "clause_review_pair.csv": {"text_cols": ['clause', 'review'], "key_cols": ['checkpoint']},
        "GT_contract_risk.csv": {
            "text_cols": ['clause'], 
            "key_cols": ['checkpoint'],
            "specific_cleaning": {'gt': lambda x: str(x).strip().upper() if pd.notna(x) else ''}
        }
    }

    for file_name, rules in files_to_preprocess.items():
        _preprocess_single_csv(
            current_raw_path,         # <--- 传递通过 app_config 获取的路径
            current_processed_path,   # <--- 传递通过 app_config 获取的路径
            file_name,
            text_cols=rules.get("text_cols"),
            key_cols=rules.get("key_cols"),
            specific_column_cleaning=rules.get("specific_cleaning")
        )

    logger.info("所有数据文件预处理完成。")

if __name__ == '__main__':
    # 直接运行此脚本时的基本日志配置 (主要用于开发测试)
    if not logging.getLogger().root.hasHandlers(): # 检查根logger是否已有处理器
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    logger.info("直接运行 data_preprocess.py 进行预处理...")
    if not os.path.exists(app_config.RAW_DATA_PATH):
        logger.error(f"错误: 原始数据路径 '{app_config.RAW_DATA_PATH}' 不存在。请检查配置或文件路径。")
    elif not os.path.exists(app_config.PROCESSED_DATA_PATH) or os.access(os.path.dirname(app_config.PROCESSED_DATA_PATH), os.W_OK):
        # 如果PROCESSED_DATA_PATH不存在，或者其父目录可写（意味着makedirs应该能成功）
        preprocess_all_data()
    else:
        logger.error(f"错误: 处理后数据路径 '{app_config.PROCESSED_DATA_PATH}' 存在问题或其父目录不可写。")