# src/common/config.py

import os
from dotenv import load_dotenv
import sys

# (load_app_config 函数定义如前，它会加载 .env 文件)
def load_app_config():
    """
    从项目的根目录加载 .env 文件中的环境变量。
    此函数假定 config.py 位于 src/common/ 目录下。
    """
    current_file_path = os.path.abspath(__file__)
    common_dir_path = os.path.dirname(current_file_path)
    src_dir_path = os.path.dirname(common_dir_path)
    project_root_path_for_dotenv = os.path.dirname(src_dir_path) # Renamed to avoid conflict
    
    dotenv_path = os.path.join(project_root_path_for_dotenv, '.env')

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        print(f"警告：在计算的项目根目录 {dotenv_path} 未找到 .env 文件。 "
              "将尝试从默认路径加载或依赖系统环境变量。")
        load_dotenv()

load_app_config() # 确保在定义路径前加载环境变量，以防未来路径也受环境变量影响

# --- 项目核心路径定义 ---
# 当前 config.py 文件的绝对路径
_CURRENT_CONFIG_FILE_PATH = os.path.abspath(__file__) # 使用 _ 前缀表示内部使用

# src/common/ 目录的路径
_COMMON_DIR_PATH = os.path.dirname(_CURRENT_CONFIG_FILE_PATH)

# src/ 目录的路径
SRC_DIR_PATH = os.path.dirname(_COMMON_DIR_PATH)

# 项目根目录的路径 (src/ 的上一级)
PROJECT_ROOT_PATH = os.path.dirname(SRC_DIR_PATH)

# --- 数据相关路径 ---
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")

# --- 结果输出相关路径 ---
RESULTS_PATH = os.path.join(PROJECT_ROOT_PATH, "results")
METRICS_PATH = os.path.join(RESULTS_PATH, "metrics")
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")
# 根据您的文件树，如果您确实需要使用 trained_models 目录：
# TRAINED_MODELS_PATH = os.path.join(RESULTS_PATH, "trained_models")

# --- (可选) 日志文件路径 ---
# 建议在项目根目录创建一个 logs 文件夹来存放日志
LOG_PATH = os.path.join(PROJECT_ROOT_PATH, "logs")
# 您可能需要手动创建 logs 文件夹，或者在首次写入日志时通过代码创建
# if not os.path.exists(LOG_PATH):
#     os.makedirs(LOG_PATH)

# --- (可选) 其他常用路径 ---
# 例如，如果您有专门的测试数据路径或utils路径需要引用
TESTS_PATH = os.path.join(PROJECT_ROOT_PATH, "tests")
UTILS_PATH = os.path.join(PROJECT_ROOT_PATH, "utils") # utils 通常通过python import机制访问


# (get_env_variable 函数定义如前)
def get_env_variable(variable_name: str, is_mandatory: bool = True, default_value: str = None) -> str | None:
    value = os.getenv(variable_name)
    if value is None:
        if is_mandatory:
            error_message = (
                f"错误：关键环境变量 '{variable_name}' 未找到。\n"
                # ... (错误信息内容同前)
            )
            raise ValueError(error_message)
        else:
            return default_value
    return value

# --- API Key 和 Model Name 配置 (如前) ---
API_KEY = None
try:
    API_KEY = get_env_variable("GEMINI_API_KEY", is_mandatory=True)
except ValueError as e:
    print(f"严重配置错误 (API Key): {e}")
    # sys.exit("API Key 未配置，程序无法继续。") 

DEFAULT_GEMINI_MODEL_NAME = "gemini-2.0-flash" 
MODEL_NAME = get_env_variable("GEMINI_MODEL_NAME", is_mandatory=False, default_value=DEFAULT_GEMINI_MODEL_NAME)


# --- 确保定义的路径文件夹存在 (可选，按需添加) ---
# 有时为了方便，可以在配置加载时就确保某些输出目录存在
for path_to_create in [PROCESSED_DATA_PATH, METRICS_PATH, PLOTS_PATH, LOG_PATH]:
    if not os.path.exists(path_to_create):
        try:
            os.makedirs(path_to_create)
            print(f"信息：已创建目录 {path_to_create}")
        except OSError as e:
            print(f"错误：创建目录 {path_to_create} 失败: {e}")

# # --- 调试打印定义的路径 (可选) ---
# print(f"项目根目录: {PROJECT_ROOT_PATH}")
# print(f"原始数据目录: {RAW_DATA_PATH}")
# print(f"指标输出目录: {METRICS_PATH}")
# print(f"日志目录: {LOG_PATH}")