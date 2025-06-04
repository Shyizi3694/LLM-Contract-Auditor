# src/data_handling/data_loader.py
import pandas as pd
import os
import logging
from src.common import logging_setup
try:
    from src.common.config import RAW_DATA_PATH as DEFAULT_RAW_DATA_PATH # 导入默认路径
except ImportError:
    DEFAULT_RAW_DATA_PATH = "data/raw" # 回退

logger = logging.getLogger(__name__)
logging_setup()

class DataLoader:
    def __init__(self, raw_data_path_override: str | None = None): # <--- 新增参数
        """
        初始化 DataLoader。
        可以传入 raw_data_path_override 来覆盖默认的原始数据路径，主要用于测试。
        """
        self.current_raw_data_path = raw_data_path_override if raw_data_path_override is not None else DEFAULT_RAW_DATA_PATH
        if self.current_raw_data_path is None:
             error_msg = "原始数据路径 (RAW_DATA_PATH) 未配置或导入失败。"
             logger.critical(error_msg)
             raise ValueError(error_msg)
        logger.info(f"DataLoader 使用的原始数据路径: {self.current_raw_data_path}")

        self.contracts_df: pd.DataFrame | None = None
        self.checklist_df: pd.DataFrame | None = None
        self.expert_pairs_df: pd.DataFrame | None = None
        self.ground_truth_df: pd.DataFrame | None = None
        
        self._load_all_data()

    def _construct_file_path(self, file_name: str) -> str:
        """
        构建文件的完整路径并检查文件是否存在 (使用 self.current_raw_data_path)。
        """
        path = os.path.join(self.current_raw_data_path, file_name) # <--- 使用实例变量
        if not os.path.exists(path):
            logger.error(f"数据文件未找到: {path}")
            raise FileNotFoundError(f"数据文件未找到: {path}。")
        return path

    # _load_csv, load_contracts, load_checklist, 等方法保持不变，
    # 它们内部会通过调用 self._construct_file_path() 来使用 self.current_raw_data_path

    def _load_csv(self, file_name: str, required_columns: list[str] | None = None) -> pd.DataFrame:
        """
        通用的CSV文件加载函数，包含错误处理和列验证。
        """
        try:
            full_path = self._construct_file_path(file_name) # Will use self.current_raw_data_path
            df = pd.read_csv(full_path)
            logger.info(f"成功加载数据 '{file_name}'。数据形状: {df.shape}")

            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    logger.error(f"文件 '{file_name}' 缺少必需的列: {', '.join(missing_cols)}。实际列: {list(df.columns)}")
            return df
        except FileNotFoundError:
            return pd.DataFrame() 
        except pd.errors.EmptyDataError:
            logger.warning(f"数据文件 '{file_name}' 为空。")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"加载数据文件 '{file_name}' 时发生未预期错误: {e}", exc_info=True)
            return pd.DataFrame()

    # load_contracts, load_checklist, load_expert_knowledge, load_ground_truth
    # _load_all_data, get_contracts_df 等方法保持不变...
    # (此处省略这些方法的代码，假设它们已在您的文件中)
    def load_contracts(self, file_name="contract.csv") -> pd.DataFrame:
        """加载合同条款数据 (contract.csv)"""
        return self._load_csv(file_name, required_columns=['checkpoint', 'clause'])

    def load_checklist(self, file_name="checklist.csv") -> pd.DataFrame:
        """加载风险审查清单数据 (checklist.csv)"""
        return self._load_csv(file_name, required_columns=['Risk_feature', 'Identified_items'])

    def load_expert_knowledge(self, file_name="clause_review_pair.csv") -> pd.DataFrame:
        """加载专家知识库数据 (clause_review_pair.csv)"""
        return self._load_csv(file_name, required_columns=['checkpoint', 'clause', 'review'])

    def load_ground_truth(self, file_name="GT_contract_risk.csv") -> pd.DataFrame:
        """加载Ground Truth风险数据 (GT_contract_risk.csv)"""
        return self._load_csv(file_name, required_columns=['checkpoint', 'clause', 'gt'])

    def _load_all_data(self):
        """
        加载所有数据集到类的实例变量中。
        """
        logger.info("开始加载所有数据集...")
        self.contracts_df = self.load_contracts()
        self.checklist_df = self.load_checklist()
        self.expert_pairs_df = self.load_expert_knowledge()
        self.ground_truth_df = self.load_ground_truth()
        
        if self.contracts_df.empty: # 示例检查
            logger.warning("合同数据 (contracts_df) 未能成功加载或文件为空。")
        
        logger.info("所有数据集加载尝试完毕。")

    def get_contracts_df(self) -> pd.DataFrame | None:
        return self.contracts_df

    def get_checklist_df(self) -> pd.DataFrame | None:
        return self.checklist_df

    def get_expert_pairs_df(self) -> pd.DataFrame | None:
        return self.expert_pairs_df

    def get_ground_truth_df(self) -> pd.DataFrame | None:
        return self.ground_truth_df
    def _load_csv(self, file_name: str, required_columns: list[str] | None = None) -> pd.DataFrame:
        """
        通用的CSV文件加载函数，包含错误处理和列验证。
        """
        try:
            full_path = self._construct_file_path(file_name)
            df = pd.read_csv(full_path)
            logger.info(f"成功加载数据 '{file_name}'。数据形状: {df.shape}")

            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    logger.error(f"文件 '{file_name}' 缺少必需的列: {', '.join(missing_cols)}。实际列: {list(df.columns)}")
                    # 根据需求，可以选择返回空的DataFrame，部分加载，或抛出错误
                    # return pd.DataFrame() # 或者 raise ValueError(...)
            return df
        except FileNotFoundError:
            # 此错误已在 _construct_file_path 中记录，这里返回空DataFrame允许程序继续
            return pd.DataFrame() 
        except pd.errors.EmptyDataError:
            logger.warning(f"数据文件 '{file_name}' 为空。")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"加载数据文件 '{file_name}' 时发生未预期错误: {e}", exc_info=True)
            return pd.DataFrame() # 返回空DataFrame以允许程序处理此情况

    def load_contracts(self, file_name="contract.csv") -> pd.DataFrame:
        """加载合同条款数据 (contract.csv)"""
        return self._load_csv(file_name, required_columns=['checkpoint', 'clause'])

    def load_checklist(self, file_name="checklist.csv") -> pd.DataFrame:
        """加载风险审查清单数据 (checklist.csv)"""
        return self._load_csv(file_name, required_columns=['Risk_feature', 'Identified_items'])

    def load_expert_knowledge(self, file_name="clause_review_pair.csv") -> pd.DataFrame:
        """加载专家知识库数据 (clause_review_pair.csv)"""
        return self._load_csv(file_name, required_columns=['checkpoint', 'clause', 'review'])

    def load_ground_truth(self, file_name="GT_contract_risk.csv") -> pd.DataFrame:
        """加载Ground Truth风险数据 (GT_contract_risk.csv)"""
        return self._load_csv(file_name, required_columns=['checkpoint', 'clause', 'gt'])

    def _load_all_data(self):
        """
        加载所有数据集到类的实例变量中。
        """
        logger.info("开始加载所有数据集...")
        self.contracts_df = self.load_contracts()
        self.checklist_df = self.load_checklist()
        self.expert_pairs_df = self.load_expert_knowledge()
        self.ground_truth_df = self.load_ground_truth()
        
        # 检查是否有任何关键的DataFrame加载失败（变为空的DataFrame）
        if self.contracts_df.empty:
            logger.warning("合同数据 (contracts_df) 未能成功加载或文件为空。")
        # 可以为其他df添加类似的检查
        
        logger.info("所有数据集加载尝试完毕。")

    # 提供访问加载数据的getter方法 (可选, 但推荐封装)
    def get_contracts_df(self) -> pd.DataFrame | None:
        return self.contracts_df

    def get_checklist_df(self) -> pd.DataFrame | None:
        return self.checklist_df

    def get_expert_pairs_df(self) -> pd.DataFrame | None:
        return self.expert_pairs_df

    def get_ground_truth_df(self) -> pd.DataFrame | None:
        return self.ground_truth_df

# # --- 用于直接运行此脚本进行简单测试的部分 ---
# if __name__ == '__main__':
#     # 为了能看到logger的输出，需要先配置logging
#     # 假设 setup_logging 在项目的其他地方定义并可以导入
#     # 或者为了简单测试，在这里进行非常基础的配置
#     if not logging.getLogger().hasHandlers(): # 检查是否已有处理器
#         logging.basicConfig(
#             level=logging.INFO, 
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[logging.StreamHandler()] # 输出到控制台
#         )
    
#     logger.info("直接测试 DataLoader 模块...")
    
#     # 重要的提醒:
#     # 当直接运行此文件 (python src/data_handling/data_loader.py) 时，
#     # Python的模块导入机制可能与从项目根目录通过 main.py 运行时的行为不同。
#     # 特别是 `from src.common.config import RAW_DATA_PATH` 这类导入。
#     # 如果遇到导入错误，通常是因为 PYTHONPATH 没有包含项目的根目录。
#     # 对于模块的单元测试，强烈推荐使用 pytest，它能更好地处理路径和测试环境。

#     # 临时的解决方案，以便能直接运行看到效果 (不推荐用于最终代码):
#     # 确保 RAW_DATA_PATH 能被正确访问。
#     # 如果 RAW_DATA_PATH 因为导入问题为 None，下面的代码会出错。
#     # 在这种情况下，您可能需要手动设置 RAW_DATA_PATH 的值用于测试。
#     # 例如 (仅用于此 __main__ 块的临时测试):
#     # if 'RAW_DATA_PATH' not in globals() or RAW_DATA_PATH is None:
#     #     print("警告: RAW_DATA_PATH 未从config加载，将使用临时路径进行直接测试。")
#     #     # D:/PROJECT/Data-Mining/LLM-Contract-Auditor/data/raw/
#     #     RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
#     #     print(f"临时 RAW_DATA_PATH: {RAW_DATA_PATH}")


#     try:
#         loader = DataLoader()
        
#         print("\n--- 合同数据 (contracts_df) ---")
#         if loader.get_contracts_df() is not None and not loader.get_contracts_df().empty:
#             print(loader.get_contracts_df().head())
#             print(f"形状: {loader.get_contracts_df().shape}")
#         else:
#             print("未能加载合同数据或文件为空。")

#         print("\n--- 风险清单 (checklist_df) ---")
#         if loader.get_checklist_df() is not None and not loader.get_checklist_df().empty:
#             print(loader.get_checklist_df().head())
#             print(f"形状: {loader.get_checklist_df().shape}")
#         else:
#             print("未能加载风险清单数据或文件为空。")

#         print("\n--- 专家知识 (expert_pairs_df) ---")
#         if loader.get_expert_pairs_df() is not None and not loader.get_expert_pairs_df().empty:
#             print(loader.get_expert_pairs_df().head())
#             print(f"形状: {loader.get_expert_pairs_df().shape}")
#         else:
#             print("未能加载专家知识数据或文件为空。")

#         print("\n--- Ground Truth (ground_truth_df) ---")
#         if loader.get_ground_truth_df() is not None and not loader.get_ground_truth_df().empty:
#             print(loader.get_ground_truth_df().head())
#             print(f"形状: {loader.get_ground_truth_df().shape}")
#         else:
#             print("未能加载Ground Truth数据或文件为空。")

#     except Exception as e:
#         logger.error(f"在 DataLoader 直接测试过程中发生错误: {e}", exc_info=True)