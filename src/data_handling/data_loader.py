# src/data_handling/data_loader.py
import pandas as pd
import os
import logging
try:
    from src.common.config import RAW_DATA_PATH as DEFAULT_RAW_DATA_PATH # 导入默认路径
except ImportError:
    DEFAULT_RAW_DATA_PATH = "data/raw" # 回退

logger = logging.getLogger(__name__)

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
            full_path = self._construct_file_path(file_name)
            df = pd.read_csv(full_path)
            logger.info(f"成功加载数据 '{file_name}'。数据形状: {df.shape}")

            # --- 新增/修改的检查逻辑 ---
            if df.empty:
                # 这个警告会在文件只有表头，或内容为空导致pandas读取后df.empty为True时触发
                logger.warning(f"数据文件 '{file_name}' 读取后为空 (可能只有表头或无数据行)。")
                # 注意：即使文件读取后为空，我们仍然可以继续检查必需的列是否存在于表头中。
                # 如果您希望在这种情况下立即返回一个完全空的DataFrame，可以取消下一行的注释：
                # return pd.DataFrame(columns=required_columns if required_columns else None)

            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    logger.error(f"文件 '{file_name}' 缺少必需的列: {', '.join(missing_cols)}。实际列: {list(df.columns)}")
                    # 如果缺少关键列，您可能希望返回一个空的DataFrame或抛出错误
                    # 例如: return pd.DataFrame()
            return df # 返回读取到的df (即使它可能是空的)

        except FileNotFoundError:
            # 此处错误已在 _construct_file_path 中记录 (或应在那里记录)
            return pd.DataFrame() 
        except pd.errors.EmptyDataError: # 这个分支主要处理完全空的文件（例如0字节）
            logger.warning(f"数据文件 '{file_name}' 完全为空 (EmptyDataError)。")
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