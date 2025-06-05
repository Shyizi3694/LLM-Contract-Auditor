# src/knowledge_retrieval/retriever.py

import pandas as pd
import logging
import random # 用于随机选取专家案例，如果需要的话

logger = logging.getLogger(__name__)

class KnowledgeRetriever:
    def __init__(self, checklist_df: pd.DataFrame, expert_pairs_df: pd.DataFrame):
        """
        初始化知识检索器。

        参数:
            checklist_df (pd.DataFrame): 预处理后的风险审查清单 DataFrame。
                                         必需列: 'risk_feature' (已规范化), 'identified_items'.
            expert_pairs_df (pd.DataFrame): 预处理后的专家知识库 DataFrame。
                                           必需列: 'checkpoint' (已规范化), 'clause', 'review'.
        """
        if not isinstance(checklist_df, pd.DataFrame):
            raise TypeError("checklist_df 必须是一个 Pandas DataFrame。")
        if not isinstance(expert_pairs_df, pd.DataFrame):
            raise TypeError("expert_pairs_df 必须是一个 Pandas DataFrame。")

        self.checklist_df = checklist_df
        self.expert_pairs_df = expert_pairs_df
        
        # 验证必需的列是否存在，这有助于早期发现问题
        required_checklist_cols = ['risk_feature', 'identified_items']
        if not all(col in self.checklist_df.columns for col in required_checklist_cols):
            raise ValueError(f"checklist_df 缺少必需的列。需要: {required_checklist_cols}, 实际有: {list(self.checklist_df.columns)}")

        required_expert_cols = ['checkpoint', 'clause', 'review']
        if not all(col in self.expert_pairs_df.columns for col in required_expert_cols):
            raise ValueError(f"expert_pairs_df 缺少必需的列。需要: {required_expert_cols}, 实际有: {list(self.expert_pairs_df.columns)}")

        logger.info("KnowledgeRetriever 初始化完成。")

    def retrieve_checklist_info(self, contract_checkpoint: str) -> dict | None:
        """
        根据合同条款的 checkpoint 从 checklist_df 中检索相关的风险特征和表现形式。
        假设 contract_checkpoint 和 checklist_df['risk_feature'] 都已规范化（例如，小写、去空格）。

        参数:
            contract_checkpoint (str): 规范化后的合同条款检查点。

        返回:
            dict: 包含 'risk_feature' 和 'identified_items' 的字典，如果找到。
                  否则返回 None。
        """
        if not contract_checkpoint or not isinstance(contract_checkpoint, str):
            logger.warning("提供的 contract_checkpoint 无效 (空或非字符串)，无法检索清单信息。")
            return None

        try:
            # 假设 'risk_feature' 是 checklist_df 中与 contract_checkpoint 对应的列
            matched_rows = self.checklist_df[self.checklist_df['risk_feature'] == contract_checkpoint]

            if not matched_rows.empty:
                # 通常一个checkpoint对应一个risk_feature条目，取第一个
                # 如果可能有一对多，需要根据业务逻辑调整
                first_match = matched_rows.iloc[0]
                checklist_info = {
                    'risk_feature': first_match['risk_feature'],
                    'identified_items': first_match['identified_items']
                }
                logger.info(f"为 checkpoint '{contract_checkpoint}' 找到了匹配的风险清单条目。")
                return checklist_info
            else:
                logger.warning(f"未能在风险清单中找到与 checkpoint '{contract_checkpoint}' 匹配的条目。")
                return None
        except KeyError as e:
            logger.error(f"检索风险清单信息时发生KeyError: {e}。请检查checklist_df的列名是否正确。")
            return None
        except Exception as e:
            logger.error(f"检索风险清单信息时发生意外错误: {e}", exc_info=True)
            return None

    def retrieve_expert_examples(self, contract_checkpoint: str, num_examples: int = 2) -> list[dict]:
        """
        根据合同条款的 checkpoint 从 expert_pairs_df 中检索相关的专家案例。
        假设 contract_checkpoint 和 expert_pairs_df['checkpoint'] 都已规范化。

        参数:
            contract_checkpoint (str): 规范化后的合同条款检查点。
            num_examples (int): 希望检索的专家案例数量。

        返回:
            list[dict]: 包含专家案例的列表，每个案例是一个包含 'clause' 和 'review' 的字典。
                        如果找不到则返回空列表。
        """
        if not contract_checkpoint or not isinstance(contract_checkpoint, str):
            logger.warning("提供的 contract_checkpoint 无效 (空或非字符串)，无法检索专家案例。")
            return []
        if not isinstance(num_examples, int) or num_examples < 0:
            logger.warning(f"num_examples ({num_examples}) 无效，将使用默认值0（不检索任何案例）。")
            num_examples = 0
        
        if num_examples == 0:
            return []

        try:
            matched_examples = self.expert_pairs_df[self.expert_pairs_df['checkpoint'] == contract_checkpoint]

            if not matched_examples.empty:
                # 如果匹配到的案例少于或等于期望数量，则全部返回
                if len(matched_examples) <= num_examples:
                    selected_examples = matched_examples
                else:
                    # 如果匹配到的案例多于期望数量，随机选择或选择前几个
                    # 为了测试的确定性，我们先选择前 num_examples 个
                    selected_examples = matched_examples.head(num_examples)
                    # 或者随机选择:
                    # selected_examples = matched_examples.sample(n=num_examples, random_state=42) # random_state用于可复现

                result = selected_examples[['clause', 'review']].to_dict(orient='records')
                logger.info(f"为 checkpoint '{contract_checkpoint}' 找到了 {len(result)} 个专家案例 (期望最多 {num_examples} 个)。")
                return result
            else:
                logger.warning(f"未能在专家知识库中找到与 checkpoint '{contract_checkpoint}' 匹配的案例。")
                return []
        except KeyError as e:
            logger.error(f"检索专家案例时发生KeyError: {e}。请检查expert_pairs_df的列名是否正确。")
            return []
        except Exception as e:
            logger.error(f"检索专家案例时发生意外错误: {e}", exc_info=True)
            return []

    def retrieve_knowledge(self, contract_checkpoint: str, num_expert_examples: int = 2) -> dict:
        """
        主检索方法，整合风险清单信息和专家案例。

        参数:
            contract_checkpoint (str): 规范化后的合同条款检查点。
            num_expert_examples (int): 希望检索的专家案例数量。

        返回:
            dict: 包含 'checklist_info' 和 'expert_examples' 的字典。
        """
        logger.info(f"开始为 checkpoint '{contract_checkpoint}' 检索知识...")
        
        # 规范化输入的 checkpoint (尽管我们假设它已规范化，但多一层保障无妨或在此强调调用者责任)
        normalized_checkpoint = contract_checkpoint.lower().strip() if isinstance(contract_checkpoint, str) else ""

        checklist_info = self.retrieve_checklist_info(normalized_checkpoint)
        expert_examples = self.retrieve_expert_examples(normalized_checkpoint, num_examples=num_expert_examples)

        retrieved_data = {
            "contract_checkpoint": contract_checkpoint, # 返回原始传入的，或规范化的，取决于需求
            "checklist_info": checklist_info,
            "expert_examples": expert_examples
        }
        logger.info(f"知识检索完成，共找到清单条目: {'1' if checklist_info else '0'}, "
                    f"专家案例: {len(expert_examples)}。")
        return retrieved_data