# src/prompting/prompt_builder.py
import logging
from typing import List, Dict, Optional

# 获取模块级 logger
logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self):
        """
        初始化 PromptBuilder。
        目前不需要特定配置，但可以预留扩展性。
        """
        logger.info("PromptBuilder 初始化完成。")

    def build_prompt(self,
                     contract_clause: str,
                     contract_checkpoint: str, # 用于在提示词中提供原始分类上下文
                     checklist_info: Optional[Dict[str, str]],
                     expert_examples: List[Dict[str, str]]) -> str:
        """
        构建将发送给大语言模型的完整英文提示词。

        参数:
            contract_clause (str): 需要审查的合同条款文本 (英文)。
            contract_checkpoint (str): 与该合同条款关联的原始检查点/分类 (英文, 规范化后的)。
            checklist_info (Optional[Dict[str, str]]): 从风险清单中检索到的信息。
                期望键: 'risk_feature', 'identified_items' (均为英文)。
            expert_examples (List[Dict[str, str]]): 从专家知识库中检索到的案例列表。
                每个字典期望键: 'clause', 'review' (均为英文)。

        返回:
            str: 构建完成的英文提示词字符串。
        """
        if not contract_clause:
            logger.warning("输入的合同条款为空，生成的提示词可能效果不佳。")
            # 即使条款为空，也继续构建提示词的其余部分，让模型知道这是一个空条款的情况

        prompt_parts = []

        # --- 1. 角色设定和主要任务描述 (英文) ---
        prompt_parts.append(
            "You are an expert legal assistant specializing in contract risk identification. "
            "Your primary task is to meticulously review the provided contract clause. "
            "Based on the associated risk checklist information and any expert review examples provided for similar clauses, "
            "you must determine if the contract clause presents any potential risks."
        )
        prompt_parts.append(
            "\nInstructions for your response:\n"
            "1. Assess the risk: Label the clause as 'A' if it has a discernible risk, or 'B' if it appears to have no risk based on the information provided.\n"
            "2. Provide Explanation: Briefly explain your reasoning for the assigned risk label. If a risk is identified (Label 'A'), clearly state what the risk is and why. If no risk is identified (Label 'B'), briefly state why it's considered safe or compliant.\n"
            "3. Output Format: Respond strictly with a single JSON object containing two fields: "
            "\"risk_label\" (string: \"A\" or \"B\") and \"explanation\" (string: your concise reasoning)."
        )
        prompt_parts.append("\n--- BEGIN CONTEXTUAL INFORMATION ---")

        # --- 2. 相关的风险清单信息 (如果可用) ---
        if checklist_info and isinstance(checklist_info, dict):
            prompt_parts.append("\n[Relevant Risk Checklist Information]")
            # 使用 get 方法以防万一键不存在，尽管 KnowledgeRetriever 应保证结构
            rf = checklist_info.get('risk_feature', 'N/A')
            items = checklist_info.get('identified_items', 'N/A')
            prompt_parts.append(f"Risk Category (Derived from Checkpoint): {rf}")
            prompt_parts.append(f"Common Manifestations or Points to Check for this Risk Category:\n{items}")
        else:
            prompt_parts.append("\n[Relevant Risk Checklist Information]")
            prompt_parts.append("No specific risk checklist information was retrieved or provided for this clause's category.")
        
        prompt_parts.append("---")

        # --- 3. 专家评审案例 (Few-Shot Learning, 如果可用) ---
        if expert_examples and isinstance(expert_examples, list) and len(expert_examples) > 0:
            prompt_parts.append("\n[Expert Review Examples for Similar Clause Categories]")
            for i, example in enumerate(expert_examples):
                if isinstance(example, dict) and 'clause' in example and 'review' in example:
                    prompt_parts.append(f"\nExample {i+1}:")
                    prompt_parts.append(f"Expert Example Clause:\n\"\"\"\n{example['clause']}\n\"\"\"")
                    prompt_parts.append(f"Expert Example Review:\n\"\"\"\n{example['review']}\n\"\"\"")
                else:
                    # 记录日志，但不在提示词中引入错误信息，避免干扰模型
                    logger.warning(f"跳过格式不正确的专家案例: {example}")
            prompt_parts.append("---")
        else:
            prompt_parts.append("\n[Expert Review Examples for Similar Clause Categories]")
            prompt_parts.append("No specific expert review examples were retrieved or provided for this clause's category.")
        
        prompt_parts.append("--- END CONTEXTUAL INFORMATION ---")

        # --- 4. 需要评审的合同条款原文 ---
        prompt_parts.append("\n[Contract Clause to Review]")
        prompt_parts.append(f"Original Checkpoint Category of the Clause: \"{contract_checkpoint}\"")
        prompt_parts.append("Clause Text requiring your assessment:")
        prompt_parts.append(f"\"\"\"\n{contract_clause}\n\"\"\"") # 使用三引号以支持多行条款

        # --- 5. 再次强调任务和输出格式 (可选，但有时有益) ---
        prompt_parts.append(
            "\nBased on all the information above (checklist, expert examples if any, and the clause text itself), "
            "provide your risk assessment solely in the specified JSON format."
        )

        full_prompt = "\n".join(prompt_parts)
        
        # 记录提示词的前一部分用于调试，避免记录整个可能很长的提示词
        log_prompt_snippet = full_prompt[:500].replace('\n', '\\n') # 替换换行符以便单行日志查看
        logger.debug(f"构建的提示词 (前500字符): {log_prompt_snippet}...")
        
        return full_prompt