# src/response_processing/parser.py

import logging
from typing import Optional, Dict, Any

# 获取模块级 logger
logger = logging.getLogger(__name__)

class ResponseParser:
    def __init__(self):
        """
        初始化响应解析器。
        """
        logger.info("ResponseParser 初始化完成。")

    def parse(self, llm_response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        解析、验证并清洗来自大语言模型的响应字典。

        参数:
            llm_response (Optional[Dict[str, Any]]): 
                来自 GeminiService 的原始响应字典，可能为 None。
                期望的结构: {'risk_label': 'A'|'B', 'explanation': '...'}

        返回:
            Optional[Dict[str, Any]]: 
                如果输入有效，返回一个包含 'risk_label' 和 'explanation' 的干净字典。
                如果输入无效或不符合规范，返回 None。
        """
        # 1. 检查输入是否为 None
        if llm_response is None:
            logger.info("输入的LLM响应为 None，解析中止。")
            return None

        # 2. 检查输入是否为字典类型
        if not isinstance(llm_response, dict):
            logger.warning(f"输入的LLM响应不是一个字典，而是一个 {type(llm_response)}。解析失败。")
            return None
        
        logger.debug(f"开始解析原始响应: {llm_response}")

        # 3. 检查必需的键是否存在
        if 'risk_label' not in llm_response:
            logger.warning(f"LLM响应中缺少必需的键: 'risk_label'。响应内容: {llm_response}")
            return None
        
        if 'explanation' not in llm_response:
            logger.warning(f"LLM响应中缺少必需的键: 'explanation'。响应内容: {llm_response}")
            return None
        
        # 4. 验证和规范化 'risk_label'
        risk_label = llm_response.get('risk_label')
        if not isinstance(risk_label, str):
            logger.warning(f"LLM响应中的 'risk_label' 不是字符串类型，而是 {type(risk_label)}。值: {risk_label}")
            return None
        
        normalized_label = risk_label.strip().upper()
        if normalized_label not in ['A', 'B']:
            logger.warning(f"LLM响应中的 'risk_label' 值无效: '{normalized_label}'。期望值为 'A' 或 'B'。")
            return None
            
        # 5. 验证和清洗 'explanation'
        explanation = llm_response.get('explanation')
        if not isinstance(explanation, str):
            logger.debug(f"LLM响应中的 'explanation' 不是字符串类型 (类型为 {type(explanation)})，将尝试转换为字符串。")
            cleaned_explanation = str(explanation).strip()
        else:
            cleaned_explanation = explanation.strip()

        # 6. 所有检查通过，构建并返回干净的结果字典
        parsed_result = {
            'risk_label': normalized_label,
            'explanation': cleaned_explanation
        }
        
        logger.info(f"成功解析LLM响应。风险标签: {normalized_label}")
        return parsed_result