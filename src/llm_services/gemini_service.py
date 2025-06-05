# src/llm_services/gemini_service.py

import logging
import json
from typing import Optional, Dict, Any

# 导入 google-generativeai 库，通常简写为 genai
import google.generativeai as genai
# 导入可能出现的特定API异常类型，用于更精确的错误处理
from google.api_core import exceptions as google_api_exceptions
from google.generativeai.types import generation_types

# 从公共配置中导入API密钥和模型名称
try:
    from src.common.config import API_KEY, MODEL_NAME
except ImportError:
    # 这是一个回退，理想情况下，运行环境应能正确找到src.common.config
    logger_temp = logging.getLogger(__name__)
    logger_temp.critical("无法从 src.common.config 导入配置。请检查PYTHONPATH或项目结构。")
    # 为了让代码能继续（尽管会失败），设置默认值
    API_KEY = None
    MODEL_NAME = "gemini-2.0-flash" # 一个默认值

# 获取模块级 logger
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        """
        初始化Gemini服务。
        - 检查并配置API密钥。
        - 初始化GenerativeModel实例。
        """
        logger.info("正在初始化 GeminiService...")
        if not API_KEY:
            # 如果API_KEY在config.py中加载失败，这里会是None
            raise ValueError("Gemini API Key 未在配置中找到或加载失败。")

        try:
            genai.configure(api_key=API_KEY)
            self.model_name = MODEL_NAME
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"GeminiService 初始化成功，使用模型: {self.model_name}")
        except Exception as e:
            logger.critical(f"配置 google-generativeai 时发生严重错误: {e}", exc_info=True)
            # 抛出异常，因为如果配置失败，服务将无法工作
            raise ConnectionError("无法配置 Gemini API，请检查API密钥或网络连接。") from e

    def get_risk_assessment(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        向 Gemini API 发送提示词以获取风险评估结果，并期望返回JSON。

        参数:
            prompt (str): 发送给模型的完整提示词 (英文)。

        返回:
            Optional[Dict[str, Any]]: 从模型返回的JSON字符串解析成的Python字典。
                                      如果发生任何错误，则返回 None。
        """
        if not prompt:
            logger.warning("输入的提示词为空，取消API调用。")
            return None

        logger.info(f"正在向 Gemini API ({self.model_name}) 发送风险评估请求...")
        
        # 配置模型以强制返回JSON格式
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )
        
        try:
            # 发送请求
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                # 可在此添加 safety_settings 等其他参数
            )
            
            # Gemini API 在返回JSON时，会自动将 response.text 处理好
            # 我们只需要解析这个文本
            logger.debug(f"从 Gemini API 收到原始响应文本: {response.text}")
            
            # 解析JSON字符串为Python字典
            parsed_response = json.loads(response.text)
            logger.info("成功从API获取并解析了JSON响应。")
            return parsed_response

        except json.JSONDecodeError as e:
            # 如果模型返回的不是有效的JSON字符串
            logger.error(f"解析 Gemini 响应时发生JSON解码错误: {e}", exc_info=True)
            logger.error(f"无法解析的响应文本内容: '{response.text if 'response' in locals() else 'N/A'}'")
            return None
        except generation_types.BlockedPromptException as e:
            # 如果提示词因为安全原因被阻止
            logger.error(f"提示词被安全策略阻止: {e}", exc_info=True)
            return None
        except generation_types.StopCandidateException as e:
            # 如果生成的内容因为安全原因被阻止
            logger.error(f"模型生成的内容被安全策略阻止: {e}", exc_info=True)
            return None
        except google_api_exceptions.GoogleAPIError as e:
            # 处理更广泛的Google Cloud API错误 (例如，权限、资源耗尽等)
            logger.error(f"发生 Google API 核心错误: {e}", exc_info=True)
            return None
        except Exception as e:
            # 捕获其他所有可能的异常 (例如网络问题、库内部错误等)
            logger.error(f"调用 Gemini API 时发生未预期错误: {e}", exc_info=True)
            return None