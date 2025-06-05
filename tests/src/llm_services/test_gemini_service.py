# tests/src/llm_services/test_gemini_service.py
import pytest
import unittest.mock
import json
import logging

# 导入被测试的类
from src.llm_services.gemini_service import GeminiService
# 导入需要被 mock 的库
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_api_exceptions

# --- 测试用例 ---

# 使用 patch.multiple 来一次性 mock 多个模块和函数，在测试函数执行期间生效
@unittest.mock.patch.multiple(
    'src.llm_services.gemini_service', # 要打补丁的模块路径
    # 将模块内的 genai 变量替换为 MagicMock 对象
    genai=unittest.mock.DEFAULT,
    # 将模块内的 API_KEY 和 MODEL_NAME 替换为测试用的值
    API_KEY="test_api_key_is_present", 
    MODEL_NAME="test-model"
)
def test_gemini_service_initialization_success(**mocks):
    """测试 GeminiService 在配置正确时能否成功初始化。"""
    # **mocks 是 patch.multiple 注入的 mocks 字典
    # 我们可以用它来配置 mock 对象的行为，但在这个测试中暂时不需要
    try:
        service = GeminiService()
        assert service is not None
        assert service.model_name == "test-model"
        # 验证 genai.configure 和 genai.GenerativeModel 是否被调用
        mocks['genai'].configure.assert_called_once_with(api_key="test_api_key_is_present")
        mocks['genai'].GenerativeModel.assert_called_once_with(model_name="test-model")
    except Exception as e:
        pytest.fail(f"GeminiService 初始化不应失败: {e}")

@unittest.mock.patch.multiple('src.llm_services.gemini_service', API_KEY=None, MODEL_NAME="test-model")
def test_gemini_service_initialization_no_api_key(**mocks):
    """测试当 API_KEY 为 None 时，初始化是否抛出 ValueError。"""
    with pytest.raises(ValueError, match="Gemini API Key 未在配置中找到或加载失败"):
        GeminiService()

@unittest.mock.patch.multiple(
    'src.llm_services.gemini_service',
    genai=unittest.mock.DEFAULT,
    API_KEY="test_key", 
    MODEL_NAME="test-model"
)
def test_get_risk_assessment_success(**mocks):
    """测试 get_risk_assessment 成功获取并解析JSON响应的场景。"""
    # 1. 准备 mock 数据
    mock_genai = mocks['genai'] # 获取 genai 的 mock 对象
    mock_model_instance = mock_genai.GenerativeModel.return_value
    mock_response = unittest.mock.MagicMock()
    
    expected_dict = {"risk_label": "A", "explanation": "This is a risky clause."}
    mock_response.text = json.dumps(expected_dict)
    mock_model_instance.generate_content.return_value = mock_response

    # 2. 调用被测试的方法
    service = GeminiService()
    result = service.get_risk_assessment("some prompt")
    
    # 3. 断言返回结果正确
    assert result == expected_dict, "返回的结果应与期望的字典一致"
    
    # --- 4. 修改后的 API 调用验证 ---
    
    # 4a. 验证 genai.GenerationConfig 是否以正确的参数被调用了一次
    mock_genai.GenerationConfig.assert_called_once_with(
        response_mime_type="application/json"
    )

    # 4b. 验证 generate_content 是否被正确调用
    mock_model_instance.generate_content.assert_called_once()
    
    # (可选) 验证传递给 generate_content 的 generation_config 参数
    # 正是我们 mock 的 GenerationConfig() 的返回值
    args, kwargs = mock_model_instance.generate_content.call_args
    assert "generation_config" in kwargs
    # 确认传递的对象就是 mock 构造函数返回的那个 mock 对象
    assert kwargs["generation_config"] == mock_genai.GenerationConfig.return_value 


@unittest.mock.patch.multiple('src.llm_services.gemini_service', genai=unittest.mock.DEFAULT, API_KEY="test_key")
def test_get_risk_assessment_api_error(caplog, **mocks):
    """测试当API调用抛出通用异常时，方法是否返回None并记录错误。"""
    # 配置 mock 模型在被调用时抛出异常
    mock_model_instance = mocks['genai'].GenerativeModel.return_value
    mock_model_instance.generate_content.side_effect = google_api_exceptions.InternalServerError("API is down")

    with caplog.at_level(logging.ERROR):
        service = GeminiService()
        result = service.get_risk_assessment("some prompt")

    assert result is None, "发生API错误时应返回 None"
    assert "发生 Google API 核心错误" in caplog.text
    assert "API is down" in caplog.text


@unittest.mock.patch.multiple('src.llm_services.gemini_service', genai=unittest.mock.DEFAULT, API_KEY="test_key")
def test_get_risk_assessment_invalid_json(caplog, **mocks):
    """测试当API返回非JSON文本时，方法是否返回None并记录错误。"""
    # 配置 mock 模型返回一个非JSON的响应
    mock_model_instance = mocks['genai'].GenerativeModel.return_value
    mock_response = unittest.mock.MagicMock()
    mock_response.text = "This is not a valid JSON response."
    mock_model_instance.generate_content.return_value = mock_response

    with caplog.at_level(logging.ERROR):
        service = GeminiService()
        result = service.get_risk_assessment("some prompt")

    assert result is None, "响应为无效JSON时应返回 None"
    assert "解析 Gemini 响应时发生JSON解码错误" in caplog.text


@unittest.mock.patch.multiple('src.llm_services.gemini_service', genai=unittest.mock.DEFAULT, API_KEY="test_key")
def test_get_risk_assessment_safety_blocked(caplog, **mocks):
    """测试当提示词或响应被安全策略阻止时，方法是否返回None并记录错误。"""
    # 配置 mock 模型在被调用时抛出 BlockedPromptException
    mock_model_instance = mocks['genai'].GenerativeModel.return_value
    mock_model_instance.generate_content.side_effect = genai_types.BlockedPromptException("Prompt blocked for safety reasons.")

    with caplog.at_level(logging.ERROR):
        service = GeminiService()
        result = service.get_risk_assessment("some prompt")

    assert result is None, "安全阻止时应返回 None"
    assert "提示词被安全策略阻止" in caplog.text


def test_get_risk_assessment_empty_prompt(caplog):
    """测试当传入空提示词时，是否不调用API并记录警告。"""
    # 这个测试不需要mock API，因为它应该在调用API前就返回
    # 但我们仍然需要mock初始化过程
    with unittest.mock.patch.multiple(
        'src.llm_services.gemini_service', 
        genai=unittest.mock.DEFAULT, 
        API_KEY="test_key"
    ):
        with caplog.at_level(logging.WARNING):
            service = GeminiService()
            result = service.get_risk_assessment("") # 传入空字符串

        assert result is None
        assert "输入的提示词为空，取消API调用。" in caplog.text