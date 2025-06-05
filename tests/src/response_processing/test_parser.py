# tests/src/response_processing/test_parser.py
import pytest
import logging

# 导入被测试的类
from src.response_processing.parser import ResponseParser

# --- Fixtures ---
@pytest.fixture
def parser() -> ResponseParser:
    """提供一个 ResponseParser 实例。"""
    return ResponseParser()

# --- 测试用例 ---
class TestResponseParser:

    # --- 成功场景测试 ---
    def test_parse_success_case_A(self, parser: ResponseParser):
        """测试一个标准、成功的 'A' 风险响应能否被正确解析。"""
        llm_response = {'risk_label': 'A', 'explanation': 'This clause is risky due to ambiguity.'}
        expected = {'risk_label': 'A', 'explanation': 'This clause is risky due to ambiguity.'}
        result = parser.parse(llm_response)
        assert result == expected

    def test_parse_success_with_normalization(self, parser: ResponseParser):
        """测试带有额外空格和不同大小写的有效响应能否被正确规范化。"""
        llm_response = {'risk_label': '  b ', 'explanation': '  No risks found.  '}
        expected = {'risk_label': 'B', 'explanation': 'No risks found.'}
        result = parser.parse(llm_response)
        assert result == expected
    
    def test_parse_explanation_is_not_string(self, parser: ResponseParser):
        """测试当 'explanation' 不是字符串时，能否被健壮地处理。"""
        llm_response = {'risk_label': 'A', 'explanation': 12345}
        expected = {'risk_label': 'A', 'explanation': '12345'}
        result = parser.parse(llm_response)
        assert result == expected

    # --- 失败和边界场景测试 ---
    def test_parse_input_is_none(self, parser: ResponseParser, caplog):
        """测试当输入为 None 时，应返回 None。"""
        with caplog.at_level(logging.INFO):
            result = parser.parse(None)
        assert result is None
        assert "输入的LLM响应为 None" in caplog.text

    def test_parse_input_not_a_dict(self, parser: ResponseParser, caplog):
        """测试当输入不是字典时，应返回 None 并记录警告。"""
        with caplog.at_level(logging.WARNING):
            result = parser.parse("this is a string, not a dict")
        assert result is None
        assert "不是一个字典" in caplog.text

    def test_parse_missing_risk_label_key(self, parser: ResponseParser, caplog):
        """测试当缺少 'risk_label' 键时，应返回 None 并记录警告。"""
        llm_response = {'explanation': 'Some text without a label.'}
        with caplog.at_level(logging.WARNING):
            result = parser.parse(llm_response)
        assert result is None
        assert "缺少必需的键: 'risk_label'" in caplog.text

    def test_parse_missing_explanation_key(self, parser: ResponseParser, caplog):
        """测试当缺少 'explanation' 键时，应返回 None 并记录警告。"""
        llm_response = {'risk_label': 'A'}
        with caplog.at_level(logging.WARNING):
            result = parser.parse(llm_response)
        assert result is None
        assert "缺少必需的键: 'explanation'" in caplog.text

    def test_parse_invalid_risk_label_value(self, parser: ResponseParser, caplog):
        """测试当 'risk_label' 的值无效时（不是'A'或'B'），应返回 None 并记录警告。"""
        llm_response = {'risk_label': 'C', 'explanation': 'Invalid label.'}
        with caplog.at_level(logging.WARNING):
            result = parser.parse(llm_response)
        assert result is None
        assert "值无效: 'C'" in caplog.text

    def test_parse_risk_label_is_not_string(self, parser: ResponseParser, caplog):
        """测试当 'risk_label' 不是字符串类型时，应返回 None 并记录警告。"""
        llm_response = {'risk_label': True, 'explanation': 'Label is a boolean.'}
        with caplog.at_level(logging.WARNING):
            result = parser.parse(llm_response)
        assert result is None
        assert "'risk_label' 不是字符串类型" in caplog.text