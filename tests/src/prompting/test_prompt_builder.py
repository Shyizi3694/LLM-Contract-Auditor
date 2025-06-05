# tests/src/prompting/test_prompt_builder.py
import pytest
import logging # 用于 caplog fixture
from src.prompting.prompt_builder import PromptBuilder # 假设 PromptBuilder 在此路径

# --- Fixtures ---
@pytest.fixture
def builder() -> PromptBuilder:
    """提供一个 PromptBuilder 实例。"""
    return PromptBuilder()

@pytest.fixture
def sample_clause_data_all_present() -> dict:
    """提供包含所有组件的示例数据。"""
    return {
        "contract_clause": "The lessee agrees to pay a monthly rent of $1000, due on the 1st of each month. "
                           "A late fee of 10% per day will be applied to overdue payments.",
        "contract_checkpoint": "payment terms", # 假设这是规范化后的
        "checklist_info": {
            'risk_feature': 'payment terms', # 假设这个与上面的checkpoint匹配
            'identified_items': 'Ensure clarity on amount, due date, penalties. Verify penalty reasonableness.'
        },
        "expert_examples": [
            {'clause': 'Rent is $500 due on the 1st of the month.', 
             'review': 'Compliant: Clear amount and due date. No penalty mentioned, might need clarification if desired.'},
            {'clause': 'A penalty of 50% of outstanding rent applies for late payments.', 
             'review': 'High Risk: Penalty is excessively high and likely unenforceable.'}
        ]
    }

@pytest.fixture
def sample_clause_data_no_checklist(sample_clause_data_all_present: dict) -> dict:
    data = sample_clause_data_all_present.copy()
    data["checklist_info"] = None
    return data

@pytest.fixture
def sample_clause_data_no_experts(sample_clause_data_all_present: dict) -> dict:
    data = sample_clause_data_all_present.copy()
    data["expert_examples"] = []
    return data

@pytest.fixture
def sample_clause_data_no_context(sample_clause_data_all_present: dict) -> dict:
    data = sample_clause_data_all_present.copy()
    data["checklist_info"] = None
    data["expert_examples"] = []
    return data

# --- 测试用例 ---
class TestPromptBuilder:

    def test_build_prompt_all_info_present(self, builder: PromptBuilder, sample_clause_data_all_present: dict):
        data = sample_clause_data_all_present
        prompt = builder.build_prompt(
            contract_clause=data["contract_clause"],
            contract_checkpoint=data["contract_checkpoint"],
            checklist_info=data["checklist_info"],
            expert_examples=data["expert_examples"]
        )
        # 检查关键指令和占位符内容是否存在
        assert "You are an expert legal assistant" in prompt
        assert "Respond strictly with a single JSON object" in prompt
        assert "[Relevant Risk Checklist Information]" in prompt
        assert data["checklist_info"]["risk_feature"] in prompt
        assert data["checklist_info"]["identified_items"] in prompt
        assert "[Expert Review Examples for Similar Clause Categories]" in prompt
        assert data["expert_examples"][0]["clause"] in prompt
        assert data["expert_examples"][0]["review"] in prompt
        assert data["expert_examples"][1]["clause"] in prompt
        assert data["expert_examples"][1]["review"] in prompt
        assert "[Contract Clause to Review]" in prompt
        
        # 调试：打印完整的prompt
        print("-----完整生成的Prompt内容开始-----")
        print(prompt)
        print("-----完整生成的Prompt内容结束-----")
        
        expected_string = f"Original Checkpoint Category of the Clause: \"{data['contract_checkpoint']}\""
        print(f"期望包含的字符串: '{expected_string}'")

        # --- 修改后的断言方式 ---
        found_line_in_prompt = False
        for line in prompt.split('\n'):
            # 为了更稳健的比较，可以先去除行首尾可能存在的空格
            if expected_string == line.strip():
                found_line_in_prompt = True
                break
        
        assert found_line_in_prompt, \
            f"期望的字符串 '{expected_string}' 未在生成的 prompt 中作为独立的一行准确找到。\n" \
            f"请检查 prompt 的实际内容和期望字符串是否有细微差异（如空格、特殊字符）。"
        # --- 修改后的断言方式结束 ---

        # assert f"Original Checkpoint Category: \"{data['contract_checkpoint']}\"" in prompt
        assert data["contract_clause"] in prompt
        assert f"\"\"\"\n{data['contract_clause']}\n\"\"\"" in prompt # 检查三引号格式

    def test_build_prompt_no_checklist_info(self, builder: PromptBuilder, sample_clause_data_no_checklist: dict):
        data = sample_clause_data_no_checklist
        prompt = builder.build_prompt(
            contract_clause=data["contract_clause"],
            contract_checkpoint=data["contract_checkpoint"],
            checklist_info=data["checklist_info"], # This is None
            expert_examples=data["expert_examples"]
        )
        assert "No specific risk checklist information was retrieved or provided" in prompt
        assert data["expert_examples"][0]["clause"] in prompt # 专家案例应仍在

    def test_build_prompt_no_expert_examples(self, builder: PromptBuilder, sample_clause_data_no_experts: dict):
        data = sample_clause_data_no_experts
        prompt = builder.build_prompt(
            contract_clause=data["contract_clause"],
            contract_checkpoint=data["contract_checkpoint"],
            checklist_info=data["checklist_info"],
            expert_examples=data["expert_examples"] # This is []
        )
        assert "No specific expert review examples were retrieved or provided" in prompt
        assert data["checklist_info"]["risk_feature"] in prompt # 清单信息应仍在

    def test_build_prompt_no_contextual_info(self, builder: PromptBuilder, sample_clause_data_no_context: dict):
        data = sample_clause_data_no_context
        prompt = builder.build_prompt(
            contract_clause=data["contract_clause"],
            contract_checkpoint=data["contract_checkpoint"],
            checklist_info=data["checklist_info"], # None
            expert_examples=data["expert_examples"] # []
        )
        assert "No specific risk checklist information was retrieved or provided" in prompt
        assert "No specific expert review examples were retrieved or provided" in prompt
        assert data["contract_clause"] in prompt # 原始条款应仍在

    def test_build_prompt_with_empty_clause_string(self, builder: PromptBuilder, sample_clause_data_all_present: dict, caplog):
        data = sample_clause_data_all_present
        empty_clause = ""
        with caplog.at_level(logging.WARNING):
            prompt = builder.build_prompt(
                contract_clause=empty_clause,
                contract_checkpoint=data["contract_checkpoint"],
                checklist_info=data["checklist_info"],
                expert_examples=data["expert_examples"]
            )
        # --- 修改这里的期望日志字符串为中文 ---
        expected_warning = "输入的合同条款为空，生成的提示词可能效果不佳。"
        
        assert expected_warning in caplog.text, \
            f"期望捕获的中文警告日志未找到。期望: '{expected_warning}', 实际捕获: '{caplog.text}'"

        assert 'Clause Text requiring your assessment:\n"""\n\n"""' in prompt

    def test_build_prompt_handles_malformed_expert_example(self, builder: PromptBuilder, sample_clause_data_all_present: dict, caplog):
        data = sample_clause_data_all_present
        malformed_dict = {"wrong_key_clause": "This is a malformed example.", "wrong_key_review": "It lacks 'clause' and 'review'."}
        malformed_examples = [
            data["expert_examples"][0], # 第一个正常
            malformed_dict,             # 格式错误的字典
            data["expert_examples"][1]  # 第二个正常
        ]

        with caplog.at_level(logging.WARNING):
            prompt = builder.build_prompt(
                contract_clause=data["contract_clause"],
                contract_checkpoint=data["contract_checkpoint"],
                checklist_info=data["checklist_info"],
                expert_examples=malformed_examples
            )
        
        # --- 新的、更健壮的断言方式 ---
        # 1. 确认有一条警告日志被记录了
        assert len(caplog.records) >= 1, "应该至少有一条警告日志被记录"

        # 2. 找到那条特定的警告日志
        found_the_log = False
        static_expected_message = "跳过格式不正确的专家案例:"
        
        for record in caplog.records:
            if record.levelno == logging.WARNING and static_expected_message in record.message:
                # 3. (可选但推荐) 验证这条日志消息中是否包含了错误字典的关键内容，而不用关心其精确格式
                assert "'wrong_key_clause'" in record.message
                assert "'wrong_key_review'" in record.message
                assert "This is a malformed example." in record.message
                found_the_log = True
                break
        
        assert found_the_log, f"期望的警告日志 '{static_expected_message}' 未在捕获的日志中找到。"

        # --- 其他断言保持不变 ---
        # 验证正常的案例应该还在提示词中
        assert data["expert_examples"][0]['clause'] in prompt
        assert data["expert_examples"][1]['clause'] in prompt
        # 验证格式错误的案例内容不应出现在提示词中
        assert "This is a malformed example." not in prompt