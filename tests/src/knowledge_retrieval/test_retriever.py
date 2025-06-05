# tests/src/knowledge_retrieval/test_retriever.py
import pytest
import pandas as pd
import logging

from src.knowledge_retrieval.retriever import KnowledgeRetriever

# --- Test Fixtures ---
@pytest.fixture
def sample_checklist_df() -> pd.DataFrame:
    """创建一个模拟的、预处理过的 checklist_df."""
    data = {
        'risk_feature': [
            'payment terms', # 假设已小写和strip
            'termination clause',
            'data privacy'
        ],
        'identified_items': [
            'Unclear payment schedule or excessive late fees.',
            'No provision for early termination by lessee.',
            'Lack of GDPR compliance statements.'
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_expert_pairs_df() -> pd.DataFrame:
    """创建一个模拟的、预处理过的 expert_pairs_df."""
    data = {
        'checkpoint': [ # 假设已小写和strip
            'payment terms', 
            'payment terms', 
            'payment terms', # 3个 payment terms 案例
            'data privacy',
            'confidentiality' # 这个checkpoint在checklist中没有
        ],
        'clause': [
            'Lessee shall pay rent monthly.', 
            'Rent is due on the 1st, a 10% late fee applies after 5 days.',
            'All payments must be made via wire transfer.',
            'All user data will be handled according to internal policy.',
            'Neither party shall disclose confidential information.'
        ],
        'review': [
            'Compliant: Clear and simple.',
            'Risky: Late fee might be too high, check local regulations.',
            'Neutral: Payment method specified.',
            'Non-compliant: "Internal policy" is vague, needs GDPR reference.',
            'Compliant: Standard confidentiality clause.'
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def retriever_instance(sample_checklist_df, sample_expert_pairs_df) -> KnowledgeRetriever:
    """提供一个用模拟数据初始化的 KnowledgeRetriever 实例。"""
    return KnowledgeRetriever(checklist_df=sample_checklist_df, expert_pairs_df=sample_expert_pairs_df)

# --- 测试用例 ---

class TestKnowledgeRetriever:

    def test_initialization_success(self, sample_checklist_df, sample_expert_pairs_df):
        """测试KnowledgeRetriever能否成功初始化。"""
        try:
            retriever = KnowledgeRetriever(sample_checklist_df, sample_expert_pairs_df)
            assert retriever is not None
        except Exception as e:
            pytest.fail(f"KnowledgeRetriever 初始化失败: {e}")

    def test_initialization_missing_columns_checklist(self, sample_expert_pairs_df):
        """测试checklist_df缺少必需列时初始化是否抛出ValueError。"""
        faulty_checklist_df = pd.DataFrame({'risk_feature_WRONG': ['a'], 'identified_items': ['b']})
        with pytest.raises(ValueError, match="checklist_df 缺少必需的列"):
            KnowledgeRetriever(faulty_checklist_df, sample_expert_pairs_df)

    def test_initialization_missing_columns_expert(self, sample_checklist_df):
        """测试expert_pairs_df缺少必需列时初始化是否抛出ValueError。"""
        faulty_expert_df = pd.DataFrame({'checkpoint_WRONG': ['a'], 'clause':['b'], 'review':['c']})
        with pytest.raises(ValueError, match="expert_pairs_df 缺少必需的列"):
            KnowledgeRetriever(sample_checklist_df, faulty_expert_df)

    # --- 测试 retrieve_checklist_info ---
    def test_retrieve_checklist_info_found(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'payment terms' # 已规范化
        result = retriever_instance.retrieve_checklist_info(checkpoint)
        assert result is not None
        assert result['risk_feature'] == 'payment terms'
        assert result['identified_items'] == 'Unclear payment schedule or excessive late fees.'

    def test_retrieve_checklist_info_not_found(self, retriever_instance: KnowledgeRetriever, caplog):
        checkpoint = 'non_existent_checkpoint'
        with caplog.at_level(logging.WARNING):
            result = retriever_instance.retrieve_checklist_info(checkpoint)
        assert result is None
        assert f"未能在风险清单中找到与 checkpoint '{checkpoint}' 匹配的条目。" in caplog.text

    def test_retrieve_checklist_info_invalid_input(self, retriever_instance: KnowledgeRetriever, caplog):
        with caplog.at_level(logging.WARNING):
            result_none = retriever_instance.retrieve_checklist_info(None)
            result_empty = retriever_instance.retrieve_checklist_info("")
        assert result_none is None
        assert result_empty is None
        assert "提供的 contract_checkpoint 无效" in caplog.text

    # --- 测试 retrieve_expert_examples ---
    def test_retrieve_expert_examples_found_multiple_request_less(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'payment terms' # 有3个案例
        num_to_get = 2
        results = retriever_instance.retrieve_expert_examples(checkpoint, num_examples=num_to_get)
        assert len(results) == num_to_get
        # 验证返回的是预期的前两个 (因为我们用了 .head())
        assert results[0]['clause'] == 'Lessee shall pay rent monthly.'
        assert results[1]['clause'] == 'Rent is due on the 1st, a 10% late fee applies after 5 days.'

    def test_retrieve_expert_examples_found_multiple_request_more(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'payment terms' # 有3个案例
        num_to_get = 5
        results = retriever_instance.retrieve_expert_examples(checkpoint, num_examples=num_to_get)
        assert len(results) == 3 # 最多返回所有匹配到的

    def test_retrieve_expert_examples_found_one(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'data privacy' # 有1个案例
        results = retriever_instance.retrieve_expert_examples(checkpoint, num_examples=2)
        assert len(results) == 1
        assert results[0]['clause'] == 'All user data will be handled according to internal policy.'

    def test_retrieve_expert_examples_not_found(self, retriever_instance: KnowledgeRetriever, caplog):
        checkpoint = 'non_existent_checkpoint_for_expert'
        with caplog.at_level(logging.WARNING):
            results = retriever_instance.retrieve_expert_examples(checkpoint)
        assert len(results) == 0
        assert f"未能在专家知识库中找到与 checkpoint '{checkpoint}' 匹配的案例。" in caplog.text
    
    def test_retrieve_expert_examples_num_examples_zero(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'payment terms'
        results = retriever_instance.retrieve_expert_examples(checkpoint, num_examples=0)
        assert len(results) == 0

    def test_retrieve_expert_examples_invalid_input(self, retriever_instance: KnowledgeRetriever, caplog):
        with caplog.at_level(logging.WARNING):
            results_none_cp = retriever_instance.retrieve_expert_examples(None)
            results_empty_cp = retriever_instance.retrieve_expert_examples("")
            results_neg_num = retriever_instance.retrieve_expert_examples("payment terms", num_examples=-1)
        assert len(results_none_cp) == 0
        assert len(results_empty_cp) == 0
        assert len(results_neg_num) == 0 # 因为num_examples会被设为0
        assert "提供的 contract_checkpoint 无效" in caplog.text
        assert "num_examples (-1) 无效" in caplog.text


    # --- 测试主方法 retrieve_knowledge ---
    def test_retrieve_knowledge_all_found(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'payment terms' # 规范化后的
        knowledge = retriever_instance.retrieve_knowledge(checkpoint, num_expert_examples=2)
        
        assert knowledge['contract_checkpoint'] == checkpoint
        assert knowledge['checklist_info'] is not None
        assert knowledge['checklist_info']['risk_feature'] == 'payment terms'
        assert len(knowledge['expert_examples']) == 2

    def test_retrieve_knowledge_only_checklist_found(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'termination clause' # 只有checklist信息，没有专家案例
        knowledge = retriever_instance.retrieve_knowledge(checkpoint)
        
        assert knowledge['checklist_info'] is not None
        assert knowledge['checklist_info']['risk_feature'] == 'termination clause'
        assert len(knowledge['expert_examples']) == 0

    def test_retrieve_knowledge_only_expert_found(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'confidentiality' # 只有专家案例，没有checklist信息
        knowledge = retriever_instance.retrieve_knowledge(checkpoint)
        
        assert knowledge['checklist_info'] is None
        assert len(knowledge['expert_examples']) == 1
        assert knowledge['expert_examples'][0]['clause'] == 'Neither party shall disclose confidential information.'

    def test_retrieve_knowledge_none_found(self, retriever_instance: KnowledgeRetriever):
        checkpoint = 'completely_unknown_checkpoint'
        knowledge = retriever_instance.retrieve_knowledge(checkpoint)
        
        assert knowledge['checklist_info'] is None
        assert len(knowledge['expert_examples']) == 0