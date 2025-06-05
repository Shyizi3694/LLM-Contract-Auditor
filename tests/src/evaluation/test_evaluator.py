# tests/src/evaluation/test_evaluator.py
import pytest
import pandas as pd
import logging
import numpy as np

from src.evaluation.evaluator import Evaluator

# --- Fixtures ---
@pytest.fixture
def sample_ground_truth_df() -> pd.DataFrame:
    """创建一个模拟的、预处理过的 ground_truth_df。"""
    data = {
        'checkpoint': ['cp1', 'cp1', 'cp2', 'cp2', 'cp3'],
        'clause': [
            'clause a', # cp1, clause a -> A (risk)
            'clause b', # cp1, clause b -> B (no risk)
            'clause c', # cp2, clause c -> B (no risk)
            'clause d', # cp2, clause d -> A (risk)
            'clause e'  # cp3, clause e -> B (no risk)
        ],
        'gt': ['A', 'B', 'B', 'A', 'B']
    }
    return pd.DataFrame(data)

@pytest.fixture
def evaluator_instance(sample_ground_truth_df: pd.DataFrame) -> Evaluator:
    """提供一个用模拟数据初始化的 Evaluator 实例。"""
    return Evaluator(ground_truth_df=sample_ground_truth_df)


# --- 测试用例 ---
class TestEvaluator:

    def test_initialization(self, sample_ground_truth_df: pd.DataFrame):
        """测试Evaluator能否成功初始化并创建查找字典。"""
        evaluator = Evaluator(sample_ground_truth_df)
        assert len(evaluator.gt_lookup) == 5
        assert evaluator.gt_lookup[('cp1', 'clause a')] == 'A'
        assert evaluator.gt_lookup[('cp2', 'clause c')] == 'B'

    def test_initialization_with_empty_or_bad_df(self):
        """测试用空的或格式错误的DataFrame初始化时是否抛出异常。"""
        with pytest.raises(ValueError, match="必须是一个非空的"):
            Evaluator(pd.DataFrame())
        
        faulty_df = pd.DataFrame({'col1': [1], 'col2': [2]})
        with pytest.raises(ValueError, match="缺少必需的列"):
            Evaluator(faulty_df)

    def test_evaluate_perfect_predictions(self, evaluator_instance: Evaluator):
        """测试当所有预测都正确时，指标是否都为1.0。"""
        predictions = [
            {'checkpoint': 'cp1', 'clause': 'clause a', 'predicted_label': 'A'}, # TP
            {'checkpoint': 'cp1', 'clause': 'clause b', 'predicted_label': 'B'}, # TN
            {'checkpoint': 'cp2', 'clause': 'clause c', 'predicted_label': 'B'}, # TN
            {'checkpoint': 'cp2', 'clause': 'clause d', 'predicted_label': 'A'}, # TP
            {'checkpoint': 'cp3', 'clause': 'clause e', 'predicted_label': 'B'}, # TN
        ]
        metrics = evaluator_instance.evaluate(predictions)
        
        assert metrics['f1_score'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['accuracy'] == 1.0
        # confusion_matrix [[TP, FN], [FP, TN]] with labels=['A', 'B']
        # A是positive, B是negative
        # TP: pred=A, true=A -> 2
        # FN: pred=B, true=A -> 0
        # FP: pred=A, true=B -> 0
        # TN: pred=B, true=B -> 3
        # scikit-learn cm: [[TP, FN], [FP, TN]] for labels=['A', 'B'] -> [[2,0],[0,3]]
        assert metrics['confusion_matrix'] == [[2, 0], [0, 3]]
        assert metrics['matched_predictions'] == 5

    def test_evaluate_mixed_results(self, evaluator_instance: Evaluator):
        """测试包含各种结果（TP, FP, TN, FN）的场景。"""
        predictions = [
            {'checkpoint': 'cp1', 'clause': 'clause a', 'predicted_label': 'A'}, # TP: true=A, pred=A
            {'checkpoint': 'cp1', 'clause': 'clause b', 'predicted_label': 'A'}, # FP: true=B, pred=A
            {'checkpoint': 'cp2', 'clause': 'clause c', 'predicted_label': 'B'}, # TN: true=B, pred=B
            {'checkpoint': 'cp2', 'clause': 'clause d', 'predicted_label': 'B'}, # FN: true=A, pred=B
            # 'clause e' is not predicted
        ]
        # TP=1, FP=1, FN=1, TN=1
        # Precision = TP / (TP+FP) = 1 / (1+1) = 0.5
        # Recall = TP / (TP+FN) = 1 / (1+1) = 0.5
        # F1 = 2 * (Prec * Rec) / (Prec + Rec) = 2 * (0.5*0.5)/(0.5+0.5) = 0.5
        # Accuracy = (TP+TN) / (TP+FP+FN+TN) = (1+1)/(1+1+1+1) = 0.5
        
        metrics = evaluator_instance.evaluate(predictions)
        
        assert pytest.approx(metrics['f1_score']) == 0.5
        assert pytest.approx(metrics['precision']) == 0.5
        assert pytest.approx(metrics['recall']) == 0.5
        assert pytest.approx(metrics['accuracy']) == 0.5
        # cm for labels=['A', 'B']: [[TP, FN], [FP, TN]] -> [[1, 1], [1, 1]]
        assert metrics['confusion_matrix'] == [[1, 1], [1, 1]]
        assert metrics['matched_predictions'] == 4

    def test_evaluate_with_unmatched_prediction(self, evaluator_instance: Evaluator, caplog):
        """测试当预测无法在ground truth中找到匹配项时，是否会被跳过。"""
        predictions = [
            {'checkpoint': 'cp1', 'clause': 'clause a', 'predicted_label': 'A'}, # Matched
            {'checkpoint': 'cp_unknown', 'clause': 'clause_unknown', 'predicted_label': 'A'}, # Unmatched
        ]
        
        with caplog.at_level(logging.WARNING):
            metrics = evaluator_instance.evaluate(predictions)
        
        assert "未能为以下条款找到真实标签" in caplog.text
        assert metrics['total_predictions'] == 2
        assert metrics['matched_predictions'] == 1
        assert metrics['unmatched_predictions'] == 1
        # 此时 y_true=['A'], y_pred=['A'], 所以所有指标都是1.0
        assert metrics['f1_score'] == 1.0

    def test_evaluate_with_incomplete_prediction(self, evaluator_instance: Evaluator, caplog):
        """测试当预测记录格式不完整时，是否会被跳过。"""
        predictions = [
            {'checkpoint': 'cp1', 'clause': 'clause a', 'predicted_label': 'A'}, # Matched
            {'checkpoint': 'cp1'}, # Incomplete
        ]
        with caplog.at_level(logging.WARNING):
            metrics = evaluator_instance.evaluate(predictions)
        assert "跳过格式不完整的预测记录" in caplog.text
        assert metrics['unmatched_predictions'] == 1

    def test_evaluate_with_empty_predictions(self, evaluator_instance: Evaluator, caplog):
        """测试当传入空的预测列表时，能否正常处理。"""
        with caplog.at_level(logging.WARNING):
            metrics = evaluator_instance.evaluate([])
        
        assert "输入的预测列表为空" in caplog.text
        assert metrics['f1_score'] == 0.0
        assert metrics['matched_predictions'] == 0

    def test_evaluate_no_matches_found(self, evaluator_instance: Evaluator, caplog):
        """测试当所有预测都无法匹配到ground truth时，能否正常处理。"""
        predictions = [
            {'checkpoint': 'cp_unknown', 'clause': 'clause_unknown', 'predicted_label': 'A'},
        ]
        with caplog.at_level(logging.ERROR):
            metrics = evaluator_instance.evaluate(predictions)
        
        assert "没有任何预测能够匹配到真实标签" in caplog.text
        assert metrics['f1_score'] == 0.0