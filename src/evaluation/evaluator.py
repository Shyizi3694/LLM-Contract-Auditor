# src/evaluation/evaluator.py

import pandas as pd
import logging
from typing import List, Dict, Any

# 从 scikit-learn 库导入评估指标计算函数
try:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
except ImportError:
    # 提醒用户安装 scikit-learn
    logging.critical("scikit-learn 库未安装。请运行 'pip install scikit-learn'。评估模块无法工作。")
    # 定义虚拟函数以避免在导入时就崩溃，但在调用时会失败
    f1_score = precision_score = recall_score = accuracy_score = confusion_matrix = lambda *args, **kwargs: None


logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, ground_truth_df: pd.DataFrame):
        """
        初始化评估器。

        参数:
            ground_truth_df (pd.DataFrame): 预处理过的、包含真实标签的DataFrame。
                                           必需列: 'checkpoint', 'clause', 'gt'。
                                           假设 'checkpoint' 和 'clause' 列已规范化。
        """
        if not isinstance(ground_truth_df, pd.DataFrame) or ground_truth_df.empty:
            raise ValueError("提供的 ground_truth_df 必须是一个非空的 Pandas DataFrame。")
        
        required_cols = ['checkpoint', 'clause', 'gt']
        if not all(col in ground_truth_df.columns for col in required_cols):
            raise ValueError(f"ground_truth_df 缺少必需的列。需要: {required_cols}, 实际有: {list(ground_truth_df.columns)}")
        
        # 为了高效查找，创建一个以 (checkpoint, clause) 为键，gt 为值的字典
        # 假设 (checkpoint, clause) 组合是唯一的。如果不是，需要处理重复情况。
        self.gt_lookup = pd.Series(
            ground_truth_df['gt'].values, 
            index=pd.MultiIndex.from_frame(ground_truth_df[['checkpoint', 'clause']])
        ).to_dict()

        logger.info(f"评估器初始化完成，加载了 {len(self.gt_lookup)} 条真实标签记录。")

    def evaluate(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估预测结果的性能。

        参数:
            predictions (List[Dict[str, Any]]): 预测结果列表。
                每个字典应包含: 'checkpoint', 'clause', 'predicted_label'。
                假设 'checkpoint' 和 'clause' 字段已规范化。

        返回:
            Dict[str, Any]: 包含各项评估指标的字典。
        """
        if not predictions:
            logger.warning("输入的预测列表为空，无法进行评估。")
            return {
                "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
                "confusion_matrix": None, "matched_predictions": 0, "total_predictions": 0,
                "note": "No predictions provided."
            }

        y_true = []
        y_pred = []
        unmatched_count = 0

        for pred in predictions:
            # 从预测结果中提取用于匹配的键
            checkpoint = pred.get('checkpoint')
            clause = pred.get('clause')
            predicted_label = pred.get('predicted_label')

            if checkpoint is None or clause is None or predicted_label is None:
                logger.warning(f"跳过格式不完整的预测记录: {pred}")
                unmatched_count += 1
                continue
            
            # 在我们的查找字典中寻找真实标签
            true_label = self.gt_lookup.get((checkpoint, clause))
            
            if true_label is not None:
                # 找到了匹配的真实标签
                y_true.append(true_label)
                y_pred.append(predicted_label)
            else:
                # 未找到匹配项
                logger.warning(f"未能为以下条款找到真实标签，跳过评估: checkpoint='{checkpoint}', clause='{clause[:50]}...'")
                unmatched_count += 1
        
        logger.info(f"评估匹配完成。总预测数: {len(predictions)}, "
                    f"成功匹配到真实标签的预测数: {len(y_true)}, 未匹配数: {unmatched_count}")

        if not y_true:
            logger.error("没有任何预测能够匹配到真实标签，无法计算评估指标。")
            return {
                "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
                "confusion_matrix": None, "matched_predictions": 0, "total_predictions": len(predictions),
                "note": "No predictions could be matched to ground truth."
            }
        
        # 使用 scikit-learn 计算指标。
        # 'A' 通常是 "positive" class (有风险), 'B' 是 "negative" class.
        # pos_label='A' 确保了F1, precision, recall是针对“识别风险”这个任务的。
        try:
            f1 = f1_score(y_true, y_pred, pos_label='A', zero_division=0.0)
            precision = precision_score(y_true, y_pred, pos_label='A', zero_division=0.0)
            recall = recall_score(y_true, y_pred, pos_label='A', zero_division=0.0)
            accuracy = accuracy_score(y_true, y_pred)
            # 混淆矩阵的标签顺序通常是排序后的，即 ['A', 'B']
            # tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['B', 'A']).ravel()
            cm = confusion_matrix(y_true, y_pred, labels=['A', 'B']) # labels=['A', 'B'] 让矩阵顺序固定

            metrics = {
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist(), # 转为列表方便JSON序列化
                "matched_predictions": len(y_true),
                "total_predictions": len(predictions),
                "unmatched_predictions": unmatched_count
            }
            logger.info(f"评估指标计算完成: { {k: v for k, v in metrics.items() if k != 'confusion_matrix'} }")
            return metrics
        except Exception as e:
            logger.error(f"计算评估指标时发生错误: {e}", exc_info=True)
            return {}