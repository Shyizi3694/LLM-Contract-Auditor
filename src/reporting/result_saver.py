# src/reporting/result_saver.py

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

# 导入 Matplotlib 和 Seaborn 用于绘图
# 这需要确保这两个库已安装: pip install matplotlib seaborn
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_LIBS_AVAILABLE = True
except ImportError:
    PLOTTING_LIBS_AVAILABLE = False
    logging.warning("Matplotlib 或 Seaborn 未安装。绘图功能将不可用。请运行 'pip install matplotlib seaborn'。")

# 从配置中导入路径
try:
    from src.common.config import METRICS_PATH, PLOTS_PATH
except ImportError:
    # 回退路径，用于直接测试或路径问题
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    METRICS_PATH = os.path.join(_project_root, "results", "metrics")
    PLOTS_PATH = os.path.join(_project_root, "results", "plots")

logger = logging.getLogger(__name__)

class ResultSaver:
    def __init__(self, experiment_name: str | None = None):
        """
        初始化结果保存器。
        会根据实验名称创建一个独特的子目录来存放本次运行的所有结果。

        参数:
            experiment_name (str | None): 本次实验的名称。
                如果为 None，将根据当前时间自动生成一个。
        """
        if experiment_name is None:
            # 根据当前时间生成一个唯一的实验名称，例如 'exp_20250605_163000'
            # 注意：这里使用的是您当前时区的时间
            self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
        
        # 定义本次实验结果的专属存储路径
        self.experiment_metrics_path = os.path.join(METRICS_PATH, self.experiment_name)
        self.experiment_plots_path = os.path.join(PLOTS_PATH, self.experiment_name)
        
        # 创建这些目录（如果它们不存在）
        os.makedirs(self.experiment_metrics_path, exist_ok=True)
        os.makedirs(self.experiment_plots_path, exist_ok=True)
        
        logger.info(f"ResultSaver 初始化完成，结果将保存到以下目录中：")
        logger.info(f" - 指标: {self.experiment_metrics_path}")
        logger.info(f" - 图表: {self.experiment_plots_path}")

    def save_metrics(self, metrics: Dict[str, Any]):
        """
        将评估指标字典保存为 JSON 文件。

        参数:
            metrics (Dict[str, Any]): 来自 Evaluator.evaluate() 的指标字典。
        """
        if not isinstance(metrics, dict):
            logger.error("提供的 metrics 不是一个字典，无法保存。")
            return

        file_path = os.path.join(self.experiment_metrics_path, "evaluation_metrics.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 使用 indent=4 使 JSON 文件格式优美，方便阅读
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            logger.info(f"评估指标已成功保存到: {file_path}")
        except TypeError as e:
            logger.error(f"序列化指标为JSON时出错（可能包含非JSON兼容类型）: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"保存指标文件时发生错误: {e}", exc_info=True)

    def save_confusion_matrix_plot(self, cm: List[List[int]], class_labels: List[str]):
        """
        将混淆矩阵绘制成热力图并保存为图片文件。

        参数:
            cm (List[List[int]]): 混淆矩阵，例如 [[TP, FN], [FP, TN]]。
            class_labels (List[str]): 类别标签，例如 ['A (Risk)', 'B (No Risk)']。
        """
        if not PLOTTING_LIBS_AVAILABLE:
            logger.warning("绘图库 (Matplotlib/Seaborn) 不可用，跳过绘制混淆矩阵。")
            return
            
        if not isinstance(cm, list) or not all(isinstance(row, list) for row in cm):
            logger.error(f"提供的混淆矩阵格式不正确，无法绘图。期望格式: List[List[int]]")
            return

        file_path = os.path.join(self.experiment_plots_path, "confusion_matrix.png")
        
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f'Confusion Matrix - {self.experiment_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # 保存图表
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵图表已成功保存到: {file_path}")
            
            # 关闭图表以释放内存
            plt.close()

        except Exception as e:
            logger.error(f"绘制或保存混淆矩阵图表时发生错误: {e}", exc_info=True)

    def save_predictions_to_csv(self, predictions_df: pd.DataFrame):
        """
        将包含预测结果的完整DataFrame保存到CSV文件，便于后续深入分析。

        参数:
            predictions_df (pd.DataFrame): 包含例如条款、真实标签、预测标签等信息的DataFrame。
        """
        if not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
            logger.warning("提供的预测结果 DataFrame 为空或格式不正确，跳过保存。")
            return

        file_path = os.path.join(self.experiment_metrics_path, "detailed_predictions.csv")
        try:
            predictions_df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"详细的预测结果已保存到CSV文件: {file_path}")
        except Exception as e:
            logger.error(f"保存详细预测结果到CSV时发生错误: {e}", exc_info=True)