# tests/src/reporting/test_result_saver.py
import pytest
import os
import json
from pathlib import Path
import pandas as pd

from src.reporting.result_saver import ResultSaver, PLOTTING_LIBS_AVAILABLE

# --- Fixture ---
@pytest.fixture
def temp_results_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """创建一个临时的 metrics 和 plots 根目录。"""
    metrics_dir = tmp_path / "results" / "metrics"
    plots_dir = tmp_path / "results" / "plots"
    # 注意：ResultSaver 会在这些目录下创建实验子目录
    return metrics_dir, plots_dir

# --- 测试用例 ---
class TestResultSaver:

    def test_initialization_creates_experiment_dirs(self, temp_results_dirs: tuple[Path, Path]):
        """测试 ResultSaver 初始化时是否能正确创建实验子目录。"""
        metrics_base_path, plots_base_path = temp_results_dirs
        experiment_name = "test_run_01"
        
        saver = ResultSaver(
            experiment_name=experiment_name
        )
        # 在 ResultSaver 的 __init__ 中，它使用了从 config.py 导入的路径。
        # 为了让测试指向我们的临时目录，我们需要 monkeypatch。
        # (此处为了简化，假设我们可以通过某种方式让ResultSaver使用临时路径)
        # 更好的做法是在测试中直接传递路径给 ResultSaver 的构造函数。
        # 让我们先假设 __init__ 被修改为接受路径，或者我们用 monkeypatch。
        
        # 使用 Monkeypatch
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr('src.reporting.result_saver.METRICS_PATH', str(metrics_base_path))
        monkeypatch.setattr('src.reporting.result_saver.PLOTS_PATH', str(plots_base_path))
        
        saver = ResultSaver(experiment_name=experiment_name)
        
        expected_metrics_dir = metrics_base_path / experiment_name
        expected_plots_dir = plots_base_path / experiment_name
        
        assert expected_metrics_dir.exists() and expected_metrics_dir.is_dir()
        assert expected_plots_dir.exists() and expected_plots_dir.is_dir()

    def test_save_metrics(self, temp_results_dirs: tuple[Path, Path], monkeypatch):
        """测试 save_metrics 是否能正确保存JSON文件。"""
        metrics_base_path, plots_base_path = temp_results_dirs
        experiment_name = "test_metrics_save"
        
        monkeypatch.setattr('src.reporting.result_saver.METRICS_PATH', str(metrics_base_path))
        monkeypatch.setattr('src.reporting.result_saver.PLOTS_PATH', str(plots_base_path))
        
        saver = ResultSaver(experiment_name=experiment_name)
        
        sample_metrics = {
            "f1_score": 0.88,
            "precision": 0.9,
            "recall": 0.86,
            "confusion_matrix": [[10, 2], [1, 12]]
        }
        saver.save_metrics(sample_metrics)
        
        output_file = metrics_base_path / experiment_name / "evaluation_metrics.json"
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data == sample_metrics

    @pytest.mark.skipif(not PLOTTING_LIBS_AVAILABLE, reason="Matplotlib/Seaborn not installed.")
    def test_save_confusion_matrix_plot(self, temp_results_dirs: tuple[Path, Path], monkeypatch):
        """测试 save_confusion_matrix_plot 是否能创建图片文件。"""
        metrics_base_path, plots_base_path = temp_results_dirs
        experiment_name = "test_plot_save"
        
        monkeypatch.setattr('src.reporting.result_saver.METRICS_PATH', str(metrics_base_path))
        monkeypatch.setattr('src.reporting.result_saver.PLOTS_PATH', str(plots_base_path))

        saver = ResultSaver(experiment_name=experiment_name)
        
        cm_data = [[25, 5], [3, 30]]
        labels = ['A (Risk)', 'B (No Risk)']
        saver.save_confusion_matrix_plot(cm=cm_data, class_labels=labels)
        
        output_file = plots_base_path / experiment_name / "confusion_matrix.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0 # 确保文件不是空的

    def test_save_predictions_to_csv(self, temp_results_dirs: tuple[Path, Path], monkeypatch):
        """测试保存详细预测结果到CSV文件的功能。"""
        metrics_base_path, plots_base_path = temp_results_dirs
        experiment_name = "test_csv_save"

        monkeypatch.setattr('src.reporting.result_saver.METRICS_PATH', str(metrics_base_path))
        monkeypatch.setattr('src.reporting.result_saver.PLOTS_PATH', str(plots_base_path))

        saver = ResultSaver(experiment_name=experiment_name)

        predictions_data = {
            "clause": ["clause 1 text", "clause 2 text"],
            "true_label": ["A", "B"],
            "predicted_label": ["A", "A"],
            "is_correct": [True, False]
        }
        predictions_df = pd.DataFrame(predictions_data)
        
        saver.save_predictions_to_csv(predictions_df)
        
        output_file = metrics_base_path / experiment_name / "detailed_predictions.csv"
        assert output_file.exists()
        
        loaded_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(loaded_df, predictions_df)