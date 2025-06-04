# tests/src/data_handling/test_data_preprocess.py
import pytest
import pandas as pd
import os
from pathlib import Path
import shutil # 用于在测试开始前清理可能的旧的临时processed目录

# 被测试的函数和模块
from src.data_handling.data_preprocess import preprocess_all_data, _clean_and_normalize_df
from src.common import config as app_config # 用于 monkeypatching

# --- Fixture 来创建临时的原始CSV数据 ---
@pytest.fixture
def create_temp_raw_data(tmp_path: Path) -> tuple[Path, Path]:
    """
    在临时目录中创建一些模拟的原始CSV文件。
    返回 (临时原始数据目录路径, 临时处理后数据目录路径)。
    """
    raw_dir = tmp_path / "raw_data_for_test"
    raw_dir.mkdir()
    
    processed_dir_parent = tmp_path / "processed_data_parent_for_test" # 父目录
    processed_dir_parent.mkdir()
    processed_dir = processed_dir_parent / "processed" # 目标 processed 目录
    # preprocess_all_data 应该会创建 processed_dir 如果它不存在

    # 1. contract.csv
    contract_content = {
        "checkpoint": ["  CP1 ", "cp two  ", None, "CpThRee"],
        "clause": ["Clause  A  ", "  Clause B with   extra spaces", "Clause C\nwith newline", None]
    }
    pd.DataFrame(contract_content).to_csv(raw_dir / "contract.csv", index=False)

    # 2. checklist.csv
    checklist_content = {
        "Risk_feature": ["  RF1 ", "rf TWO  ", None, "RfThRee"],
        "Identified_items": ["Item A  ", "  Item B with   tab\tinside", None, "Item D"]
    }
    pd.DataFrame(checklist_content).to_csv(raw_dir / "checklist.csv", index=False)

    # 3. clause_review_pair.csv
    expert_content = {
        "checkpoint": ["Expert CP1  ", "  expert cp two", None],
        "clause": ["Expert Clause A  ", "  Expert Clause B", None],
        "review": ["Good Review  ", None, "  Another Review  "]
    }
    pd.DataFrame(expert_content).to_csv(raw_dir / "clause_review_pair.csv", index=False)

    # 4. GT_contract_risk.csv
    gt_content = {
        "checkpoint": ["GT CP1  ", "  gt cp two", None],
        "clause": ["GT Clause A  ", "  GT Clause B", "Clause for NaN GT"],
        "gt": ["  a ", " B  ", None]
    }
    pd.DataFrame(gt_content).to_csv(raw_dir / "GT_contract_risk.csv", index=False)
    
    return raw_dir, processed_dir


# --- 测试主预处理函数 ---
def test_preprocess_all_data_creates_files_and_processed_dir(create_temp_raw_data, monkeypatch):
    """
    测试 preprocess_all_data 是否创建了 processed 目录和所有预期的处理后文件。
    """
    temp_raw_dir, temp_processed_dir = create_temp_raw_data
    
    # 使用 monkeypatch 来临时修改 config.py 中的路径变量，使其指向我们的临时目录
    monkeypatch.setattr(app_config, 'RAW_DATA_PATH', str(temp_raw_dir))
    monkeypatch.setattr(app_config, 'PROCESSED_DATA_PATH', str(temp_processed_dir))

    # 运行预处理
    preprocess_all_data()

    assert temp_processed_dir.exists(), "处理后数据的目录应该被创建"
    assert temp_processed_dir.is_dir()

    expected_files = [
        "processed_contract.csv",
        "processed_checklist.csv",
        "processed_clause_review_pair.csv",
        "processed_GT_contract_risk.csv"
    ]
    for f_name in expected_files:
        assert (temp_processed_dir / f_name).exists(), f"处理后的文件 {f_name} 未找到"
        assert (temp_processed_dir / f_name).is_file()

# --- 测试具体文件的预处理正确性 ---
def test_contract_csv_content_after_preprocessing(create_temp_raw_data, monkeypatch):
    temp_raw_dir, temp_processed_dir = create_temp_raw_data
    monkeypatch.setattr(app_config, 'RAW_DATA_PATH', str(temp_raw_dir))
    monkeypatch.setattr(app_config, 'PROCESSED_DATA_PATH', str(temp_processed_dir))

    preprocess_all_data() # 运行完整的预处理

    df_processed = pd.read_csv(temp_processed_dir / "processed_contract.csv").fillna('') # 读取时也fillna以进行比较

    # 验证 'checkpoint' 列: 转小写, 去首尾空格, NaN -> ''
    expected_checkpoints = ["cp1", "cp two", "", "cpthree"]
    assert df_processed["checkpoint"].tolist() == expected_checkpoints, \
        f"Contract Checkpoint 不匹配: 期望 {expected_checkpoints}, 得到 {df_processed['checkpoint'].tolist()}"

    # 验证 'clause' 列: 去首尾空格, NaN -> ''
    # 注意：.str.strip() 只处理首尾空格，不处理中间的多个空格或换行符
    expected_clauses = ["Clause  A", "Clause B with   extra spaces", "Clause C\nwith newline", ""]
    assert df_processed["clause"].tolist() == expected_clauses, \
        f"Contract Clause 不匹配: 期望 {expected_clauses}, 得到 {df_processed['clause'].tolist()}"


def test_checklist_csv_content_after_preprocessing(create_temp_raw_data, monkeypatch):
    temp_raw_dir, temp_processed_dir = create_temp_raw_data
    monkeypatch.setattr(app_config, 'RAW_DATA_PATH', str(temp_raw_dir))
    monkeypatch.setattr(app_config, 'PROCESSED_DATA_PATH', str(temp_processed_dir))

    preprocess_all_data()

    df_processed = pd.read_csv(temp_processed_dir / "processed_checklist.csv").fillna('')

    # 验证 'Risk_feature' 列
    expected_risk_features = ["rf1", "rf two", "", "rfthree"]
    assert df_processed["Risk_feature"].tolist() == expected_risk_features

    # 验证 'Identified_items' 列
    expected_items = ["Item A", "Item B with   tab\tinside", "", "Item D"]
    assert df_processed["Identified_items"].tolist() == expected_items


def test_gt_contract_risk_csv_gt_column_cleaning(create_temp_raw_data, monkeypatch):
    temp_raw_dir, temp_processed_dir = create_temp_raw_data
    monkeypatch.setattr(app_config, 'RAW_DATA_PATH', str(temp_raw_dir))
    monkeypatch.setattr(app_config, 'PROCESSED_DATA_PATH', str(temp_processed_dir))

    preprocess_all_data()

    df_processed = pd.read_csv(temp_processed_dir / "processed_GT_contract_risk.csv").fillna('')
    
    # 验证 'gt' 列: 去首尾空格, 转大写, NaN -> ''
    expected_gt_values = ["A", "B", ""]
    assert df_processed["gt"].tolist() == expected_gt_values


# 可以为 _clean_and_normalize_df 编写更直接的单元测试
def test_clean_and_normalize_df_logic():
    sample_data = {
        "key_col": ["  Key1 ", "key TWO  ", None, "KEYThree"],
        "text_col": ["Text  A  ", "  Text B  ", "Text\tC", None],
        "other_col": [1, 2, 3, 4]
    }
    df = pd.DataFrame(sample_data)
    
    processed = _clean_and_normalize_df(
        df, 
        text_cols_to_strip_fillna=['text_col', 'non_existent_text_col'],
        key_cols_to_lower_strip_fillna=['key_col', 'non_existent_key_col']
    )
    
    expected_key_col = ["key1", "key two", "", "keythree"]
    assert processed["key_col"].tolist() == expected_key_col
    
    expected_text_col = ["Text  A", "Text B", "Text\tC", ""]
    assert processed["text_col"].tolist() == expected_text_col
    
    assert "other_col" in processed.columns # 其他列应保留