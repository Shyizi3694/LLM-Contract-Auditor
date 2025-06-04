# tests/data_handling/test_data_loader.py
import pytest
import pandas as pd
import os
from pathlib import Path # 用于 pytest 的 tmp_path fixture
import logging # 用于 caplog fixture

# 假设 DataLoader 和默认的 RAW_DATA_PATH 可以这样导入
from src.data_handling.data_loader import DataLoader
from src.common.config import RAW_DATA_PATH as ACTUAL_RAW_DATA_PATH # 用于对比  或部分测试

# --- Test Fixtures (可选，但有助于代码复用) ---
@pytest.fixture
def real_data_loader() -> DataLoader:
    """提供一个使用实际RAW_DATA_PATH配置的DataLoader实例。"""
    # 注意：这个fixture依赖于config.py中RAW_DATA_PATH的正确配置和实际数据文件的存在
    try:
        return DataLoader()
    except ValueError as e: # 如果RAW_DATA_PATH未配置，DataLoader的__init__会抛出ValueError
        pytest.skip(f"跳过 real_data_loader 测试，因为 RAW_DATA_PATH 配置错误: {e}")
    except FileNotFoundError as e: # 如果DataLoader在初始化时因为找不到文件而失败 (虽然当前设计是返回空DF)
         pytest.skip(f"跳过 real_data_loader 测试，因为某些数据文件未找到: {e}")
    return None # Should not be reached if skipped


# --- 测试用例 ---
def test_dataloader_initialization_with_real_data(real_data_loader: DataLoader | None):
    """测试DataLoader能否使用实际路径成功初始化并加载数据。"""
    if real_data_loader is None: # 如果fixture跳过了
        return

    assert real_data_loader.get_contracts_df() is not None
    assert real_data_loader.get_checklist_df() is not None
    assert real_data_loader.get_expert_pairs_df() is not None
    assert real_data_loader.get_ground_truth_df() is not None
    
    # 进一步检查，如果实际文件存在，DataFrame不应为空
    # (这里假设测试环境中实际数据文件是存在的且非空)
    contract_file = os.path.join(ACTUAL_RAW_DATA_PATH, "contract.csv")
    if os.path.exists(contract_file) and os.path.getsize(contract_file) > 0: # 检查文件存在且非空
        assert not real_data_loader.get_contracts_df().empty, "实际合同数据文件存在且非空时不应加载为空DF"
        assert 'checkpoint' in real_data_loader.get_contracts_df().columns
        assert 'clause' in real_data_loader.get_contracts_df().columns

    # 可以为其他DataFrame添加类似的检查


def test_load_specific_file_successfully(tmp_path: Path):
    """测试加载单个特定类型的文件是否成功（使用临时路径和文件）。"""
    temp_data_dir = tmp_path / "raw_test_data"
    temp_data_dir.mkdir()
    
    # 创建一个临时的 contract.csv 文件
    sample_contract_data = {"checkpoint": ["cp1"], "clause": ["cl1"]}
    pd.DataFrame(sample_contract_data).to_csv(temp_data_dir / "contract.csv", index=False)

    # 使用临时路径初始化DataLoader
    loader = DataLoader(raw_data_path_override=str(temp_data_dir))
    
    df_contracts = loader.get_contracts_df()
    assert df_contracts is not None
    assert not df_contracts.empty
    assert len(df_contracts) == 1
    assert df_contracts.iloc[0]['checkpoint'] == "cp1"
    assert 'clause' in df_contracts.columns

    # 测试checklist，此时因为临时目录中没有checklist.csv，应加载为空
    df_checklist = loader.get_checklist_df()
    assert df_checklist is not None
    assert df_checklist.empty, "Checklist应为空，因为它不在临时目录中"


def test_load_file_not_found(tmp_path: Path, caplog):
    """测试当指定的数据文件不存在时，DataLoader的行为。"""
    non_existent_dir = tmp_path / "this_dir_does_not_exist"
    # 我们不创建这个目录，DataLoader会尝试从这里加载

    loader = DataLoader(raw_data_path_override=str(non_existent_dir))
    
    with caplog.at_level(logging.ERROR): # 捕获ERROR级别的日志
        df_contracts = loader.get_contracts_df() # 会尝试加载 non_existent_dir/contract.csv

    assert df_contracts is not None, "即使文件未找到，DataFrame也应被实例化（可能为空）"
    assert df_contracts.empty, "当数据文件未找到时，返回的DataFrame应该为空"
    assert any(f"数据文件未找到: {os.path.join(str(non_existent_dir), 'contract.csv')}" in record.message for record in caplog.records), \
           "应记录文件未找到的错误日志"


def test_load_empty_csv_file(tmp_path: Path, caplog):
    """测试当CSV文件为空（或只有表头）时，DataLoader的行为。"""
    temp_data_dir = tmp_path / "empty_csv_test_data"
    temp_data_dir.mkdir()
    
    # 创建一个空的 checklist.csv (只有表头)
    empty_checklist_file = temp_data_dir / "checklist.csv"
    pd.DataFrame(columns=['Risk_feature', 'Identified_items']).to_csv(empty_checklist_file, index=False)

    loader = DataLoader(raw_data_path_override=str(temp_data_dir))

    with caplog.at_level(logging.WARNING): # 捕获WARNING级别的日志
        df_checklist = loader.get_checklist_df()

    assert df_checklist is not None
    assert df_checklist.empty, "当CSV文件为空时，返回的DataFrame应该为空"
    assert any(f"数据文件 'checklist.csv' 为空" in record.message for record in caplog.records), \
           "应记录文件为空的警告日志"

def test_load_csv_with_missing_required_columns(tmp_path: Path, caplog):
    """测试当CSV文件缺少必需列时的行为。"""
    temp_data_dir = tmp_path / "missing_cols_data"
    temp_data_dir.mkdir()
    
    # 创建一个缺少 'clause' 列的 contract.csv
    faulty_contract_data = {"checkpoint": ["cp1", "cp2"]} # 缺少 'clause'
    pd.DataFrame(faulty_contract_data).to_csv(temp_data_dir / "contract.csv", index=False)

    loader = DataLoader(raw_data_path_override=str(temp_data_dir))
    
    with caplog.at_level(logging.ERROR):
        df_contracts = loader.get_contracts_df() # required_columns=['checkpoint', 'clause']

    # 当前_load_csv设计在列缺失时仍返回df，但记录错误
    assert df_contracts is not None 
    # assert not df_contracts.empty # 实际上还是加载了checkpoint列
    assert 'clause' not in df_contracts.columns
    assert any("缺少必需的列: clause" in record.message for record in caplog.records), \
           "应记录缺少列的错误日志"