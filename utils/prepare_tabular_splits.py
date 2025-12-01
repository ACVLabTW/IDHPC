"""
時間切分的表格資料前處理：
- 讀取 Dataset A/B/C 原始 CSV。
- 依照年份分割：2016-2018 為 train，2019 為 val，2020 為 test（時間序避免洩漏）。
- 移除 fold 標記欄與不必要欄位，僅保留模型使用欄位與標籤。
- 輸出至 processed/tabular/<dataset>/{train,val,test}.csv。
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASETS: Dict[str, str] = {
    "A": "dataset/DATA_A_FinalFinished[2024-09-27-filter-age]fixed.csv",
    "B": "dataset/DATA_B_FinalFinished[2024-09-27-filter-age]fixed.csv",
    "C": "dataset/DATA_C_FinalFinished[2024-09-27-filter-age]fixed.csv",
}

# 模型會用到的欄位（與 MLP notebook 對齊）
BASE_FEATURES: List[str] = [
    "性別",
    "入院方式",
    "HCV",
    "HBV",
    "有無糖尿病",
    "FISTULA",
    "GRAFT",
    "Catheter",
    "Intact PTH",
    "age",
    "體重1開始",
    "開始血壓SBP",
    "開始血壓DBP",
    "開始脈搏",
    "體溫",
    "體重實際脫水",
    "每公斤脫水量(ml/kg)",
    "BUN",
    "K",
    "HGB",
    "URR%",
    "Na",
    "Ca",
    "P",
    "透析液 Ca",
    "ALBUMIN",
    "ALT (SGPT)",
    "Alk.phosphatase",
    "Ferritin",
    "IRON/TIBC",
    "MCV",
    "MCHC",
    "MCH",
    "Iron",
    "Glucose AC",
    "RBC",
    "WBC",
    "Platelet",
    "Creatinine",
    "AST (SGOT)",
    "TIBC",
    "Bilirubin-T",
    "Cholesterol-T",
    "CRP",
    "Max Diff mbp",
    "Max Diff sbp",
    "結束脈搏",
]

TARGET_COL = "Final Judge"
EXTRA_COLS = ["Raw Index", "ID", "洗腎紀錄時間去時分"]
FOLD_COLS = [f"fold_{i}" for i in range(5)]


def prepare_dataset(name: str, src: Path, out_dir: Path) -> None:
    df = pd.read_csv(src, parse_dates=["洗腎紀錄時間"])
    df = df.sort_values(["ID", "洗腎紀錄時間"])

    keep_cols = BASE_FEATURES + [TARGET_COL, "洗腎紀錄時間"] + EXTRA_COLS + FOLD_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.dropna(subset=[TARGET_COL])

    df["year"] = df["洗腎紀錄時間"].dt.year

    train_df = df[df["year"].between(2016, 2018)].drop(columns=["year"] + FOLD_COLS, errors="ignore")
    val_df = df[df["year"] == 2019].drop(columns=["year"] + FOLD_COLS, errors="ignore")
    test_df = df[df["year"] == 2020].drop(columns=["year"] + FOLD_COLS, errors="ignore")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"[{name}] train={len(train_df)}, val={len(val_df)}, test={len(test_df)} -> {out_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    out_root = root / "processed" / "tabular"
    for name, path in DATASETS.items():
        prepare_dataset(name, root / path, out_root / name)


if __name__ == "__main__":
    main()
