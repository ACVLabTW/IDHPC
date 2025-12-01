"""
以時間切分（Train:2016-2018, Val:2019, Test:2020）重跑表格模型：
- 模型：簡單 MLP（與 notebook 架構對齊）。
- 資料集：A/B/C（對應不同前處理策略）。
- 多個 random seed 產生重複實驗，並以 Wilcoxon 檢定比較 A/B/C。
輸出：
  - results/tabular_timesplit_metrics.json
  - results/tabular_wilcoxon.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


DATASETS = ["A", "B", "C"]
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed" / "tabular"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

TARGET = "Final Judge"
SEEDS = [0, 1, 2]  # 保留多次重複以做檢定，避免運算時間過長
EPOCHS = 10
BATCH_SIZE = 256
HIDDEN_SIZE = 128


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_cat_cols(dataset: str) -> List[str]:
    if dataset == "A":
        return ["入院方式", "性別", "體溫", "FISTULA", "GRAFT", "Catheter", "有無糖尿病", "Intact PTH", "HCV", "HBV"]
    return ["入院方式", "性別", "FISTULA", "GRAFT", "Catheter", "有無糖尿病", "HCV", "HBV"]


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = PROCESSED / dataset / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing split: {path}")
    return pd.read_csv(path)


def normalize(train_df: pd.DataFrame, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    mean = train_df[num_cols].mean()
    std = train_df[num_cols].std().replace(0, 1)
    df[num_cols] = (df[num_cols] - mean) / std
    return df


def prep_tensors(dataset: str) -> Tuple[DataLoader, DataLoader, pd.Series, pd.DataFrame]:
    cat_cols = get_cat_cols(dataset)
    num_cols = [c for c in BASE_FEATURES if c not in cat_cols]

    train_df = load_split(dataset, "train").copy()
    val_df = load_split(dataset, "val").copy()
    test_df = load_split(dataset, "test").copy()

    # 填補缺失
    for col in num_cols:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        val_df[col] = val_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)
    for col in cat_cols:
        mode = train_df[col].mode(dropna=True)
        fill_val = mode.iloc[0] if not mode.empty else -1
        train_df[col] = train_df[col].fillna(fill_val).astype(int)
        val_df[col] = val_df[col].fillna(fill_val).astype(int)
        test_df[col] = test_df[col].fillna(fill_val).astype(int)

    train_df = normalize(train_df, train_df, num_cols)
    val_df = normalize(train_df, val_df, num_cols)
    test_df = normalize(train_df, test_df, num_cols)

    feature_cols = num_cols + cat_cols
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[TARGET].values, dtype=torch.float32)
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[TARGET].values, dtype=torch.float32)
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[TARGET].values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_data = (X_test, y_test)
    return train_loader, val_loader, test_data, test_df[feature_cols]


def train_one(dataset: str, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_data, _ = prep_tensors(dataset)
    input_size = train_loader.dataset.tensors[0].shape[1]
    model = MLP(input_size, HIDDEN_SIZE).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    for epoch in range(EPOCHS):
        model.train()
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 驗證
        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(features).squeeze(1)
                val_losses.append(criterion(outputs, labels).item())
        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 評估 test
    model.load_state_dict(best_state)
    model.eval()
    X_test, y_test = test_data
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    with torch.no_grad():
        logits = model(X_test).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    y_true = y_test.cpu().numpy()
    y_prob = probs.cpu().numpy()
    y_pred = preds.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "dataset": dataset,
        "seed": seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def run_all() -> Dict[str, List[Dict[str, float]]]:
    all_metrics: Dict[str, List[Dict[str, float]]] = {d: [] for d in DATASETS}
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"Start training dataset={dataset}, seed={seed}", flush=True)
            m = train_one(dataset, seed)
            print(f"[{dataset}] seed={seed} auc={m['auc']:.4f} ap={m['average_precision']:.4f}")
            all_metrics[dataset].append(m)
    return all_metrics


def wilcoxon_tests(all_metrics: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    def extract(metric: str, dataset: str) -> List[float]:
        return [m[metric] for m in all_metrics[dataset]]

    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    metrics = ["auc", "average_precision"]
    results: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        for a, b in pairs:
            stat, p = wilcoxon(extract(m, a), extract(m, b))
            key = f"{a}_vs_{b}_{m}"
            results[key] = {"statistic": float(stat), "p_value": float(p)}
    return results


def main():
    print(f"Device: {DEVICE}")
    all_metrics = run_all()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULT_DIR / "tabular_timesplit_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    w_results = wilcoxon_tests(all_metrics)
    w_path = RESULT_DIR / "tabular_wilcoxon.json"
    w_path.write_text(json.dumps(w_results, ensure_ascii=False, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Wilcoxon saved to {w_path}")


if __name__ == "__main__":
    main()
