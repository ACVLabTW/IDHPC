"""
以時間序列（GRU, window=8）驗證模型架構影響：
- 使用時間切分資料（Train:2016-2018, Val:2019, Test:2020）。
- 資料集：預設 Dataset B（mask 策略），亦可指定 A/C。
- 特徵沿用表格欄位，依患者時間排序後產生滑動視窗序列。
- 輸出 metrics 至 results/sequence_gru_metrics.json。
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed" / "tabular"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 8
SEEDS = [0, 1, 2]
BATCH_SIZE = 64
EPOCHS = 8
HIDDEN = 64

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


def get_cat_cols(dataset: str) -> List[str]:
    if dataset == "A":
        return ["入院方式", "性別", "體溫", "FISTULA", "GRAFT", "Catheter", "有無糖尿病", "Intact PTH", "HCV", "HBV"]
    return ["入院方式", "性別", "FISTULA", "GRAFT", "Catheter", "有無糖尿病", "HCV", "HBV"]


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = PROCESSED / dataset / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["洗腎紀錄時間"])
    return df.sort_values(["ID", "洗腎紀錄時間"])


def build_sequences(
    df: pd.DataFrame, dataset: str, num_mean: pd.Series = None, num_std: pd.Series = None
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    cat_cols = get_cat_cols(dataset)
    num_cols = [c for c in BASE_FEATURES if c not in cat_cols]
    feature_cols = num_cols + cat_cols

    # 填補缺失
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        fill_val = mode.iloc[0] if not mode.empty else -1
        df[col] = df[col].fillna(fill_val).astype(int)

    if num_mean is None:
        num_mean = df[num_cols].mean()
    if num_std is None:
        num_std = df[num_cols].std().replace(0, 1)
    df[num_cols] = (df[num_cols] - num_mean) / num_std

    sequences: List[np.ndarray] = []
    labels: List[float] = []
    for _, grp in df.groupby("ID"):
        if len(grp) < WINDOW:
            continue
        values = grp[feature_cols].values
        y_vals = grp[TARGET].values
        for i in range(WINDOW, len(grp) + 1):
            seq = values[i - WINDOW : i]
            label = y_vals[i - 1]
            sequences.append(seq)
            labels.append(label)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        num_mean,
        num_std,
    )


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRUModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)


def train_one(dataset: str, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = load_split(dataset, "train")
    val_df = load_split(dataset, "val")
    test_df = load_split(dataset, "test")

    X_train, y_train, num_mean, num_std = build_sequences(train_df, dataset)
    X_val, y_val, _, _ = build_sequences(val_df, dataset, num_mean=num_mean, num_std=num_std)
    X_test, y_test, _, _ = build_sequences(test_df, dataset, num_mean=num_mean, num_std=num_std)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError(f"insufficient sequences for dataset {dataset}")
    print(f"[GRU] dataset={dataset}, seed={seed}, train_seq={len(X_train)}, val_seq={len(X_val)}, test_seq={len(X_test)}", flush=True)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUModel(input_dim=X_train.shape[-1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for feats, labels in train_loader:
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(feats)
                val_losses.append(criterion(logits, labels).item())
        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no model state saved")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(DEVICE)
        y_test_t = torch.from_numpy(y_test).to(DEVICE)
        logits = model(X_test_t)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    y_true = y_test_t.cpu().numpy()
    y_prob = probs.cpu().numpy()
    y_pred = preds.cpu().numpy()

    metrics = {
        "dataset": dataset,
        "seed": seed,
        "window": WINDOW,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }
    return metrics


def main(dataset: str = "B"):
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset}, window={WINDOW}")
    all_metrics: List[Dict[str, float]] = []
    for seed in SEEDS:
        print(f"Start training seed={seed}", flush=True)
        m = train_one(dataset, seed)
        print(f"[GRU] seed={seed} auc={m['auc']:.4f} ap={m['average_precision']:.4f}")
        all_metrics.append(m)
    out = RESULT_DIR / "sequence_gru_metrics.json"
    out.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
