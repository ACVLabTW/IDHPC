import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from util import data_process

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dp1 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dp1(x)
        x = self.fc3(x)
        return x


def trainer(
    model: nn.Module,
    TrainData: DataLoader,
    ValidData: DataLoader,
    epochs: int,
    opt: optim.Optimizer,
    crit: nn.Module,
    save_path: str,
    fold_n: int,
) -> Tuple[float, Dict[str, float]]:
    best_loss = np.inf
    best_metrics: Dict[str, float] = {}
    lr_lambda = lambda epoch: 0.9 ** epoch if epoch > 5 else 1
    scheduler = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lr_lambda)

    for epoch in range(epochs):
        model.train()
        for features, labels in TrainData:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(features).squeeze(1)
            loss = crit(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

        scheduler.step()
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        model.eval()
        y_true: List[float] = []
        y_pred: List[int] = []
        y_probs: List[float] = []
        y_loss: List[float] = []

        with torch.no_grad():
            for features, labels in ValidData:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(features).squeeze(1)
                loss = crit(outputs, labels)
                probs = torch.sigmoid(outputs)
                predictions = probs > 0.5

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())
                y_probs.extend(probs.cpu().tolist())
                y_loss.append(loss.item())

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(y_true, y_probs)
        mean_loss = float(np.mean(y_loss))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_metrics = {
                "loss": best_loss,
                "accuracy": accuracy,
                "f1_macro": f1,
                "recall_macro": recall,
                "auc": auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, f"best_model_cv{fold_n}.pth"))
            print(
                f"New best @epoch {epoch+1}: loss={best_loss:.4f}, "
                f"acc={accuracy:.4f}, f1={f1:.4f}, recall={recall:.4f}, auc={auc:.4f}, "
                f"TP={tp} FP={fp} TN={tn} FN={fn}",
                flush=True,
            )

    return best_loss, best_metrics


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data_paths = {
        "A": "../dataset/DATA_A_FinalFinished[2024-09-27-filter-age]fixed.csv",
        "B": "../dataset/DATA_B_FinalFinished[2024-09-27-filter-age]fixed.csv",
        "C": "../dataset/DATA_C_FinalFinished[2024-09-27-filter-age]fixed.csv",
    }

    base_select_cols = [
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
    ]

    select_cols = base_select_cols + [
        "Max Diff mbp",
        "Max Diff sbp",
        "結束脈搏",
        "Final Judge",
        "Raw Index",
        "ID",
        "洗腎紀錄時間去時分",
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    ]

    hidden_size = 128
    epochs = 100
    batch_size = 128
    save_path = "./mlp_ckpt/nonmark"
    summary: List[Dict[str, object]] = []

    print(f"Running on device: {DEVICE}", flush=True)

    for db_type in ["A", "B", "C"]:
        print(f"\n=== DATASET {db_type} ===", flush=True)
        for fold_idx in [0, 1, 2, 3, 4]:
            print(f"-- Fold {fold_idx+1}/5", flush=True)
            if db_type == "A":
                cat_col_names = ["入院方式", "性別", "體溫", "FISTULA", "GRAFT", "Catheter", "有無糖尿病", "Intact PTH", "HCV", "HBV"]
            else:
                cat_col_names = ["入院方式", "性別", "FISTULA", "GRAFT", "Catheter", "有無糖尿病", "HCV", "HBV"]

            DATA_PROCESS = data_process()
            train_X, train_y, _, _ = DATA_PROCESS.data_loader(data_paths[db_type], "train", fold_idx, select_cols, cat_col_names)
            val_X, val_y, _, _ = DATA_PROCESS.data_loader(data_paths[db_type], "val", fold_idx, select_cols, cat_col_names)

            X_train_tensor = torch.tensor(train_X.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(train_y.values, dtype=torch.float32)
            X_val_tensor = torch.tensor(val_X.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(val_y.values, dtype=torch.float32)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

            input_size = train_X.shape[1]
            model = MLP(input_size, hidden_size, output_size=1).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_loss, best_metrics = trainer(
                model=model,
                TrainData=train_loader,
                ValidData=valid_loader,
                epochs=epochs,
                opt=optimizer,
                crit=criterion,
                save_path=f"{save_path}_{db_type}",
                fold_n=fold_idx,
            )

            summary.append(
                {
                    "dataset": db_type,
                    "fold": fold_idx,
                    "input_size": input_size,
                    "best_loss": best_loss,
                    **best_metrics,
                }
            )

    os.makedirs(save_path, exist_ok=True)
    summary_path = os.path.join(save_path, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFinished. Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
