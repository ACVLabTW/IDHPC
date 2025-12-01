# Data Guide (No Raw Data Included)

本壓縮包不含任何原始/處理後資料檔。請依下列說明放置對應檔案與欄位。

## 目錄約定
- `dataset/`：放置表格版原始檔  
  - `DATA_A_FinalFinished[2024-09-27-filter-age]fixed.csv`（臨床修正/填補）  
  - `DATA_B_FinalFinished[2024-09-27-filter-age]fixed.csv`（mask=-999）  
  - `DATA_C_FinalFinished[2024-09-27-filter-age]fixed.csv`（嚴格篩選）
- `processed/tabular/{A,B,C}/train|val|test.csv`：由 `utils/prepare_tabular_splits.py` 依年份切分產生（Train=2016-2018，Val=2019，Test=2020）。
- 時序資料若需另存，可重用上列欄位並以病人+時間排序後滑動視窗產生序列。

## 欄位需求（表格與時序共用）
- 目標：`Final Judge`
- 類別欄位（A）：`入院方式`, `性別`, `體溫`, `FISTULA`, `GRAFT`, `Catheter`, `有無糖尿病`, `Intact PTH`, `HCV`, `HBV`
- 類別欄位（B/C）：`入院方式`, `性別`, `FISTULA`, `GRAFT`, `Catheter`, `有無糖尿病`, `HCV`, `HBV`
- 連續欄位（共用）：`age`, `體重1開始`, `開始血壓SBP`, `開始血壓DBP`, `開始脈搏`, `體溫`, `體重實際脫水`, `每公斤脫水量(ml/kg)`, `BUN`, `K`, `HGB`, `URR%`, `Na`, `Ca`, `P`, `透析液 Ca`, `ALBUMIN`, `ALT (SGPT)`, `Alk.phosphatase`, `Ferritin`, `IRON/TIBC`, `MCV`, `MCHC`, `MCH`, `Iron`, `Glucose AC`, `RBC`, `WBC`, `Platelet`, `Creatinine`, `AST (SGOT)`, `TIBC`, `Bilirubin-T`, `Cholesterol-T`, `CRP`, `Max Diff mbp`, `Max Diff sbp`, `結束脈搏`
- 標識/時間欄：`ID`, `洗腎紀錄時間`（timestamp），可保留 `Raw Index`, `洗腎紀錄時間去時分` 供追蹤。

## 產生切分
```
python utils/prepare_tabular_splits.py
```
輸出將寫入 `processed/tabular/{A,B,C}/train|val|test.csv`。

## 執行實驗
- 表格 MLP：`python utils/run_tabular_experiments.py`（使用上述 processed 切分）
- 時序 GRU：`python utils/run_sequence_gru.py`（預設 Dataset B，window=8）

如需其他模型（Informer、CatBoost 等），請沿用同欄位與時間切分策略。***
