import torch
import pandas as pd
class data_process():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
        self.target = 'Final Judge'#Final Judge、體重實際脫水
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
    
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
    
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
    
    def data_loader(self, DB_URL:str, flag:str, fold_:int, select_cols:list ,cat_col_names:list):
        df_raw = pd.read_csv(DB_URL)
        df_raw = df_raw[select_cols]
        if flag=='test':
            df_raw = df_raw
        else:
            df_raw = df_raw[df_raw[f'fold_{fold_}'].str.contains(flag, case=False)]
            fold_cols_search = [col for col in df_raw.columns.tolist() if 'fold' in col]
        
            df_raw = df_raw.drop(columns=fold_cols_search)
        df_raw = df_raw.reset_index(drop=True)
        df_raw = df_raw.drop(columns=['Raw Index', 'ID', '洗腎紀錄時間去時分','Group Type'], errors='ignore')
        features_cat = cat_col_names
        features_num = df_raw.columns.drop(features_cat+[self.target]).to_list()
        features = features_num + features_cat
        df_raw = df_raw[features+[self.target]]
        # print(df_raw.head(12))
        # 將9999異常值轉會為最後一個類別分類 e.g:[1,2,3,9999]=>[1,2,3,4]
        for featrue_ in features_cat:
            df_raw[featrue_].replace(9999, sorted(df_raw[featrue_].unique())[-2]+1, inplace=True)
        # 把target的-9999改成-1
        df_raw[self.target].replace(-9999, -1, inplace=True)
        df_raw[features_cat] = df_raw[features_cat].astype(int)
        data_y = df_raw[self.target]
        df_data = df_raw.drop(columns=[self.target])
        del df_raw
        if flag=='train':
            self.fit(df_data.values)
            df_data[features] = self.transform(df_data.values)
        else:
            df_data[features] = self.transform(df_data.values)
        return df_data, data_y, features_cat, features_num
DATA_PROCESS = data_process()
