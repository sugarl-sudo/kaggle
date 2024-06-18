from models.lightgbm import LigthGBM_Trainer
import pandas as pd


def main():
    data_path = './data/train.csv'
    # パラメータの設定
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    data = pd.read_csv(data_path)
    trainer = LigthGBM_Trainer()

if __name__ == '__main__':
    # print('Hello, world!')
    main()
