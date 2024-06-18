import lightgbm as lgb
import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # サンプルデータセットの作成
# # 実際のデータセットに置き換えてください
# data = pd.DataFrame(
#     {"feature1": np.random.rand(150), "feature2": np.random.rand(150), "target": np.random.randint(3, size=150)}
# )

# # データセットのトレーニングセットとテストセットへの分割
# X = data.drop(columns=["target"])
# y = data["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # LightGBMデータセットの作成
# train_data = lgb.Dataset(X_train, label=y_train)
# test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# # パラメータの設定
# params = {
#     "objective": "multiclass",
#     "num_class": 3,
#     "metric": "multi_logloss",
#     "boosting_type": "gbdt",
#     "num_leaves": 31,
#     "learning_rate": 0.05,
#     "feature_fraction": 0.9,
# }

# # モデルのトレーニング
# num_round = 100
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)

# # 予測
# y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
# # 予測確率をクラスに変換
# y_pred_class = [np.argmax(x) for x in y_pred]


class LigthGBM_Trainer:
    def __init__(self, params, data, num_round=100, early_stopping_rounds=10, test_size=0.2):
        self.params = params
        self.data = data
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.x_test = data["x_test"]
        # データセットのトレーニングセットとテストセットへの分割
        X = data.drop(columns=["target"])
        y = data["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    def train(self):
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        test_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)
        self.bst = lgb.train(
            self.params, train_data, self.num_round, valid_sets=[test_data], early_stopping_rounds=self.early_stopping_rounds
        )

    def evaluation(self):
        # 予測
        y_pred = self.bst.predict(self.x_test, num_iteration=self.bst.best_iteration)
        # 予測確率をクラスに変換
        y_pred_class = [np.argmax(x) for x in y_pred]

        # 評価
        accuracy = accuracy_score(self.y_test, y_pred_class)
        conf_matrix = confusion_matrix(self.y_test, y_pred_class)
        class_report = classification_report(self.y_test, y_pred_class)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix: {conf_matrix:.4f}")
        print(f"Classification Report: {class_report:.4f}")
