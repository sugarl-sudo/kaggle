# import lightgbm as lgb
import optuna.integration.lightgbm as lgb
import numpy as np

# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class LigthGBM_Trainer:
    def __init__(self, params, data, num_round=100, early_stopping_rounds=10, test_size=0.2):
        self.params = params
        self.data = data
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        # データセットのトレーニングセットとテストセットへの分割
        X = data.drop(columns=["target"])
        # print('X', X)
        # X = X.drop(columns=["id"])
        y = data["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    def train(self):
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        test_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)
        # print('train_data', train_data)
        self.bst = lgb.train(self.params, train_data, self.num_round, valid_sets=[test_data])

    def evaluation(self):
        # 予測
        y_pred = self.bst.predict(self.X_test, num_iteration=self.bst.best_iteration)
        # 予測確率をクラスに変換
        y_pred_class = [np.argmax(x) for x in y_pred]

        # 評価
        accuracy = accuracy_score(self.y_test, y_pred_class)
        conf_matrix = confusion_matrix(self.y_test, y_pred_class)
        class_report = classification_report(self.y_test, y_pred_class)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix: {conf_matrix}")
        print(f"Classification Report: {class_report}")

    def predict(self, X):
        # X = X.drop(columns=["id"])
        pred = self.bst.predict(X, num_iteration=self.bst.best_iteration)
        pred = [np.argmax(x) for x in pred]
        return pred
