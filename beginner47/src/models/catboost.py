from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import sklearn
import numpy as np


class CatBoost_Trainer:
    def __init__(self, params, data, num_round=100, early_stopping_rounds=10, test_size=0.1):
        self.params = params
        self.data = data
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        # データセットのトレーニングセットとテストセットへの分割
        X = data.drop(columns=["target"])
        y = data["target"]
        # X_train, self.X_test, y_train, self.y_test = train_test_split(
        #     X, y, test_size=test_size, random_state=42
        # )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        # SMOTEによるオーバーサンプリング
        # smote = SMOTE()
        # self.X_train, self.y_train = smote.fit_resample(X_train, y_train)
        self.cat_features = ["sex", "smoker", "region"]
        self.model = CatBoostClassifier(**self.params)

    def train(self):
        train_pool = Pool(self.X_train, label=self.y_train, cat_features=self.cat_features)
        eval_pool = Pool(self.X_test, label=self.y_test, cat_features=self.cat_features)
        self.model.fit(train_pool, eval_set=eval_pool, verbose=False)

    def evaluation(self):
        # 予測
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)

        # 評価
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix: {conf_matrix}")
        print(f"Classification Report: {class_report}")

    def predict(self, X):
        pred = self.model.predict(X)
        return pred.flatten()

    def optuna_obj(self, trial):
        train_pool = Pool(self.X_train, self.y_train, cat_features=self.cat_features)
        test_pool = Pool(self.X_test, self.y_test, cat_features=self.cat_features)
        # クラス重みの調整
        class_weight_0 = trial.suggest_loguniform('class_weight_0', 0.1, 10.0)
        class_weight_1 = trial.suggest_loguniform('class_weight_1', 0.1, 10.0)
        class_weight_2 = trial.suggest_loguniform('class_weight_2', 0.1, 10.0)
        class_weights = {0: class_weight_0, 1: class_weight_1, 2: class_weight_2}

        # パラメータの指定
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "random_strength": trial.suggest_int("random_strength", 0, 100),
            "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 100.00),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            "od_wait": trial.suggest_int("od_wait", 10, 50),
            "class_weights": class_weights,
            # "class_weights": trial.suggest_loguniform("class_weights", {0: 1.0, 1: 1.0, 2: 1.0}, {0: 0.5, 1: 5.0, 2: 5.0}, {0: 0.1, 1: 10.0, 2: 10.0})
        }

        # 学習
        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        # 予測
        preds = model.predict(test_pool)
        pred_labels = np.rint(preds)
        # 精度の計算
        accuracy = sklearn.metrics.accuracy_score(self.y_test, pred_labels)
        return 1.0 - accuracy
