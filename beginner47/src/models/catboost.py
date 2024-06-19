from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class CatBoost_Trainer:
    def __init__(self, params, data, num_round=100, early_stopping_rounds=10, test_size=0.2):
        self.params = params
        self.data = data
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        # データセットのトレーニングセットとテストセットへの分割
        X = data.drop(columns=["target"])
        y = data["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.model = CatBoostClassifier(**self.params, iterations=self.num_round, early_stopping_rounds=self.early_stopping_rounds)

    def train(self):
        train_pool = Pool(self.X_train, label=self.y_train)
        eval_pool = Pool(self.X_test, label=self.y_test)
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
        return pred
