from models.lightgbm import LigthGBM_Trainer
from models.catboost import CatBoost_Trainer
import pandas as pd
import click


@click.command()
@click.option("--train_path", type=str, default="../data/train.csv")
@click.option("--test_path", type=str, default="../data/test.csv")
@click.option("--config_path", type=str, default="configs/config.yaml")
@click.option("--model", type=str, default="lightgbm")
@click.option("--save_dir", type=str, default="../results")
def main(train_path, test_path, model, save_path):
    # パラメータの設定
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }
    train_data = pd.read_csv(train_path)
    # train_data = train_data.drop(columns=["id"])
    train_data["sex"] = train_data["sex"].astype("category")
    train_data["smoker"] = train_data["smoker"].astype("category")
    train_data["region"] = train_data["region"].astype("category")
    train_data["target"] = train_data["charges"]
    train_data = train_data.drop(columns=["charges", "id"])
    print("train_data", train_data)

    test_data = pd.read_csv(test_path)
    test_data["sex"] = test_data["sex"].astype("category")
    test_data["smoker"] = test_data["smoker"].astype("category")
    test_data["region"] = test_data["region"].astype("category")
    print("test_data", test_data)
    if model == "lightgbm":
        trainer = LigthGBM_Trainer(params, train_data)
    elif model == "catboost":
        trainer = CatBoost_Trainer(params, train_data)
    trainer.train()
    trainer.evaluation()
    pred = trainer.predict(test_data.drop(columns=["id"]))
    # print('pred', pred)
    submission = pd.DataFrame({"id": test_data["id"], "charges": pred})
    submission.to_csv("./data/submission.csv", index=False, header=False)


if __name__ == "__main__":
    # print('Hello, world!')
    main()
