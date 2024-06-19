from models.lightgbm import LigthGBM_Trainer
from models.catboost import CatBoost_Trainer
from utils.utils import get_params
import pandas as pd
import numpy as np
import click
import optuna
from sklearn.model_selection import train_test_split
from catboost import Pool
import sklearn.metrics




@click.command()
@click.option("--train_path", type=str, default="./data/train.csv")
@click.option("--test_path", type=str, default="./data/test.csv")
@click.option("--config_path", type=str, default="configs/config.yaml")
@click.option("--model", type=str, default="lightgbm")
@click.option("--save_dir", type=str, default="../results")
def main(train_path, test_path, config_path, model, save_dir):
    # パラメータの設定
    params = get_params(model)
    train_data = pd.read_csv(train_path)
    train_data["target"] = train_data["charges"]
    train_data = train_data.drop(columns=["charges", "id"])
    print("train_data", train_data)

    test_data = pd.read_csv(test_path)
    print("test_data", test_data)
    if model == "lightgbm":
        trainer = LigthGBM_Trainer(params, train_data)
    elif model == "catboost":
        trainer = CatBoost_Trainer(params, train_data)
    # study = optuna.create_study()
    # study.optimize(trainer.optuna_obj, n_trials=1000)
    # print(study.best_params)
    trainer.train()
    trainer.evaluation()
    pred = trainer.predict(test_data.drop(columns=["id"]))
    # print('pred', pred)
    print("pred", pred)
    print("test_data['id']", test_data["id"])
    print("pred shape", pred.shape)
    submission = pd.DataFrame({"id": test_data["id"], "charges": pred})
    submission.to_csv(f"{save_dir}/submission.csv", index=False, header=False)


if __name__ == "__main__":
    main()
