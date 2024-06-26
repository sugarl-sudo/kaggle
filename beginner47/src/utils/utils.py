def get_params(model_name):
    if model_name == "lightgbm":
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }
    elif model_name == "catboost":
        # パラメータを辞書型で指定
        params = {
            'iterations': 266,
            'depth': 4,
            'learning_rate': 0.28399360855713346,
            'random_strength': 0,
            'bagging_temperature': 0.2873921593868477,
            'od_type': 'IncToDec',
            'od_wait': 42,
            'class_weights': {0: 0.7422852207066004, 1: 0.7687134075998778, 2: 0.9487452170625182},
            }
    return params
