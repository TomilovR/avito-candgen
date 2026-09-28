"""Вторая стадия: LightGBM lambdarank по пулу первой стадии."""

import lightgbm as lgb
import numpy as np

from . import config


def labels(pool_idx, truth_idx):
    return np.array([[int(j in t) for j in row] for row, t in zip(pool_idx, truth_idx)], np.int32)


def fit(X_list, y_list, feats, rounds=config.LGB_ROUNDS):
    # запросы без правильного ответа в пуле ничего не дают ранкеру, выкидываем
    Xs, ys = [], []
    for X, y in zip(X_list, y_list):
        keep = y.sum(1) > 0
        Xs.append(X[keep]); ys.append(y[keep])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    Q, K, F = X.shape
    ds = lgb.Dataset(X.reshape(-1, F), y.reshape(-1), group=[K] * Q, feature_name=feats)
    return lgb.train(config.LGB_PARAMS, ds, num_boost_round=rounds)


def rerank(model, pool_idx, X, k=config.TOP_K):
    Q, K, F = X.shape
    p = model.predict(X.reshape(-1, F), num_threads=config.LGB_PARAMS["num_threads"]).reshape(Q, K)
    order = np.argsort(-p, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(pool_idx, order, 1)
