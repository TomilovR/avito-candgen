"""
python run.py -> answer.csv, метрики валидации пишутся в reports/validation.json

1. данные, отложенные выборки из train (val для оценки, trn для обучения ранкера)
2. эмбеддинги BGE-M3, TF-IDF, гео
3. признаки запросов для trn / val / test
4. линейный скор по всему каталогу -> top-150
5. LightGBM lambdarank -> top-50
"""

import argparse
import json
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from src import config, data, features, metrics, ranker, scoring, submission
from src.encoders import Encoder


def set_determinism():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_side(ctx, tag, qdf, trf):
    side = features.build_side(ctx, tag, qdf, trf, config.USE_CLICK)
    sc = scoring.Scorer(ctx, side, config.USE_CLICK)
    idx, X, feats = sc.pool(config.LINEAR_WEIGHTS)
    del sc
    torch.cuda.empty_cache()
    log(f"[{tag}] {len(qdf)} запросов, пул {idx.shape[1]}, признаков {X.shape[2]}")
    return {"idx": idx, "X": X, "feats": feats, "seen": side["seen"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.ROOT / "answer.csv"))
    args = ap.parse_args()
    set_determinism()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("1. Загрузка данных")
    items, queries, train = data.load()
    val = data.make_holdout(train, items, config.N_VAL, config.VAL_SEED)
    trn = data.make_holdout(train, items, config.N_TRN, config.TRN_SEED, exclude_texts=val["qn"])
    item_pos = {iid: i for i, iid in enumerate(items["item_id"])}
    test_seen_share = float(queries["qn"].isin(set(train["qn"])).mean())
    log(f"   каталог {len(items):,}, бенчмарк {len(queries):,}, train {len(train):,}; "
        f"val {len(val)}, trn {len(trn)}; доля бенчмарка с текстом из train {test_seen_share:.3f}")

    log("2. Контекст: эмбеддинги, TF-IDF, гео")
    ctx = features.build_context(items, train, Encoder(), config.USE_CLICK)

    log("3-4. Признаки сторон и пулы первой стадии")
    sides = {}
    for tag, qdf in (("trn", trn), ("val", val)):
        sides[tag] = run_side(ctx, tag, qdf, train[~data.heldout_mask(train, qdf)])
        sides[tag]["truth"] = [[item_pos[i] for i in t] for t in qdf["truth"]]
        sides[tag]["y"] = ranker.labels(sides[tag]["idx"], sides[tag]["truth"])
    sides["test"] = run_side(ctx, "test", queries, train)
    feats = sides["test"]["feats"]

    log("5a. Валидация (ранкер обучен только на trn)")
    v = sides["val"]
    m_lin = metrics.recall_at(v["idx"][:, :config.TOP_K], v["truth"], v["seen"], test_seen_share)
    m_pool = metrics.recall_at(v["idx"], v["truth"], v["seen"], test_seen_share)
    log(f"   линейная первая стадия @50: {metrics.fmt(m_lin)}")
    log(f"   покрытие пула @{config.K_POOL}: {metrics.fmt(m_pool)}")
    report = {"test_seen_share": test_seen_share, "linear_at50": m_lin, "pool": m_pool}
    if config.USE_RANKER:
        mdl = ranker.fit([sides["trn"]["X"]], [sides["trn"]["y"]], feats)
        m_rk = metrics.recall_at(ranker.rerank(mdl, v["idx"], v["X"]), v["truth"], v["seen"], test_seen_share)
        log(f"   ранкер @50: {metrics.fmt(m_rk)}")
        report["ranker_at50"] = m_rk
        imp = sorted(zip(feats, mdl.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1])
        report["feature_importance_gain"] = imp

    log("5b. Финальный ответ")
    te = sides["test"]
    if config.USE_RANKER:   # финальный ранкер учится на trn + val
        mdl = ranker.fit([sides["trn"]["X"], sides["val"]["X"]], [sides["trn"]["y"], sides["val"]["y"]], feats)
        top = ranker.rerank(mdl, te["idx"], te["X"])
    else:
        top = te["idx"][:, :config.TOP_K]
    ans = submission.build_answers(top, items, queries, train)
    submission.save_and_check(ans, args.out, items, queries)
    # ответ без ранкера, для сравнения
    (config.ROOT / "reports").mkdir(exist_ok=True)
    lin = submission.build_answers(te["idx"][:, :config.TOP_K], items, queries, train)
    submission.save_and_check(lin, config.ROOT / "reports" / "answer_linear.csv", items, queries)

    with open(config.ROOT / "reports" / "validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"Готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
