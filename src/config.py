"""Пути, сиды, веса первой стадии, параметры ранкера."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AVITO_DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("AVITO_CACHE_DIR", ROOT / "cache"))

# https://huggingface.co/BAAI/bge-m3, можно передать локальный путь через BGE_M3_PATH
BGE_M3 = os.environ.get("BGE_M3_PATH", "BAAI/bge-m3")
BGE_MAX_LEN = 64
BGE_BATCH = 256

SEED = 42

# отложенные выборки из train, см. README
N_VAL, VAL_SEED = 2000, 0      # оценка
N_TRN, TRN_SEED = 6000, 1      # обучение ранкера, тексты не пересекаются с val

TOP_K = 50
K_POOL = 150                   # кандидатов первой стадии на ранкер
KNN_TEXTS = 20                 # похожих запросов train для приоров

# веса линейной первой стадии, подобраны перебором на валидации
LINEAR_WEIGHTS = {
    "dq": 1.0,      # косинус BGE-M3 запрос / объявление
    "dqp": 0.25,    # то же, запрос вместе с фильтрами
    "lex": 0.75,    # TF-IDF
    "same": 0.3,    # та же локация
    "trans": 0.02,  # log P(локация объявления | локация поиска)
    "geo": 0.05,    # -log(1 + dist/10км)
    "mc05": 0.01,   # log P(микрокатегория | похожие запросы), tau=0.05
    "mc02": 0.0,    # tau=0.02, только для ранкера
    "mcp": 0.02,    # log P(микрокатегория | фильтры)
    "hist": 0.1,    # клики по объявлению с тем же текстом запроса
    "cat": 0.1,     # категория 114
    "rev": 0.01,    # log(1 + отзывы)
    "dclick": 0.5,  # близость к объявлениям, которые выбирали по похожим запросам
}

USE_RANKER = True
LGB_ROUNDS = 200
LGB_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[50], learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambdarank_truncation_level=60, verbose=-1,
    seed=SEED, deterministic=True, force_row_wise=True, num_threads=8,
)
USE_CLICK = True
