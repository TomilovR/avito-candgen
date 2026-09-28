"""Загрузка данных и отложенные выборки."""

import re

import pandas as pd
import pyarrow.parquet as pq

from . import config

ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_location_id", "item_latitude",
    "item_longitude", "item_rating", "item_rating_reviews_count", "item_price",
]

# один текст с разной локацией или фильтрами считаем разными запросами
SIG = ["qn", "search_location_id", "pn", "search_category"]

RATING_RE = re.compile(r"рейтинг[^,]*?(выше|звезд\w*)", re.I)


def norm(s) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip() if isinstance(s, str) else ""


def query_with_params(q, p) -> str:
    # фильтр по рейтингу выкидываем, он ничего не говорит об услуге
    p = RATING_RE.sub(" ", p if isinstance(p, str) else "")
    p = re.sub(r"\s+", " ", p).strip()
    q = q if isinstance(q, str) else ""
    return f"{q}. {p}" if p else q


def load(data_dir=config.DATA_DIR):
    items = pq.read_table(data_dir / "benchmark_items.parquet", columns=ITEM_COLS).to_pandas()
    queries = pq.read_table(data_dir / "benchmark_queries.parquet").to_pandas()
    train = pq.read_table(data_dir / "train.parquet").to_pandas()
    for df in (queries, train):
        df["qn"] = df["search_query"].map(norm)
        df["pn"] = df["search_infm_params_text"].map(norm)
    # часть числовых колонок в parquet хранится как decimal
    for df in (items, train):
        for c in ("item_latitude", "item_longitude", "item_price", "item_rating", "item_rating_reviews_count"):
            df[c] = df[c].astype(float)
    return items, queries, train


def make_holdout(train, items, n, seed, exclude_texts=()):
    # Берём клики train, у которых объявление есть в каталоге бенчмарка (~33k строк),
    # поэтому ищем по настоящему каталогу. Один запрос на текст, как в бенчмарке.
    in_cat = train["item_id"].isin(set(items["item_id"]))
    sub = train[in_cat & (train["search_is_delivery_search"] == 0)]   # в бенчмарке доставки нет
    if len(exclude_texts):
        sub = sub[~sub["qn"].isin(set(exclude_texts))]
    g = (sub.groupby(SIG, sort=True)
         .agg(truth=("item_id", lambda s: sorted(set(s))),
              search_query=("search_query", "first"),
              search_infm_params_text=("search_infm_params_text", "first"))
         .reset_index())
    g = g.sample(frac=1.0, random_state=seed).drop_duplicates("qn").head(n)
    return g.reset_index(drop=True)


def heldout_mask(train, hold):
    # строки train с запросами из hold, их не используем при подсчёте признаков.
    # тот же текст с другой локацией остаётся (в бенчмарке 37% текстов есть в train)
    return pd.MultiIndex.from_frame(train[SIG]).isin(pd.MultiIndex.from_frame(hold[SIG]))
