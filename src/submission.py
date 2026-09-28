"""Сборка answer.csv и проверка формата."""

from collections import Counter, defaultdict

import pandas as pd

from .data import norm


def build_answers(top_idx, items, queries, train, k=50):
    # сначала объявления, которые выбирали в train по такому же запросу в той же локации
    # (есть у 91 запроса), потом выдача модели
    ids = items["item_id"].values
    in_cat = train[train["item_id"].isin(set(ids))]
    exact = defaultdict(Counter)
    for q, l, i in zip(in_cat["search_query"].map(norm), in_cat["search_location_id"], in_cat["item_id"]):
        exact[(q, l)][i] += 1
    rows = []
    for qi, (qid, q, l) in enumerate(zip(queries["query_id"], queries["qn"], queries["search_location_id"])):
        pref = [i for i, _ in exact.get((q, l), Counter()).most_common()]
        seen, ans = set(), []
        for iid in pref + [ids[j] for j in top_idx[qi]]:
            if iid not in seen:
                seen.add(iid); ans.append(iid)
            if len(ans) == k:
                break
        rows.append((qid, " ".join(ans)))
    return pd.DataFrame(rows, columns=["query_id", "answer"])


def save_and_check(df, path, items, queries):
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\r\n")   # одинаково на любой ОС
    chk = pd.read_csv(path, dtype=str, keep_default_na=False)
    valid = set(items["item_id"])
    assert list(chk.columns) == ["query_id", "answer"], "колонки"
    assert len(chk) == len(queries) and chk["query_id"].is_unique, "строки / повторы query_id"
    assert set(chk["query_id"]) == set(queries["query_id"]), "набор query_id"
    for a in chk["answer"]:
        it = a.split(" ")
        assert 0 < len(it) <= 50 and len(set(it)) == len(it), "размер / повторы в строке"
        assert all(x in valid for x in it), "item_id не из каталога"
    print(f"  {path} OK: {len(chk)} строк, по {len(it)} item_id, все ID из каталога", flush=True)
