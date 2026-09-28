"""Признаки запросов. Для val/trn всё, что считается по кликам, берётся из train без их строк."""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .data import query_with_params
from .encoders import Encoder, item_text

DEV = "cuda"


def item_lex_text(title, params, desc) -> str:
    # заголовок и параметры повторяем, чтобы поднять их вес
    return " ".join([str(title or "")] * 3 + [str(params or "")] * 2 + [str(desc or "")[:300]])


@dataclass
class Context:
    items: pd.DataFrame
    train: pd.DataFrame
    enc: Encoder
    loc_index: dict
    mc_index: dict
    loc_lat: np.ndarray
    loc_lon: np.ndarray
    item_emb: np.ndarray
    vec: TfidfVectorizer
    X_items: sparse.csr_matrix
    texts: list          # уникальные тексты запросов train
    text_pos: dict
    text_emb: np.ndarray
    titem_pos: dict      # item_id из train -> строка titem_emb
    titem_emb: np.ndarray


def build_context(items, train, enc: Encoder, use_click: bool) -> Context:
    locs = pd.unique(pd.concat([items["item_location_id"], train["item_location_id"],
                                train["search_location_id"]]))
    loc_index = {l: i for i, l in enumerate(locs)}
    mcs = pd.unique(pd.concat([items["item_microcat_id"], train["item_microcat_id"]]))
    mc_index = {m: i for i, m in enumerate(mcs)}

    # центр локации = медиана координат её объявлений
    coords = pd.concat([items[["item_location_id", "item_latitude", "item_longitude"]],
                        train[["item_location_id", "item_latitude", "item_longitude"]]])
    cen = coords.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
    fb = train.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
    loc_lat = np.zeros(len(loc_index), np.float32)
    loc_lon = np.zeros(len(loc_index), np.float32)
    for loc, i in loc_index.items():
        src = cen if loc in cen.index else fb if loc in fb.index else None
        if src is not None:
            loc_lat[i], loc_lon[i] = src.loc[loc].values

    item_emb = enc.encode([item_text(t, d, p) for t, d, p in zip(
        items["item_title_raw"], items["item_description_raw"], items["item_infm_params_text"])],
        "items_emb")
    texts = sorted(set(train["qn"]))
    text_emb = enc.encode(texts, "train_texts_emb")

    # объявления train нужны для dclick
    titem_pos, titem_emb = {}, np.zeros((0, 1024), np.float16)
    if use_click:
        ti = train.drop_duplicates("item_id")
        titem_pos = {iid: i for i, iid in enumerate(ti["item_id"])}
        titem_emb = enc.encode([item_text(t, d, p) for t, d, p in zip(
            ti["item_title_raw"], ti["item_description_raw"], ti["item_infm_params_text"])],
            "train_items_emb")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                          sublinear_tf=True, dtype=np.float32)
    X_items = vec.fit_transform([item_lex_text(t, p, d) for t, p, d in zip(
        items["item_title_raw"], items["item_infm_params_text"], items["item_description_raw"])])

    return Context(items, train, enc, loc_index, mc_index, loc_lat, loc_lon, item_emb, vec,
                   X_items, texts, {t: i for i, t in enumerate(texts)}, text_emb,
                   titem_pos, titem_emb)


def build_side(ctx: Context, tag: str, qdf: pd.DataFrame, trf: pd.DataFrame, use_click: bool) -> dict:
    # qdf - запросы, trf - строки train, которые можно использовать
    Q = len(qdf)
    items = ctx.items
    item_pos = {iid: i for i, iid in enumerate(items["item_id"])}
    side = {"Q": Q, "q_loc": qdf["search_location_id"].map(ctx.loc_index).values.astype(np.int64)}

    # эмбеддинги: запрос и запрос + фильтры
    qtexts = qdf["search_query"].fillna("").tolist()
    key = hashlib.md5("|".join(qtexts).encode()).hexdigest()[:8]   # кэш привязан к набору запросов
    side["q_emb"] = ctx.enc.encode(qtexts, f"{tag}_{key}_emb_q")
    side["qp_emb"] = ctx.enc.encode([query_with_params(q, p) for q, p in zip(
        qdf["search_query"], qdf["search_infm_params_text"])], f"{tag}_{key}_emb_qp")

    # TF-IDF, разреженная Q x N
    qt = [f"{q or ''} {p or ''}".strip() for q, p in zip(qdf["search_query"], qdf["search_infm_params_text"])]
    side["lex"] = (ctx.vec.transform(qt) @ ctx.X_items.T).tocsr().astype(np.float32)

    # переходы между локациями. ~16% кликов уходят в другую локацию, обычно соседнюю
    L = len(ctx.loc_index)
    sl = trf["search_location_id"].map(ctx.loc_index).values.astype(int)
    il = trf["item_location_id"].map(ctx.loc_index).values.astype(int)
    cnt = sparse.coo_matrix((np.ones(len(trf), np.float32), (sl, il)), shape=(L, L)).tocsr()
    glob = np.asarray(cnt.sum(0)).ravel() + 1.0
    glob /= glob.sum()
    rows = cnt[side["q_loc"]].toarray()
    side["trans"] = np.log((rows + 20 * glob[None]) / (rows.sum(1, keepdims=True) + 20)).astype(np.float32)

    # распределение микрокатегорий по 20 похожим запросам train (веса softmax по близости)
    M = len(ctx.mc_index)
    tcodes, tuniq = pd.factorize(trf["qn"].values)
    mc_of = trf["item_microcat_id"].map(ctx.mc_index).values.astype(int)
    C = sparse.coo_matrix((np.ones(len(trf), np.float32), (tcodes, mc_of)), shape=(len(tuniq), M)).tocsr()
    Cn = sparse.diags(1.0 / np.asarray(C.sum(1)).ravel()) @ C
    mc_glob = np.asarray(C.sum(0)).ravel() + 1e-3
    mc_glob /= mc_glob.sum()
    E = torch.from_numpy(ctx.text_emb[[ctx.text_pos[t] for t in tuniq]]).to(DEV)
    qe = torch.from_numpy(side["q_emb"]).to(DEV)
    K = config.KNN_TEXTS
    topv, topi = [], []
    for s in range(0, Q, 512):
        v, i = (qe[s:s + 512] @ E.T).float().topk(K, dim=1)
        topv.append(v.cpu()); topi.append(i.cpu())
    topv, topi = torch.cat(topv).numpy(), torch.cat(topi).numpy()
    del E, qe
    for name, tau in (("mc02", 0.02), ("mc05", 0.05)):
        w = np.exp((topv - topv[:, :1]) / tau)
        w /= w.sum(1, keepdims=True)
        W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(Q), K), topi.ravel())), shape=(Q, len(tuniq)))
        P = 0.95 * np.asarray((W @ Cn).todense()) + 0.05 * mc_glob[None]
        side[name] = np.log(P + 1e-6).astype(np.float32)
    side["nn1"] = topv[:, 0].astype(np.float32)          # близость ближайшего запроса train

    # микрокатегории по тексту фильтров
    t = trf[trf["pn"] != ""]
    pcodes, puniq = pd.factorize(t["pn"].values)
    Cp = sparse.coo_matrix((np.ones(len(t), np.float32), (pcodes, t["item_microcat_id"].map(ctx.mc_index).values.astype(int))),
                           shape=(len(puniq), M)).tocsr()
    pglob = np.asarray(Cp.sum(0)).ravel() + 1e-3
    pglob /= pglob.sum()
    ppos = {p: i for i, p in enumerate(puniq)}
    F = np.zeros((Q, M), np.float32)       # без фильтров признак нулевой
    for qi, p in enumerate(qdf["pn"]):
        if p in ppos:
            row = Cp[ppos[p]].toarray().ravel()
            F[qi] = np.log((row + 5 * pglob) / (row.sum() + 5) + 1e-6)
    side["mcp"] = F

    # история: сколько раз по такому же тексту выбирали объявление
    t2 = trf[trf["item_id"].isin(item_pos)]
    hist = defaultdict(Counter)
    for q, iid in zip(t2["qn"], t2["item_id"]):
        hist[q][item_pos[iid]] += 1
    r, c, d = [], [], []
    for qi, q in enumerate(qdf["qn"]):
        for ii, n in hist.get(q, {}).items():
            r.append(qi); c.append(ii); d.append(np.log1p(n))
    side["hist"] = sparse.csr_matrix((np.array(d, np.float32), (r, c)), shape=(Q, len(items)))
    side["seen"] = qdf["qn"].isin(set(trf["qn"])).values
    side["hasp"] = (qdf["pn"] != "").values.astype(np.float32)

    # dclick: средний эмбеддинг объявлений, которые выбирали по похожим запросам
    if use_click:
        side["click"] = click_centroids(ctx, trf, side["q_emb"])
    torch.cuda.empty_cache()
    return side


def click_centroids(ctx: Context, trf, q_emb) -> np.ndarray:
    T, U = len(ctx.texts), len(ctx.titem_pos)
    A = sparse.csr_matrix((np.ones(len(trf), np.float32),
                           (trf["qn"].map(ctx.text_pos).values, trf["item_id"].map(ctx.titem_pos).values)),
                          shape=(T, U))
    n = np.asarray(A.sum(1)).ravel()
    A = sparse.diags(1.0 / np.maximum(n, 1)) @ A
    # на CPU: torch.sparse.mm на GPU суммирует в произвольном порядке, и результат плавает в последних знаках
    cent = np.asarray(A @ ctx.titem_emb.astype(np.float32), dtype=np.float32)
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
    cent = torch.from_numpy(cent).to(DEV)
    has = torch.from_numpy(n > 0).to(DEV)
    text_emb = torch.from_numpy(ctx.text_emb).to(DEV)
    q = torch.from_numpy(q_emb).to(DEV)
    out = []
    for s in range(0, len(q), 512):
        sims = (q[s:s + 512] @ text_emb.T).float()
        sims[:, ~has] = -1
        v, i = sims.topk(config.KNN_TEXTS, dim=1)
        w = torch.softmax((v - v[:, :1]) / 0.05, dim=1)
        out.append(torch.nn.functional.normalize((w[:, :, None] * cent[i]).sum(1), dim=1).half().cpu().numpy())
    return np.concatenate(out)
