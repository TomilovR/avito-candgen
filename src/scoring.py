"""Первая стадия: скор по всему каталогу на GPU (без предварительной отсечки), пул и признаки пар."""

import numpy as np
import torch

from . import config
from .features import DEV, Context

# признаки для ранкера
PAIR_FEATS = ["dq", "dqp", "lex", "same", "trans", "geo", "mc05", "mc02", "mcp", "hist",
              "cat", "rev", "rating", "price", "lin", "rank", "q_seen", "q_nn1", "q_hasp",
              "q_nloc", "q_poolmax_dq", "d_dq_top", "item_nloc"]


class Scorer:
    def __init__(self, ctx: Context, side: dict, use_click: bool):
        it = ctx.items
        self.side, self.use_click = side, use_click
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=DEV)
        self.item_emb = t(ctx.item_emb, torch.float16)
        self.q_emb, self.qp_emb = t(side["q_emb"], torch.float16), t(side["qp_emb"], torch.float16)
        self.click = t(side["click"], torch.float16) if use_click else None
        self.trans, self.mc05, self.mc02, self.mcp = (t(side[k]) for k in ("trans", "mc05", "mc02", "mcp"))
        self.q_loc = t(side["q_loc"], torch.int64)
        self.item_loc = t(it["item_location_id"].map(ctx.loc_index).values, torch.int64)
        self.item_mc = t(it["item_microcat_id"].map(ctx.mc_index).values, torch.int64)
        self.cat = t((it["item_category_id"].values == 114).astype(np.float32))
        self.rev = t(np.log1p(it["item_rating_reviews_count"].astype(float).fillna(0).values))
        self.rating = t(it["item_rating"].astype(float).fillna(0).values)
        self.price = t(np.log1p(it["item_price"].astype(float).fillna(0).clip(lower=0).values))
        self.item_lat = torch.deg2rad(t(it["item_latitude"].fillna(55.75).values))
        self.item_lon = torch.deg2rad(t(it["item_longitude"].fillna(37.6).values))
        self.loc_lat, self.loc_lon = torch.deg2rad(t(ctx.loc_lat)), torch.deg2rad(t(ctx.loc_lon))
        cnt = np.bincount(it["item_location_id"].map(ctx.loc_index).values, minlength=len(ctx.loc_index))
        self.loc_cnt = t(np.log1p(cnt))
        self.N = len(it)

    def batch(self, s, e):
        # признаки запросов [s, e) против всех объявлений, B x N
        sd = self.side
        f = {"dq": self.q_emb[s:e] @ self.item_emb.T, "dqp": self.qp_emb[s:e] @ self.item_emb.T,
             "lex": torch.from_numpy(sd["lex"][s:e].toarray()).to(DEV).half(),
             "hist": torch.from_numpy(sd["hist"][s:e].toarray()).to(DEV).half()}
        if self.use_click:
            f["dclick"] = self.click[s:e] @ self.item_emb.T
        ql = self.q_loc[s:e]
        f["same"] = (ql[:, None] == self.item_loc[None]).half()
        # расстояние (гаверсинус), км
        la1, lo1 = self.loc_lat[ql][:, None], self.loc_lon[ql][:, None]
        a = (torch.sin((self.item_lat[None] - la1) / 2) ** 2
             + torch.cos(la1) * torch.cos(self.item_lat[None]) * torch.sin((self.item_lon[None] - lo1) / 2) ** 2)
        dist = 2 * 6371 * torch.asin(torch.sqrt(a.clamp(0, 1)))
        f["geo"] = (-torch.log1p(dist / 10)).half()
        return f

    def linear(self, s, e, f, w):
        S = torch.zeros((e - s, self.N), device=DEV)
        for k in ("dq", "dqp", "lex", "hist", "geo", "same", "dclick"):
            if w.get(k, 0) and k in f:
                S.add_(f[k].float(), alpha=w[k])
        if w.get("trans", 0):
            S.add_(self.trans[s:e][:, self.item_loc], alpha=w["trans"])
        for k in ("mc05", "mc02", "mcp"):
            if w.get(k, 0):
                S.add_(getattr(self, k)[s:e][:, self.item_mc], alpha=w[k])
        for k in ("cat", "rev"):
            if w.get(k, 0):
                S.add_(getattr(self, k)[None], alpha=w[k])
        return S

    @torch.no_grad()
    def pool(self, w, k=config.K_POOL, bs=250):
        # top-k по линейному скору и признаки пар Q x k x F
        feats = PAIR_FEATS + (["dclick"] if self.use_click else [])
        sd = self.side
        idx_all, X_all = [], []
        for s in range(0, sd["Q"], bs):
            e = min(s + bs, sd["Q"])
            B = e - s
            f = self.batch(s, e)
            lin, idx = self.linear(s, e, f, w).topk(k, dim=1)
            g = lambda x: x.gather(1, idx).float()
            qcol = lambda v: torch.as_tensor(v, device=DEV, dtype=torch.float32)[:, None].expand(B, k)
            c = {n: g(f[n]) for n in ("dq", "dqp", "lex", "same", "geo", "hist") + (("dclick",) if self.use_click else ())}
            c["trans"] = self.trans[s:e].gather(1, self.item_loc[idx])
            for n in ("mc05", "mc02", "mcp"):
                c[n] = getattr(self, n)[s:e].gather(1, self.item_mc[idx])
            for n in ("cat", "rev", "rating", "price"):
                c[n] = getattr(self, n)[idx]
            c["lin"] = lin
            c["rank"] = torch.arange(k, device=DEV, dtype=torch.float32)[None].expand(B, k)
            c["q_seen"] = qcol(sd["seen"][s:e].astype(np.float32))
            c["q_nn1"] = qcol(sd["nn1"][s:e])
            c["q_hasp"] = qcol(sd["hasp"][s:e])
            c["q_nloc"] = self.loc_cnt[self.q_loc[s:e]][:, None].expand(B, k)
            c["q_poolmax_dq"] = c["dq"].max(1, keepdim=True).values.expand(B, k)
            c["d_dq_top"] = c["dq"] - c["q_poolmax_dq"]
            c["item_nloc"] = self.loc_cnt[self.item_loc[idx]]
            X_all.append(torch.stack([c[n] for n in feats], dim=2).cpu().numpy())
            idx_all.append(idx.cpu().numpy())
        return np.concatenate(idx_all), np.concatenate(X_all), feats
