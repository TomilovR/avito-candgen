"""Recall@50 как на платформе."""

import numpy as np


def recall_at(top_idx, truth_idx, seen, test_seen_share):
    # R_w: перевзвешиваем под долю бенчмарка, у которой текст запроса есть в train
    r = np.array([len(set(t) & set(p.tolist())) / len(t) for p, t in zip(top_idx, truth_idx)])
    rs, ru = r[seen].mean(), r[~seen].mean()
    return {"R": float(r.mean()), "R_seen": float(rs), "R_unseen": float(ru),
            "R_w": float(test_seen_share * rs + (1 - test_seen_share) * ru)}


def fmt(m):
    return f"R_w={m['R_w']:.4f} (R={m['R']:.4f}, seen={m['R_seen']:.4f}, unseen={m['R_unseen']:.4f})"
