"""BGE-M3 с кэшем на диске."""

import time

import numpy as np

from . import config


def item_text(title, desc, params) -> str:
    return ". ".join((str(title or ""), str(desc or "")[:200], str(params or "")[:80]))


class Encoder:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:   # сначала локальный кэш
                m = SentenceTransformer(config.BGE_M3, device="cuda", local_files_only=True)
            except Exception:
                m = SentenceTransformer(config.BGE_M3, device="cuda")
            m.half()
            m.max_seq_length = config.BGE_MAX_LEN
            self._model = m
        return self._model

    def encode(self, texts, name) -> np.ndarray:
        path = config.CACHE_DIR / f"{name}.npy"
        if path.exists():
            return np.load(path)
        t0 = time.time()
        v = self.model.encode(list(texts), batch_size=config.BGE_BATCH, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
        np.save(path, v)
        print(f"  encoded {len(texts):,} texts -> {path.name} ({time.time() - t0:.0f}s)", flush=True)
        return v
