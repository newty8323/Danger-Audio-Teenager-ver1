"""BEATs extractor smoke — skipped unless the checkpoint is present locally."""
import os

import numpy as np
import pytest

CKPT = "weights/beats/BEATs_iter3_plus_AS2M.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="BEATs checkpoint not downloaded")


def test_beats_extracts_frame_embeddings():
    from models.beats_extractor import BEATsExtractor
    ex = BEATsExtractor(device="cpu")
    assert ex.out_dim == 768
    feats = ex.extract((0.1 * np.random.randn(16000)).astype(np.float32))  # 1s
    assert feats.ndim == 3 and feats.shape[0] == 1 and feats.shape[2] == 768
    assert np.isfinite(feats).all()
