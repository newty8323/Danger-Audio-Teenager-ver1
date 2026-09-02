import pytest
import torch

from models.backbones import ConvFrameBackbone, MFCCBiLSTMBackbone, build_backbone
from models.harm_model import HarmModel, ModelConfig
from models.heads import ClassifierHead, ProjectionHead
from models.pooling import MILAttentionPooling


def test_mil_attention_shapes_and_weights_sum_to_one():
    pool = MILAttentionPooling(dim=64, attn_dim=32)
    frames = torch.randn(4, 20, 64)
    z, attn = pool(frames)
    assert z.shape == (4, 64)
    assert attn.shape == (4, 20)
    torch.testing.assert_close(attn.sum(dim=1), torch.ones(4), atol=1e-5, rtol=0)


def test_mil_pooled_is_convex_combination():
    pool = MILAttentionPooling(dim=8, attn_dim=8)
    frames = torch.randn(2, 10, 8)
    z, _ = pool(frames)
    # weighted average lies within the per-dim [min, max] over frames
    assert (z <= frames.max(dim=1).values + 1e-5).all()
    assert (z >= frames.min(dim=1).values - 1e-5).all()


def test_mil_mask_excludes_frames():
    pool = MILAttentionPooling(dim=4, attn_dim=4)
    frames = torch.randn(1, 5, 4)
    mask = torch.tensor([[True, True, False, False, False]])
    _, attn = pool(frames, mask=mask)
    assert torch.allclose(attn[0, 2:], torch.zeros(3), atol=1e-6)


def test_mil_fully_masked_row_no_nan():
    pool = MILAttentionPooling(dim=4, attn_dim=4)
    frames = torch.randn(1, 5, 4)
    mask = torch.zeros(1, 5, dtype=torch.bool)  # nothing kept
    z, attn = pool(frames, mask=mask)
    assert torch.isfinite(attn).all()
    assert torch.isfinite(z).all()


def test_projection_head_is_unit_norm():
    head = ProjectionHead(dim=32, proj_dim=16)
    out = head(torch.randn(6, 32))
    torch.testing.assert_close(out.norm(dim=1), torch.ones(6), atol=1e-5, rtol=0)


def test_classifier_head_shape():
    head = ClassifierHead(dim=32, num_classes=23)
    assert head(torch.randn(3, 32)).shape == (3, 23)


def test_conv_backbone_frame_shape():
    bb = ConvFrameBackbone(out_dim=128)
    frames = bb(torch.randn(2, 1, 128, 100))
    assert frames.ndim == 3
    assert frames.shape[0] == 2 and frames.shape[2] == 128


def test_build_backbone_unwired_raises():
    with pytest.raises(NotImplementedError):
        build_backbone("panns")  # still needs external weights + adapter
    with pytest.raises(ValueError):
        build_backbone("does_not_exist")


def test_mfcc_bilstm_backbone_shape_and_time_preserved():
    bb = MFCCBiLSTMBackbone(out_dim=128)
    x = torch.randn(2, 1, 128, 60)
    frames = bb(x)
    assert frames.shape == (2, 60, 128)  # T preserved, out_dim = 2*hidden
    assert bb.out_dim == 128


def test_mfcc_bilstm_odd_out_dim_raises():
    with pytest.raises(ValueError):
        MFCCBiLSTMBackbone(out_dim=127)


def test_harm_model_with_mfcc_backbone():
    model = HarmModel(num_classes=23, cfg=ModelConfig(backbone="mfcc_bilstm",
                                                      backbone_out_dim=128))
    out = model(torch.randn(2, 1, 128, 50))
    assert out["logits"].shape == (2, 23)
    assert out["embeddings"].shape == (2, model.cfg.proj_dim)


def test_passthrough_backbone_identity():
    from models.backbones import PassthroughBackbone
    bb = PassthroughBackbone(out_dim=768)
    frames = torch.randn(2, 40, 768)  # precomputed frame embeddings
    torch.testing.assert_close(bb(frames), frames)


def test_harm_model_with_passthrough_on_frame_embeddings():
    # frozen-BEATs path: model trains directly on (B, T, D) frame embeddings
    model = HarmModel(num_classes=23, cfg=ModelConfig(backbone="beats", backbone_out_dim=768))
    out = model(torch.randn(2, 496, 768))  # BEATs-shaped features
    assert out["logits"].shape == (2, 23)
    assert out["attention"].shape == (2, 496)


def test_harm_model_forward_shapes():
    model = HarmModel(num_classes=23, cfg=ModelConfig(backbone_out_dim=128))
    x = torch.randn(3, 1, 128, 100)
    out = model(x)
    assert out["logits"].shape == (3, 23)
    assert out["embeddings"].shape == (3, model.cfg.proj_dim)
    assert out["attention"].shape[0] == 3
    torch.testing.assert_close(
        out["embeddings"].norm(dim=1), torch.ones(3), atol=1e-5, rtol=0
    )


def test_harm_model_skip_projection():
    model = HarmModel(num_classes=5, cfg=ModelConfig(backbone_out_dim=64))
    out = model(torch.randn(2, 1, 128, 80), return_projection=False)
    assert "embeddings" not in out


def test_predict_proba_in_unit_range():
    model = HarmModel(num_classes=9, cfg=ModelConfig(backbone_out_dim=64))
    p = model.predict_proba(torch.randn(2, 1, 128, 80))
    assert p.shape == (2, 9)
    assert (p >= 0).all() and (p <= 1).all()


def test_predict_proba_preserves_training_mode():
    model = HarmModel(num_classes=4, cfg=ModelConfig(backbone_out_dim=64))
    model.train()
    model.predict_proba(torch.randn(2, 1, 128, 80))
    assert model.training is True  # not left in eval mode


def test_gradient_flows_through_model():
    model = HarmModel(num_classes=4, cfg=ModelConfig(backbone_out_dim=64))
    out = model(torch.randn(2, 1, 128, 80))
    (out["logits"].sum() + out["embeddings"].sum()).backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)
