import torch
import torch.nn.functional as F

from losses.combined import CombinedLoss, LossConfig
from losses.focal import focal_bce_with_logits
from losses.supcon import jaccard_positive_mask, multilabel_supcon

# ---------- focal ----------

def test_focal_gamma_zero_equals_bce():
    logits = torch.randn(8, 5)
    targets = (torch.rand(8, 5) > 0.5).float()
    focal = focal_bce_with_logits(logits, targets, gamma=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(focal, bce)


def test_focal_downweights_easy_examples():
    # confident-correct predictions -> focal << bce
    logits = torch.tensor([[8.0, -8.0]])
    targets = torch.tensor([[1.0, 0.0]])
    focal = focal_bce_with_logits(logits, targets, gamma=2.0)
    bce = focal_bce_with_logits(logits, targets, gamma=0.0)
    assert focal < bce
    assert focal.item() < 1e-3


def test_focal_reduction_none_shape():
    logits = torch.randn(4, 3)
    targets = torch.zeros(4, 3)
    out = focal_bce_with_logits(logits, targets, reduction="none")
    assert out.shape == (4, 3)


def test_focal_is_nonnegative():
    logits = torch.randn(10, 7)
    targets = (torch.rand(10, 7) > 0.5).float()
    assert focal_bce_with_logits(logits, targets).item() >= 0.0


# ---------- jaccard positive mask ----------

def test_jaccard_mask_identifies_overlap():
    # rows: A={0,1}, B={0,1} (jacc 1), C={0} (jacc with A = 1/2 -> positive),
    #       D={2} (jacc 0 with all)
    labels = torch.tensor([
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]).float()
    mask = jaccard_positive_mask(labels, threshold=0.5)
    assert mask[0, 1] == 1 and mask[1, 0] == 1
    assert mask[0, 2] == 1  # jaccard {0,1} vs {0} = 1/2 >= 0.5
    assert mask[0, 3] == 0
    assert mask[3].sum() == 0  # D has no positive
    assert torch.diagonal(mask).sum() == 0  # self excluded


def test_jaccard_empty_labels_not_positive():
    labels = torch.zeros(3, 4)  # all safe
    mask = jaccard_positive_mask(labels)
    assert mask.sum() == 0


# ---------- supcon ----------

def test_supcon_zero_when_no_positives():
    emb = F.normalize(torch.randn(4, 16), dim=-1)
    labels = torch.eye(4, 8)[:, :8]  # each clip a distinct single label -> no pairs
    loss = multilabel_supcon(emb, labels)
    assert loss.item() == 0.0


def test_supcon_lower_when_positives_aligned():
    torch.manual_seed(0)
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]]).float()
    # aligned: same-label embeddings identical -> low loss
    base_a = F.normalize(torch.randn(1, 16), dim=-1)
    base_b = F.normalize(torch.randn(1, 16), dim=-1)
    aligned = torch.cat([base_a, base_a, base_b, base_b], dim=0)
    misaligned = F.normalize(torch.randn(4, 16), dim=-1)
    assert multilabel_supcon(aligned, labels) < multilabel_supcon(misaligned, labels)


def test_supcon_has_gradient():
    emb = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]]).float()
    loss = multilabel_supcon(emb, labels)
    loss.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


# ---------- combined ----------

def test_combined_equals_focal_plus_mu_supcon():
    torch.manual_seed(1)
    logits = torch.randn(4, 2)
    emb = torch.randn(4, 16)
    targets = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]]).float()
    cfg = LossConfig(mu=0.2)
    loss_fn = CombinedLoss(cfg)
    total, parts = loss_fn(logits, emb, targets)
    expected = parts["focal"] + cfg.mu * parts["supcon"]
    torch.testing.assert_close(total.detach(), expected)
    assert set(parts) == {"focal", "supcon", "total"}


def test_combined_backward():
    logits = torch.randn(4, 2, requires_grad=True)
    emb = torch.randn(4, 16, requires_grad=True)
    targets = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]]).float()
    total, _ = CombinedLoss()(logits, emb, targets)
    total.backward()
    assert logits.grad is not None and emb.grad is not None
