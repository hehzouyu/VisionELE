"""增强模块单测：频域增强 / 图学习 / 元迁移 / 小样本生成与筛选。"""
from __future__ import annotations

import numpy as np
import pytest

from src.enhanced import cdcgan, cdmth, freq, graph


# ---------------------------------------------------------------- freq
def test_morlet_wavelet_shape_and_energy():
    t = np.arange(-16, 17, dtype=float)
    k = freq.morlet_wavelet(t, scale=4.0, center_freq=0.8)
    assert k.shape == t.shape
    assert np.all(np.isfinite(k))
    assert np.argmax(np.abs(k)) == len(k) // 2  # 峰在中心


def test_cwt_morlet_output_shape():
    x = np.sin(np.linspace(0, 8 * np.pi, 64))
    out = freq.cwt_morlet(x, scales=[1, 2, 4])
    assert out.shape == (3, 64)
    assert np.all(np.isfinite(out))


def test_vmd_bands_and_fusion_shapes():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 16, 16))
    bands = freq.vmd_bands(batch[0].ravel(), modes=3)
    assert bands.shape == (3, 256)
    fused = freq.fuse_features(batch)
    assert fused.shape == batch.shape
    assert np.all(np.isfinite(fused))


def test_cwt_response_branch_outputs_same_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 8, 8))
    att = freq.cwt_response(x, scales=[1, 2])
    assert att.shape == x.shape
    assert np.all(att >= 0.0)


# ---------------------------------------------------------------- graph
def test_build_knn_graph_symmetric_zero_diag():
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(9, 5))
    adj = graph.build_knn_graph(feats, k=3)
    assert adj.shape == (9, 9)
    assert np.allclose(adj, adj.T)
    assert np.allclose(np.diag(adj), 0.0)
    assert np.all(adj >= 0.0)


def test_normalize_adjacency_symmetric_norm():
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(6, 4))
    adj = graph.build_knn_graph(feats, k=2)
    norm = graph.normalize_adjacency(adj)
    assert np.allclose(norm, norm.T)
    # 对称归一化 D^-1/2 (A+I) D^-1/2：特征值落在 [-1, 1]
    eigvals = np.linalg.eigvalsh(norm)
    assert np.all(eigvals <= 1.0 + 1e-8)
    assert np.all(eigvals >= -1.0 - 1e-8)


def test_multi_scale_gcn_forward_and_fit_decreases_loss():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 6))
    y = np.repeat([0, 1, 2], 4)
    adj = graph.build_knn_graph(X, k=3)
    model = graph.MultiScaleGCN(in_dim=6, num_classes=3, hidden_dim=8,
                                scales=(1, 2), layers=2, seed=0)
    logits = model.forward(adj, X)
    assert logits.shape == (12, 3)
    mask = [True] * 6 + [False] * 6
    hist = model.fit(adj, X, y, mask, epochs=40, lr=0.2)
    assert len(hist) == 40
    assert hist[-1] <= hist[0] + 1e-9


def test_semi_supervised_loss_rewards_consistency():
    logits = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 0.0, 0.0]])
    labels, labeled = [0, 1, 0], [True, True, False]
    unlabeled = [False, False, True]
    worse = np.array([[0.0, 1.0, 0.0]])
    better = np.array([[1.0, 0.0, 0.0]])
    loss_worse = graph.semi_supervised_loss(logits, labels, labeled, unlabeled, worse)
    loss_better = graph.semi_supervised_loss(logits, labels, labeled, unlabeled, better)
    assert loss_better < loss_worse


def test_graph_accuracy_and_softmax():
    z = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert graph.accuracy(z, [0, 1]) == pytest.approx(1.0)
    p = graph.softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)


# ---------------------------------------------------------------- cdmth
def test_protonet_fit_predict_and_evaluate():
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(30, 12))
    labels = np.repeat([0, 1, 2], 10)
    head = cdmth.ProtoNetHead(num_classes=3, temperature=0.1).fit(feats[:9], labels[:9])
    assert head.prototypes.shape == (3, 12)
    probs = head.probabilities(feats[9:])
    assert np.allclose(probs.sum(axis=1), 1.0)
    acc = head.evaluate(feats[9:], labels[9:])
    assert 0.0 <= acc <= 1.0


def test_protonet_separable_prototypes_predict_correctly():
    feats = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    labels = [0, 0, 1, 1]
    head = cdmth.ProtoNetHead(num_classes=2).fit(feats, labels)
    assert set(head.predict(np.array([[0.9, 0.1], [0.1, 0.9]])).tolist()) == {0, 1}


def test_protonet_raises_before_fit_and_on_bad_input():
    head = cdmth.ProtoNetHead(num_classes=2)
    with pytest.raises(RuntimeError):
        head.logits(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        cdmth.ProtoNetHead(num_classes=0)
    with pytest.raises(ValueError):
        head.fit(np.zeros((2, 2)), [0])


def test_should_trigger_condition():
    assert cdmth.should_trigger(0.3, 0.9) is True
    assert cdmth.should_trigger(0.8, 0.9) is False
    assert cdmth.should_trigger(0.3, 0.2) is False


# ---------------------------------------------------------------- cdcgan
def test_screen_generated_keeps_nearest_to_center():
    gen = [[0.0, 0.0], [50.0, 50.0], [0.2, 0.1], [9.0, 9.0]]
    real = [[0.0, 0.0], [0.1, 0.1]]
    keep = cdcgan.screen_generated(gen, real, keep_ratio=0.5)
    assert keep == [0, 2]


def test_screen_generated_empty_input():
    assert cdcgan.screen_generated([], [[1.0]]) == []


@pytest.mark.skipif(not cdcgan.TORCH_AVAILABLE, reason="需要 torch")
def test_cdcgan_builds_and_produces_images():
    import torch

    cfg = cdcgan.CdcganConfig(latent_dim=8, num_classes=3, image_size=64, base_channels=32)
    g = cdcgan.build_generator(cfg)
    d = cdcgan.build_discriminator(cfg)
    z = torch.randn(2, cfg.latent_dim)
    labels = torch.tensor([0, 2])
    fake = g(z, labels)
    assert fake.shape == (2, 3, 64, 64)
    scores = d(fake, labels)
    assert scores.shape == (2,)
    d_loss, g_loss = cdcgan.hinge_losses(scores, scores.detach())
    assert torch.isfinite(d_loss) and torch.isfinite(g_loss)


@pytest.mark.skipif(cdcgan.TORCH_AVAILABLE, reason="有 torch 时该分支不可达")
def test_cdcgan_raises_without_torch():
    with pytest.raises(RuntimeError):
        cdcgan.build_generator(cdcgan.CdcganConfig())
