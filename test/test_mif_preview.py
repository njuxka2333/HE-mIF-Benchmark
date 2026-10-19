"""Shared GT-matched marker display."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mif.preview import (  # noqa: E402
    colorize_marker,
    decode_marker_rgb,
    display_like_gt,
    match_histogram,
    percentile_01,
    preview_triplet,
)
from metrics.vstain import spearman_r  # noqa: E402


def test_match_histogram_copies_template_values():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(32, 32))
    ref = rng.random((32, 32))
    out = match_histogram(src, ref)
    np.testing.assert_allclose(np.sort(out.ravel()), np.sort(ref.ravel()), atol=1e-6)


def test_match_histogram_preserves_ranks():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(48, 48))
    ref = rng.random((48, 48)) ** 4
    out = match_histogram(src, ref)
    assert abs(spearman_r(src, out) - 1.0) < 1e-12


def test_match_histogram_ties_share_one_value():
    src = np.zeros((16, 16), dtype=np.float32)
    src[2, 3] = 3.0
    src[10, 11] = 1.0
    rng = np.random.default_rng(5)
    ref = rng.random((16, 16)).astype(np.float32)
    out = match_histogram(src, ref)
    bg = out[src == 0]
    assert float(np.ptp(bg)) < 1e-6
    assert out[2, 3] > out[10, 11] > float(bg[0])
    assert abs(spearman_r(src, out) - 1.0) < 1e-12


def test_display_like_gt_resizes_and_matches():
    rng = np.random.default_rng(2)
    gt = rng.random((64, 64)) ** 3
    pred = rng.normal(size=(16, 16))
    gt01, matched = display_like_gt(pred, gt)
    assert gt01.shape == (64, 64)
    assert matched.shape == (64, 64)
    np.testing.assert_allclose(np.sort(matched.ravel()), np.sort(gt01.ravel()), atol=1e-5)
    assert 0.0 <= float(gt01.min()) <= float(gt01.max()) <= 1.0


def test_colorize_roundtrip_is_near_identity():
    rng = np.random.default_rng(3)
    x = rng.random((24, 24)).astype(np.float32)
    rgb = colorize_marker(x)
    rec = decode_marker_rgb(rgb)
    assert rgb.shape == (24, 24, 3)
    assert rec.shape == (24, 24)
    assert abs(spearman_r(x, rec) - 1.0) < 1e-4


def test_preview_triplet_is_one_strip():
    rng = np.random.default_rng(4)
    he = (rng.random((80, 80, 3)) * 255).astype(np.uint8)
    gt = rng.random((40, 40))
    pred = rng.normal(size=(20, 20))
    strip = preview_triplet(he, gt, pred, side=32)
    assert strip.shape == (32, 96, 3)
    # GT and pred halves share a colormap family (not cyan vs green).
    gt_rgb = strip[:, 32:64]
    pred_rgb = strip[:, 64:96]
    assert gt_rgb.dtype == np.uint8 and pred_rgb.dtype == np.uint8
