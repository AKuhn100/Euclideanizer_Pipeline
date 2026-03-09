"""
Unit tests for Q (pairwise-distance similarity) analysis: q_single, _q_matrix_batch, max_q_batch.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from conftest import assert_exact_or_numerical
from src.q_analysis import q_single, _q_matrix_batch, max_q_batch, DEFAULT_DELTA


def test_q_single_identical_structures():
    """Two identical structures -> Q = 1. Exact equality required; pass with note if within atol (numerical error)."""
    np.random.seed(42)
    coords = np.random.randn(5, 3).astype(np.float32)
    q = q_single(coords, coords.copy(), delta=DEFAULT_DELTA)
    assert_exact_or_numerical(q, 1.0, atol=1e-6, name="Q(identical, identical)")


def test_q_single_very_different_structures():
    """Two structures with distances differing by a lot -> Q = 0. Exact equality required; pass with note if within atol."""
    # Same shape: 3 beads in a line. Alpha: 0,1,2 on x-axis; Beta: 0, 10, 20 so distances are 10x.
    alpha = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=np.float32)
    beta = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]], dtype=np.float32)
    q = q_single(alpha, beta, delta=DEFAULT_DELTA)
    # exp(-huge) is tiny but not exactly 0 in float32; strict atol so only numerical underflow passes
    assert_exact_or_numerical(q, 0.0, atol=1e-30, name="Q(very different)")


def test_q_matrix_batch_shape():
    """_q_matrix_batch: (B, N, 3) vs (M, N, 3) -> (B, M)."""
    B, M, N = 3, 4, 5
    queries = np.random.randn(B, N, 3).astype(np.float32)
    refs = np.random.randn(M, N, 3).astype(np.float32)
    q_mat = _q_matrix_batch(queries, refs, delta=DEFAULT_DELTA)
    assert q_mat.shape == (B, M)
    assert q_mat.dtype == np.float32


def test_q_matrix_batch_identical_query_ref():
    """When query and ref are identical, that pair has Q = 1."""
    coords = np.random.randn(2, 4, 3).astype(np.float32)
    # One query, two refs: first ref is same as query -> Q=1; second ref is different
    queries = coords[0:1]
    refs = coords
    q_mat = _q_matrix_batch(queries, refs, delta=DEFAULT_DELTA)
    assert q_mat.shape == (1, 2)
    assert_exact_or_numerical(q_mat[0, 0], 1.0, atol=1e-6, name="Q(query[0], ref[0])")
    assert 0 <= q_mat[0, 1] <= 1.0  # query[0] vs ref[1] (different structure, valid Q in [0,1])


def test_max_q_batch_shape_and_identical():
    """max_q_batch returns (n_queries,) and max Q for identical query/ref is 1."""
    coords = np.random.randn(5, 4, 3).astype(np.float32)
    # Use first 2 as queries, all 5 as refs; query 0 and 1 should get max Q 1 from ref 0 and 1
    max_q = max_q_batch(coords[:2], coords, delta=DEFAULT_DELTA, query_batch_size=2, desc=None)
    assert max_q.shape == (2,)
    assert_exact_or_numerical(max_q, 1.0, atol=1e-6, name="max_q (identical query/ref)")
