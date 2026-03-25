"""
Unit tests for config helpers, utils, metrics, and rmsd pure functions.

No dataset or GPU; fast. Run from pipeline root: pytest tests/test_utils_and_config.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from conftest import assert_exact_or_numerical
from src.config import (
    config_diff,
    configs_match_exactly,
    expand_distmap_grid,
    expand_euclideanizer_grid,
    finalize_scoring_tau_config,
    load_config,
    peek_output_dir,
)
from src.gro_io import write_structures_gro
from src.metrics import distmap_bond_lengths, distmap_rg, distmap_scaling, compute_exp_statistics
from src.rmsd import _rmsd_matrix_batch, _recon_rmsd_one_to_one
from src.utils import (
    display_path,
    get_available_cuda_count,
    get_device,
    get_distmaps,
    get_train_test_split,
    get_upper_tri,
    load_data,
    upper_tri_to_symmetric,
    validate_dataset_for_pipeline,
)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_diff_equal():
    """config_diff returns empty list when configs are identical."""
    cfg = {"data": {"path": "x", "split_seed": 0}, "a": 1}
    assert config_diff(cfg, cfg) == []


def test_config_diff_differ():
    """config_diff returns paths where configs differ."""
    a = {"data": {"path": "x", "split_seed": 0}, "x": 1}
    b = {"data": {"path": "y", "split_seed": 0}, "x": 1}
    diffs = config_diff(a, b)
    assert any("path" in d and "data" in d for d in diffs)
    assert len(diffs) >= 1


def test_config_diff_missing_key():
    """config_diff reports missing key in one config."""
    a = {"data": {"path": "x", "split_seed": 0}}
    b = {"data": {"path": "x", "split_seed": 0, "extra": 1}}
    diffs = config_diff(a, b)
    assert any("extra" in d for d in diffs)


def test_expand_distmap_grid_combinations():
    """expand_distmap_grid expands lists into one config per combination."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    cfg = load_config(cfg_path)
    # config_test distmap has epochs: [1, 2] -> 2 combinations
    configs = expand_distmap_grid(cfg)
    assert len(configs) == 2
    assert [c["epochs"] for c in configs] == [1, 2]


def test_expand_euclideanizer_grid_num_diags():
    """expand_euclideanizer_grid includes num_diags in the grid; multiple values yield multiple configs."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    cfg = load_config(cfg_path)
    # config_test has num_diags: 50 (single value). Override to two values.
    cfg["euclideanizer"] = dict(cfg["euclideanizer"])
    cfg["euclideanizer"]["num_diags"] = [50, 100]
    configs = expand_euclideanizer_grid(cfg)
    assert len(configs) >= 2
    num_diags_vals = [c["num_diags"] for c in configs]
    assert 50 in num_diags_vals and 100 in num_diags_vals


def test_batch_size_list_raises():
    """distmap.batch_size and euclideanizer.batch_size must be single integer or null; list raises ValueError."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    with pytest.raises(ValueError, match="distmap.batch_size must be a single integer or null"):
        load_config(cfg_path, {"distmap": {"batch_size": [8, 16]}})
    with pytest.raises(ValueError, match="euclideanizer.batch_size must be a single integer or null"):
        load_config(cfg_path, {"euclideanizer": {"batch_size": [8, 16]}})


def test_batch_size_null_passes():
    """batch_size: null passes validation when calibration keys (safety_margin_gb, min_fraction_reserved, etc.) are set."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    cfg = load_config(cfg_path, {"distmap": {"batch_size": None}, "euclideanizer": {"batch_size": None}})
    assert cfg["distmap"]["batch_size"] is None
    assert cfg["euclideanizer"]["batch_size"] is None
    assert cfg["calibration_safety_margin_gb"] == 2.0


def test_batch_size_zero_raises():
    """batch_size: 0 raises ValueError (must be positive integer or null)."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    with pytest.raises(ValueError, match="batch_size must be null \\(auto-calibrate\\) or a positive integer"):
        load_config(cfg_path, {"distmap": {"batch_size": 0}})


def test_peek_output_dir_and_override():
    """peek_output_dir reads output_dir from YAML; merges CLI-style output_dir override."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    assert peek_output_dir(cfg_path) is not None
    override_dir = os.path.join(_TEST_DIR, "peek_override_out")
    got = peek_output_dir(cfg_path, {"output_dir": override_dir})
    assert got == os.path.abspath(override_dir)


def test_load_config_skips_tau_when_requested():
    """validate_scoring_tau=False skips tau file I/O; finalize_scoring_tau_config still enforces it."""
    cfg_path = os.path.join(_TEST_DIR, "config_test.yaml")
    bad_tau = "/nonexistent/scoring_tau_bad.yaml"
    cfg = load_config(cfg_path, {"scoring": {"tau_config": bad_tau}}, validate_scoring_tau=False)
    assert bad_tau in cfg["scoring"]["tau_config"]
    with pytest.raises(FileNotFoundError, match="Scoring tau config not found"):
        finalize_scoring_tau_config(cfg, cfg_path)


def test_configs_match_exactly_equal():
    """configs_match_exactly True when configs are equal."""
    cfg = {"a": 1, "b": {"c": 2}}
    assert configs_match_exactly(cfg, cfg) is True


def test_configs_match_exactly_unequal():
    """configs_match_exactly False when configs differ."""
    a = {"a": 1, "b": 2}
    b = {"a": 1, "b": 3}
    assert configs_match_exactly(a, b) is False


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def test_get_upper_tri_upper_tri_to_symmetric_roundtrip():
    """get_upper_tri and upper_tri_to_symmetric round-trip (off-diagonal only; diagonal is not stored)."""
    B, N = 2, 4
    distmaps = torch.rand(B, N, N)
    distmaps = (distmaps + distmaps.transpose(1, 2)) / 2
    distmaps[:, range(N), range(N)] = 0  # get_upper_tri(offset=1) does not include diagonal
    flat = get_upper_tri(distmaps)
    back = upper_tri_to_symmetric(flat, N)
    torch.testing.assert_close(back, distmaps)


def test_load_data_valid_npz(tmp_path):
    """load_data reads a minimal valid NPZ and returns (n_structures, n_atoms, 3) float32."""
    npz = tmp_path / "tiny.npz"
    coords = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32)
    np.savez_compressed(npz, coords=coords)
    out = load_data(str(npz))
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.float32
    assert_exact_or_numerical(out[0, 0], np.array([0.1, 0.2, 0.3]), atol=1e-7, name="coords row 0")
    assert_exact_or_numerical(out[0, 1], np.array([0.4, 0.5, 0.6]), atol=1e-7, name="coords row 1")


def test_gro_roundtrip(tmp_path):
    """write_structures_gro writes canonical GRO file; verify file exists and has expected frame layout."""
    coords = np.random.rand(5, 10, 3).astype(np.float32)
    paths = write_structures_gro(coords, str(tmp_path), title="test roundtrip")
    assert len(paths) == 1
    gro_path = tmp_path / "structures.gro"
    assert gro_path.exists()
    lines = gro_path.read_text().strip().split("\n")
    # 5 frames: each frame = 1 title + 1 count + 10 atoms + 1 box = 13 lines
    assert len(lines) >= 5 * 13
    # Default title when None
    write_structures_gro(coords[:1], str(tmp_path), filename="single.gro")
    single = (tmp_path / "single.gro").read_text().split("\n")
    assert single[0] == "generated frame 0"


def test_gro_single_structure(tmp_path):
    """write_structures_gro with single structure produces one frame."""
    coords = np.random.rand(1, 20, 3).astype(np.float32)
    write_structures_gro(coords, str(tmp_path), title="single")
    gro_path = tmp_path / "structures.gro"
    assert gro_path.exists()
    lines = gro_path.read_text().strip().split("\n")
    assert lines[0] == "single"
    assert lines[1] == "20"
    assert len(lines) == 1 + 1 + 20 + 1  # title, count, atoms, box


def test_compute_exp_statistics_output_keys():
    """compute_exp_statistics produces expected keys used by scoring and plotting."""
    coords_np = np.random.rand(20, 10, 3).astype(np.float32)
    device = torch.device("cpu")
    stats = compute_exp_statistics(
        coords_np, device, get_distmaps,
        max_sep=5, chunk_size=10, avg_map_sample=20,
    )
    required = {
        "exp_distmaps", "exp_bonds", "exp_rg",
        "genomic_distances", "exp_scaling", "avg_exp_map",
    }
    assert required.issubset(stats.keys())
    assert stats["exp_distmaps"].shape == (20, 10, 10)
    assert stats["avg_exp_map"].shape == (10, 10)
    assert len(stats["genomic_distances"]) == 5


def test_load_data_raises_missing_coords_key(tmp_path):
    """load_data raises ValueError when NPZ does not contain 'coords' key."""
    npz = tmp_path / "bad.npz"
    np.savez_compressed(npz, other=np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="does not contain required key 'coords'"):
        load_data(str(npz))


def test_load_data_raises_wrong_ndim(tmp_path):
    """load_data raises ValueError when coords ndim != 3."""
    npz = tmp_path / "bad.npz"
    np.savez_compressed(npz, coords=np.zeros((5, 3)))  # 2D
    with pytest.raises(ValueError, match="must have 3 dimensions"):
        load_data(str(npz))


def test_load_data_raises_wrong_last_dim(tmp_path):
    """load_data raises ValueError when coords.shape[2] != 3."""
    npz = tmp_path / "bad.npz"
    np.savez_compressed(npz, coords=np.zeros((2, 5, 4)))
    with pytest.raises(ValueError, match="last dimension must be 3"):
        load_data(str(npz))


def test_load_data_raises_zero_structures(tmp_path):
    """load_data raises ValueError when coords has zero structures."""
    npz = tmp_path / "bad.npz"
    np.savez_compressed(npz, coords=np.zeros((0, 5, 3)))
    with pytest.raises(ValueError, match="at least one structure"):
        load_data(str(npz))


def test_load_data_raises_fewer_than_two_atoms(tmp_path):
    """load_data raises ValueError when coords has fewer than 2 atoms."""
    npz = tmp_path / "bad.npz"
    np.savez_compressed(npz, coords=np.zeros((2, 1, 3)))
    with pytest.raises(ValueError, match="at least 2 atoms"):
        load_data(str(npz))


def test_load_data_raises_non_finite(tmp_path):
    """load_data raises ValueError when coords contains NaN or inf."""
    npz = tmp_path / "bad.npz"
    coords = np.zeros((2, 5, 3), dtype=np.float32)
    coords[0, 0, 0] = np.nan
    np.savez_compressed(npz, coords=coords)
    with pytest.raises(ValueError, match="non-finite"):
        load_data(str(npz))


def test_load_data_raises_file_not_found():
    """load_data raises ValueError when file does not exist."""
    with pytest.raises(ValueError, match="does not exist or cannot be read"):
        load_data("/nonexistent/path.npz")


def test_display_path_with_root():
    """display_path with root returns path relative to root when possible."""
    root = "/foo/bar"
    path = "/foo/bar/baz/qux"
    out = display_path(path, root)
    assert "baz" in out and "qux" in out
    assert not out.startswith("/foo/bar")


def test_display_path_no_root():
    """display_path with root=None returns path unchanged."""
    path = "/some/abs/path"
    assert display_path(path, None) == path


def test_get_device_default():
    """get_device() with no args returns a device (mps/cuda/cpu)."""
    d = get_device()
    assert d.type in ("mps", "cuda", "cpu")


def test_get_device_index_none_unchanged():
    """get_device(device_index=None) matches get_device() default behavior."""
    d = get_device(device_index=None)
    assert d == get_device()


def test_get_device_index_cuda_unavailable_returns_cpu():
    """When CUDA is not available, get_device(device_index=0) returns CPU."""
    import unittest.mock as mock
    with mock.patch("torch.cuda.is_available", return_value=False):
        d = get_device(device_index=0)
    assert d.type == "cpu"


def test_get_device_index_invalid_raises():
    """get_device(device_index) raises when index >= device_count."""
    import unittest.mock as mock
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.device_count", return_value=2
    ):
        get_device(device_index=0)
        get_device(device_index=1)
        with pytest.raises(ValueError, match="device_index must be"):
            get_device(device_index=2)
        with pytest.raises(ValueError, match="device_index must be"):
            get_device(device_index=-1)


def test_get_device_index_cuda_returns_cuda_device():
    """When CUDA is available, get_device(device_index=0) returns cuda:0."""
    import unittest.mock as mock
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.device_count", return_value=4
    ):
        d = get_device(device_index=0)
        assert d.type == "cuda" and d.index == 0
        d = get_device(device_index=3)
        assert d.type == "cuda" and d.index == 3


def test_get_available_cuda_count():
    """get_available_cuda_count returns 0 when CUDA unavailable, else device_count."""
    import unittest.mock as mock
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert get_available_cuda_count() == 0
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.device_count", return_value=3
    ):
        assert get_available_cuda_count() == 3


def test_get_train_test_split_shape():
    """get_train_test_split returns train and test subsets with correct total size."""
    coords = torch.randn(100, 10, 3)
    train_ds, test_ds = get_train_test_split(coords, training_split=0.8, split_seed=42)
    assert len(train_ds) + len(test_ds) == 100
    assert len(train_ds) == 80
    assert len(test_ds) == 20


def test_get_train_test_split_reproducible():
    """get_train_test_split is reproducible with same seed."""
    coords = torch.randn(50, 5, 3)
    t1, s1 = get_train_test_split(coords, 0.7, split_seed=123)
    t2, s2 = get_train_test_split(coords, 0.7, split_seed=123)
    assert t1.indices == t2.indices
    assert s1.indices == s2.indices


def test_validate_dataset_for_pipeline_ok():
    """validate_dataset_for_pipeline does not raise for valid dataset."""
    validate_dataset_for_pipeline(10, 0.8)
    validate_dataset_for_pipeline(2, 0.5)


def test_validate_dataset_for_pipeline_too_few_structures():
    """validate_dataset_for_pipeline raises when num_structures < 2."""
    with pytest.raises(ValueError, match="At least 2 structures"):
        validate_dataset_for_pipeline(1, 0.8)


def test_validate_dataset_for_pipeline_split_imbalance():
    """validate_dataset_for_pipeline raises when train or test size is 0."""
    with pytest.raises(ValueError, match="train_size=0|At least one"):
        validate_dataset_for_pipeline(5, 0.0)
    with pytest.raises(ValueError, match="test_size=0|At least one"):
        validate_dataset_for_pipeline(5, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_distmap_bond_lengths_shape():
    """distmap_bond_lengths returns flat bond lengths (N-1) per structure."""
    B, N = 3, 5
    dm = np.random.rand(B, N, N).astype(np.float32)
    dm = (dm + dm.transpose(0, 2, 1)) / 2
    bonds = distmap_bond_lengths(dm)
    assert bonds.shape == (B * (N - 1),)
    assert np.all(bonds >= 0)


def test_distmap_bond_lengths_known_values():
    """distmap_bond_lengths returns d(i,i+1); known DM -> known bond lengths. Catches wrong indices."""
    # One structure, 3 beads: DM so d(0,1)=2, d(1,2)=3, d(0,2)=5. Symmetric, diag 0.
    dm = np.zeros((1, 3, 3), dtype=np.float32)
    dm[0, 0, 1] = dm[0, 1, 0] = 2.0
    dm[0, 1, 2] = dm[0, 2, 1] = 3.0
    dm[0, 0, 2] = dm[0, 2, 0] = 5.0
    bonds = distmap_bond_lengths(dm)
    assert bonds.shape == (2,)
    np.testing.assert_array_almost_equal(bonds, [2.0, 3.0])


def test_distmap_rg_shape_and_nonneg():
    """distmap_rg returns one Rg per structure, non-negative."""
    B, N = 4, 6
    dm = np.random.rand(B, N, N).astype(np.float32)
    dm = (dm + dm.transpose(0, 2, 1)) / 2
    rg = distmap_rg(dm)
    assert rg.shape == (B,)
    assert np.all(rg >= 0)
    # Non-degenerate distance matrix must give positive Rg (catches always-zero or wrong formula)
    assert np.all(rg > 0), "Rg must be positive for non-zero distances"


def test_distmap_scaling_shape():
    """distmap_scaling returns genomic_distances and mean_spatial_distances."""
    B, N = 2, 10
    dm = np.random.rand(B, N, N).astype(np.float32)
    dm = (dm + dm.transpose(0, 2, 1)) / 2
    gen_d, mean_d = distmap_scaling(dm, max_sep=5)
    assert gen_d.shape == (5,)
    assert mean_d.shape == (5,)
    assert np.all(mean_d >= 0)
    # Positive distances -> mean_d must be positive (catches always-zero)
    assert np.all(mean_d > 0), "mean_spatial_distances must be positive for positive DM"


# ---------------------------------------------------------------------------
# RMSD analysis
# ---------------------------------------------------------------------------


def test_rmsd_matrix_batch_shape():
    """_rmsd_matrix_batch returns (B, M) for (B, N, 3) queries and (M, N, 3) refs."""
    B, M, N = 2, 3, 4
    queries = np.random.randn(B, N, 3).astype(np.float32)
    refs = np.random.randn(M, N, 3).astype(np.float32)
    rmsd = _rmsd_matrix_batch(queries, refs)
    assert rmsd.shape == (B, M)
    assert np.all(rmsd >= 0)


def test_rmsd_matrix_batch_identical_coords_zero():
    """When query and ref are identical copies, RMSD must be zero. Exact equality required; pass with note if within atol."""
    N = 5
    coords = np.random.randn(1, N, 3).astype(np.float32)
    refs = np.copy(coords)
    rmsd = _rmsd_matrix_batch(coords, refs)
    assert rmsd.shape == (1, 1)
    assert_exact_or_numerical(rmsd[0, 0], 0.0, atol=1e-6, name="RMSD(identical coords)")


def test_rmsd_matrix_batch_rotated_aligns_to_near_zero():
    """Rotated+translated copy aligned back to original must give near-zero RMSD. Catches wrong Kabsch formula."""
    rng = np.random.default_rng(42)
    N = 20
    original = rng.uniform(-10, 10, size=(N, 3)).astype(np.float32)
    rot = Rotation.random(random_state=rng)
    translation = rng.uniform(-2, 2, size=3).astype(np.float32)
    rotated = rot.apply(original) + translation
    rmsd = _rmsd_matrix_batch(
        rotated[np.newaxis, ...],
        original[np.newaxis, ...],
    )
    assert rmsd.shape == (1, 1)
    # Wrong Kabsch gives RMSD ~1+; float32 noise can be ~1e-6
    assert rmsd[0, 0] < 1e-5, (
        f"Rotate-then-align should give ~0 RMSD; got {rmsd[0, 0]}. "
        "If this fails, Kabsch alignment is wrong."
    )


def test_rmsd_matrix_batch_matches_scipy():
    """Pipeline RMSD must match scipy Rotation.align_vectors result for the same pair."""
    rng = np.random.default_rng(123)
    N = 15
    ref = rng.uniform(-5, 5, size=(N, 3)).astype(np.float32)
    query = Rotation.random(random_state=rng).apply(ref) + rng.uniform(-1, 1, size=3).astype(np.float32)
    pipeline_rmsd = _rmsd_matrix_batch(
        query[np.newaxis, ...],
        ref[np.newaxis, ...],
    )[0, 0]
    # Scipy: align query to ref, then compute RMSD
    q_c = query - query.mean(axis=0)
    r_c = ref - ref.mean(axis=0)
    rot, _ = Rotation.align_vectors(r_c, q_c)
    aligned = rot.apply(q_c) + ref.mean(axis=0)
    scipy_rmsd = np.sqrt(np.mean((aligned - ref) ** 2))
    assert np.isclose(pipeline_rmsd, scipy_rmsd, rtol=1e-5, atol=1e-8), (
        f"Pipeline RMSD {pipeline_rmsd} should match scipy {scipy_rmsd}"
    )


def test_recon_rmsd_one_to_one():
    """_recon_rmsd_one_to_one: identical original/recon must give RMSD=0. Exact equality required; pass with note if within atol."""
    S, N = 3, 4
    original = np.random.randn(S, N, 3).astype(np.float32)
    recon = np.copy(original)
    rmsds = _recon_rmsd_one_to_one(original, recon)
    assert rmsds.shape == (S,)
    assert_exact_or_numerical(rmsds, 0.0, atol=1e-6, name="recon RMSD (identical original/recon)")


def test_recon_rmsd_one_to_one_rotated():
    """_recon_rmsd_one_to_one: recon = rotated(original) must give ~0 RMSD per pair. Catches wrong Kabsch in recon path."""
    rng = np.random.default_rng(99)
    S, N = 2, 15
    original = rng.uniform(-5, 5, size=(S, N, 3)).astype(np.float32)
    rot = Rotation.random(random_state=rng)
    trans = rng.uniform(-1, 1, size=3).astype(np.float32)
    recon = np.array([rot.apply(original[i]) + trans for i in range(S)], dtype=np.float32)
    rmsds = _recon_rmsd_one_to_one(original, recon)  # (original_coords, recon_coords)
    assert rmsds.shape == (S,)
    assert np.all(rmsds < 1e-5), (
        f"Recon RMSD (rotated->original) should be ~0; got {rmsds}. Wrong alignment in recon path?"
    )


# ---------------------------------------------------------------------------
# Multi-GPU task list and assignment (run.py logic)
# ---------------------------------------------------------------------------


def test_task_list_seeds_x_dm_groups():
    """Task list is (seed, gidx) for each seed and each dm_group index."""
    seeds = [0, 1]
    dm_groups = [{"base_config": {}}, {"base_config": {}}]
    tasks = [(s, g) for s in seeds for g in range(len(dm_groups))]
    assert tasks == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_task_assignment_round_robin_even():
    """Round-robin: 4 tasks, 2 GPUs -> 2 tasks per device."""
    tasks = [(0, 0), (0, 1), (1, 0), (1, 1)]
    n_gpus = 2
    tasks_by_device = [list() for _ in range(n_gpus)]
    for i, t in enumerate(tasks):
        tasks_by_device[i % n_gpus].append(t)
    assert len(tasks_by_device[0]) == 2
    assert len(tasks_by_device[1]) == 2
    assert tasks_by_device[0] == [(0, 0), (1, 0)]
    assert tasks_by_device[1] == [(0, 1), (1, 1)]


def test_task_assignment_round_robin_odd():
    """Round-robin: 5 tasks, 2 GPUs -> 3 and 2."""
    tasks = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    n_gpus = 2
    tasks_by_device = [list() for _ in range(n_gpus)]
    for i, t in enumerate(tasks):
        tasks_by_device[i % n_gpus].append(t)
    assert len(tasks_by_device[0]) == 3
    assert len(tasks_by_device[1]) == 2


def test_single_gpu_path_when_one_device():
    """When only 1 CUDA device is available, use_multi_gpu is False (single-threaded path)."""
    import unittest.mock as mock
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.device_count", return_value=1
    ):
        n_gpus = get_available_cuda_count()
        use_multi_gpu = (n_gpus >= 2)
        assert n_gpus == 1
        assert use_multi_gpu is False


# ---------------------------------------------------------------------------
# Plot paths (run.PLOT_TYPES / _plot_path)
# ---------------------------------------------------------------------------


def test_plot_path_from_plottypes():
    """_plot_path builds paths from PLOT_TYPES for reconstruction, recon_statistics, gen_variance."""
    from run import PLOT_TYPES, _plot_path

    run_root = "/out/seed_0/distmap/0"
    assert _plot_path(run_root, "reconstruction") == os.path.join(
        run_root, "plots", "reconstruction", "reconstruction.png"
    )
    assert _plot_path(run_root, "recon_statistics", subset="test") == os.path.join(
        run_root, "plots", "recon_statistics", "recon_statistics_test.png"
    )
    assert _plot_path(run_root, "gen_variance", var="1") == os.path.join(
        run_root, "plots", "gen_variance", "gen_variance_1.png"
    )
    assert _plot_path(run_root, "bond_length_by_genomic_distance_gen") == os.path.join(
        run_root, "plots", "bond_length_by_genomic_distance_gen", "bond_length_by_genomic_distance_gen.png"
    )
    assert set(PLOT_TYPES.keys()) == {
        "reconstruction",
        "recon_statistics",
        "gen_variance",
        "bond_length_by_genomic_distance_gen",
        "bond_length_by_genomic_distance_train",
        "bond_length_by_genomic_distance_test",
    }
