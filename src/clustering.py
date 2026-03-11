"""
Clustering analysis: dendrograms and mixing for train/generated/test (gen) or train/test/train_recon/test_recon (recon).
Uses upper-triangle distance-map RMSE, FPS subsampling, and UPGMA linkage. Seed-level cache for train/test feats.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

from . import utils
from .utils import display_path, get_train_test_split
from .plotting import _save_pdf_copy

DEFAULT_N_SUBSAMPLE = 150
DEFAULT_K_MIXING = 10
DEFAULT_N_CLUSTERS = 8
LINKAGE_METHOD = "average"
FPS_SEED = 0

SOURCE_COLORS_GEN = {
    "Training": "#4878d0",
    "Generated": "#6acc65",
    "Test": "#ee854a",
}
SOURCE_ORDER_GEN = ["Training", "Generated", "Test"]

SOURCE_COLORS_RECON = {
    "Training": "#4878d0",
    "Train recon": "#5a9c50",
    "Test": "#ee854a",
    "Test recon": "#f0a060",
}
SOURCE_ORDER_RECON = ["Training", "Train recon", "Test", "Test recon"]


def _feats_from_coords(
    coords: torch.Tensor,
    device: torch.device,
    num_atoms: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Coords (total, N, 3) -> distance maps -> upper-triangle feature vectors. Returns (total, tri_dim) float32."""
    coords = coords.to(device)
    parts = []
    for start in range(0, coords.size(0), batch_size):
        end = min(start + batch_size, coords.size(0))
        batch = coords[start:end]
        with torch.no_grad():
            dm = utils.get_distmaps(batch)
            tri = utils.get_upper_tri(dm)
        parts.append(tri.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def _kabsch_align_to_ref(coord: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align coord (N, 3) to ref (N, 3) via Kabsch; return aligned (N, 3) float32."""
    q_c = coord - coord.mean(axis=0, keepdims=True)
    r_c = ref - ref.mean(axis=0, keepdims=True)
    ref_mean = ref.mean(axis=0)
    H = q_c.T @ r_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    S = np.eye(3, dtype=H.dtype)
    S[2, 2] = np.sign(d)
    R = U @ S @ Vt
    aligned = q_c @ R + ref_mean
    return aligned.astype(np.float32)


def _feats_from_coords_aligned(coords_np: np.ndarray) -> np.ndarray:
    """Coords (N, num_atoms, 3) -> align each to coords_np[0], flatten to (N, 3*num_atoms) float32."""
    if len(coords_np) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    ref = coords_np[0]
    out = []
    for i in range(len(coords_np)):
        aligned = _kabsch_align_to_ref(coords_np[i], ref)
        out.append(aligned.ravel())
    return np.stack(out, axis=0).astype(np.float32)


def _fps_subsample(feats: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Farthest-point sampling on PCA-compressed features. Returns indices of length min(n, len(feats))."""
    N = len(feats)
    if N <= n:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    n_pca = min(30, feats.shape[1], N - 1)
    pca = PCA(n_components=n_pca, random_state=0)
    fp = pca.fit_transform(feats).astype(np.float32)
    selected = [int(rng.integers(N))]
    min_dists = np.full(N, np.inf, dtype=np.float32)
    for _ in range(n - 1):
        pt = fp[selected[-1]]
        d = ((fp - pt) ** 2).sum(-1)
        np.minimum(min_dists, d, out=min_dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))
    return np.array(selected)


def _pairwise_rmse(feats: np.ndarray) -> np.ndarray:
    """Pairwise RMSE matrix (N, N) from (N, D) features."""
    sq = (feats ** 2).sum(-1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (feats @ feats.T)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def _compute_linkage_and_cophenetic(feats: np.ndarray, method: str = LINKAGE_METHOD) -> tuple:
    """Return (linkage_matrix, cophenetic_r)."""
    D = _pairwise_rmse(feats)
    cond = squareform(D, checks=False)
    Z = linkage(cond, method=method)
    c, _ = cophenet(Z, cond)
    return Z, float(c)


def _source_labels_array(sizes: list, source_names: list) -> np.ndarray:
    """String label array: [src0]*sizes[0] ++ [src1]*sizes[1] ++ ..."""
    labels = []
    for name, sz in zip(source_names, sizes):
        labels.extend([name] * sz)
    return np.array(labels)


def _mixing_score(feats: np.ndarray, labels: np.ndarray, k: int = DEFAULT_K_MIXING) -> tuple:
    """Mean and per-structure fraction of k-NN from a different source."""
    D = _pairwise_rmse(feats)
    np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, axis=1)[:, :k]
    mix = np.array([np.mean(labels[knn[i]] != labels[i]) for i in range(len(labels))])
    return mix.mean(), mix


def _expected_mixing(labels: np.ndarray, k: int = DEFAULT_K_MIXING) -> float:
    """Expected mixing under random assignment."""
    N = len(labels)
    if N <= 1:
        return 0.0
    unique, counts = np.unique(labels, return_counts=True)
    per_struct = np.array(
        [(N - counts[np.where(unique == lbl)[0][0]]) / (N - 1) for lbl in labels]
    )
    return float(per_struct.mean())


def _cluster_source_composition(
    Z: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> tuple:
    """(n_clusters, n_sources) counts and sorted source names."""
    cluster_ids = fcluster(Z, n_clusters, criterion="maxclust")
    sources = sorted(set(labels))
    comp = np.zeros((n_clusters, len(sources)), dtype=int)
    for ci in range(1, n_clusters + 1):
        mask = cluster_ids == ci
        for si, src in enumerate(sources):
            comp[ci - 1, si] = (labels[mask] == src).sum()
    return comp, sources


def get_or_compute_distmap_clustering_feats(
    cache_path: str,
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    n_subsample: int,
    batch_size: int = 64,
    fps_seed: int = FPS_SEED,
    display_root: str | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Load or compute train/test subsampled distance-map feats; save to cache_path.
    Returns (cache_path, train_coords_np, test_coords_np).
    Cache stores train_feats, test_feats (FPS-subsampled to n_subsample) for reuse by gen and recon.
    max_train/max_test cap the reference set sizes (None = use all).
    """
    if os.path.isfile(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            train_feats = np.asarray(loaded["train_feats"], dtype=np.float32)
            test_feats = np.asarray(loaded["test_feats"], dtype=np.float32)
            loaded.close()
            coords = coords_tensor
            train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
            tr_idx = train_ds.indices
            te_idx = test_ds.indices
            if hasattr(tr_idx, "tolist"):
                tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
            train_coords_np = coords_np[tr_idx]
            test_coords_np = coords_np[te_idx]
            if max_train is not None:
                train_coords_np = train_coords_np[:max_train]
            if max_test is not None:
                test_coords_np = test_coords_np[:max_test]
            if display_root is not None:
                print(f"  Loaded seed-level distmap clustering feats cache: {display_path(cache_path, display_root)}")
            return cache_path, train_coords_np, test_coords_np
        except Exception:
            pass
    device = coords_tensor.device
    num_atoms = coords_tensor.size(1)
    train_ds, test_ds = get_train_test_split(coords_tensor, training_split, split_seed)
    tr_idx = train_ds.indices
    te_idx = test_ds.indices
    if hasattr(tr_idx, "tolist"):
        tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
    train_coords_np = coords_np[tr_idx]
    test_coords_np = coords_np[te_idx]
    if max_train is not None:
        train_coords_np = train_coords_np[:max_train]
    if max_test is not None:
        test_coords_np = test_coords_np[:max_test]
    train_coords = torch.from_numpy(train_coords_np).float().to(device)
    test_coords = torch.from_numpy(test_coords_np).float().to(device)
    train_feats_full = _feats_from_coords(train_coords, device, num_atoms, batch_size)
    test_feats_full = _feats_from_coords(test_coords, device, num_atoms, batch_size)
    tr_idx_fps = _fps_subsample(train_feats_full, n_subsample, seed=fps_seed)
    te_idx_fps = _fps_subsample(test_feats_full, n_subsample, seed=fps_seed + 1)
    train_feats = train_feats_full[tr_idx_fps]
    test_feats = test_feats_full[te_idx_fps]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, train_feats=train_feats, test_feats=test_feats)
    if display_root is not None:
        print(f"  Saved seed-level distmap clustering feats cache: {display_path(cache_path, display_root)}")
    return cache_path, train_coords_np, test_coords_np


def get_or_compute_coord_clustering_feats(
    cache_path: str,
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    n_subsample: int,
    fps_seed: int = FPS_SEED,
    display_root: str | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Load or compute train/test subsampled coordinate-aligned feats; save to cache_path.
    Returns (cache_path, train_coords_np, test_coords_np).
    Features = Kabsch-align to first structure, flatten; pairwise RMSE = pairwise RMSD.
    max_train/max_test cap the reference set sizes (None = use all).
    """
    if os.path.isfile(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            train_feats = np.asarray(loaded["train_feats"], dtype=np.float32)
            test_feats = np.asarray(loaded["test_feats"], dtype=np.float32)
            loaded.close()
            coords = coords_tensor
            train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
            tr_idx = train_ds.indices
            te_idx = test_ds.indices
            if hasattr(tr_idx, "tolist"):
                tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
            train_coords_np = coords_np[tr_idx]
            test_coords_np = coords_np[te_idx]
            if max_train is not None:
                train_coords_np = train_coords_np[:max_train]
            if max_test is not None:
                test_coords_np = test_coords_np[:max_test]
            if display_root is not None:
                print(f"  Loaded seed-level coord clustering feats cache: {display_path(cache_path, display_root)}")
            return cache_path, train_coords_np, test_coords_np
        except Exception:
            pass
    train_ds, test_ds = get_train_test_split(coords_tensor, training_split, split_seed)
    tr_idx = train_ds.indices
    te_idx = test_ds.indices
    if hasattr(tr_idx, "tolist"):
        tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
    train_coords_np = coords_np[tr_idx]
    test_coords_np = coords_np[te_idx]
    if max_train is not None:
        train_coords_np = train_coords_np[:max_train]
    if max_test is not None:
        test_coords_np = test_coords_np[:max_test]
    train_feats_full = _feats_from_coords_aligned(train_coords_np)
    test_feats_full = _feats_from_coords_aligned(test_coords_np)
    tr_idx_fps = _fps_subsample(train_feats_full, n_subsample, seed=fps_seed)
    te_idx_fps = _fps_subsample(test_feats_full, n_subsample, seed=fps_seed + 1)
    train_feats = train_feats_full[tr_idx_fps]
    test_feats = test_feats_full[te_idx_fps]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, train_feats=train_feats, test_feats=test_feats)
    if display_root is not None:
        print(f"  Saved seed-level coord clustering feats cache: {display_path(cache_path, display_root)}")
    return cache_path, train_coords_np, test_coords_np


def _load_clustering_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load train_feats, test_feats from cache. Raises if missing."""
    loaded = np.load(cache_path, allow_pickle=False)
    train_feats = np.asarray(loaded["train_feats"], dtype=np.float32)
    test_feats = np.asarray(loaded["test_feats"], dtype=np.float32)
    loaded.close()
    return train_feats, test_feats


def _plot_panel(
    ax,
    Z,
    labels,
    title: str,
    cophenetic_r: float,
    source_colors: dict,
    leaf_colors: np.ndarray | None = None,
    show_cophenetic: bool = True,
) -> None:
    """Draw one dendrogram panel with optional leaf colour strip."""
    N = len(labels)
    dn = dendrogram(
        Z, ax=ax,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#aaaaaa",
        link_color_func=lambda _k: "#aaaaaa",
    )
    if leaf_colors is not None:
        y_bottom = ax.get_ylim()[0]
        leaf_x = np.arange(5, 5 + 10 * N, 10, dtype=float)
        strip_h = (ax.get_ylim()[1] - y_bottom) * 0.035
        for pos, leaf_idx in enumerate(dn["leaves"]):
            c = leaf_colors[leaf_idx]
            ax.add_patch(mpatches.Rectangle(
                (leaf_x[pos] - 5, y_bottom - strip_h), 10, strip_h,
                color=c, clip_on=False, linewidth=0,
            ))
        ax.set_ylim(y_bottom - strip_h, ax.get_ylim()[1])
    suffix = f"  (c={cophenetic_r:.3f})" if show_cophenetic else ""
    ax.set_title(f"{title}{suffix}", fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel("RMSE (distance-map)", fontsize=9)
    ax.tick_params(axis="x", bottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _fig_pure_dendrograms(
    sub_feats: dict,
    source_order: list,
    source_colors: dict,
    output_path: str,
    plot_dpi: int,
    save_pdf_copy: bool,
    display_root: str | None,
) -> None:
    """Pure-population dendrograms (one per source)."""
    groups = [(name, sub_feats[name]) for name in source_order if name in sub_feats]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(8 * len(groups), 7))
    if len(groups) == 1:
        axes = [axes]
    fig.suptitle(
        "Hierarchical Clustering — Individual Populations\n"
        "Each leaf is one structure; distance = distance-map RMSE",
        fontsize=13, fontweight="bold", y=1.02,
    )
    for ax, (name, feats) in zip(axes, groups):
        Z, c = _compute_linkage_and_cophenetic(feats)
        color = source_colors[name]
        leaf_colors = np.array([color] * len(feats))
        _plot_panel(ax, Z, np.arange(len(feats)), name, c, source_colors, leaf_colors=leaf_colors)
        patch = mpatches.Patch(color=color, label=f"{name}  (n={len(feats)})")
        ax.legend(handles=[patch], fontsize=9, loc="upper right")
    fig.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    if save_pdf_copy:
        _save_pdf_copy(plt.gcf(), output_path, save_pdf=True, display_root=display_root)
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


def _mixed_dendrogram_panel(
    ax,
    feats_a: np.ndarray,
    name_a: str,
    feats_b: np.ndarray,
    name_b: str,
    feats_c: np.ndarray | None,
    name_c: str | None,
    source_colors: dict,
    k_mixing: int,
    n_clusters: int,
    linkage_method: str,
) -> tuple:
    """One mixed dendrogram; returns (obs_mix, exp_mix, norm_mix, Z, labels)."""
    parts = [(name_a, feats_a), (name_b, feats_b)]
    if feats_c is not None and name_c is not None:
        parts.append((name_c, feats_c))
    stacked = np.concatenate([f for _, f in parts], axis=0)
    sizes = [len(f) for _, f in parts]
    names = [n for n, _ in parts]
    labels = _source_labels_array(sizes, names)
    leaf_colors = np.concatenate([np.array([source_colors[n]] * sz) for n, sz in zip(names, sizes)])
    Z, c = _compute_linkage_and_cophenetic(stacked, method=linkage_method)
    obs_mix, _ = _mixing_score(stacked, labels, k=k_mixing)
    exp_mix = _expected_mixing(labels, k=k_mixing)
    norm_mix = obs_mix / exp_mix if exp_mix > 0 else 0.0
    title_parts = " + ".join(names)
    _plot_panel(ax, Z, labels, title_parts, c, source_colors, leaf_colors=leaf_colors)
    ax.text(0.5, -0.04, f"mixing={obs_mix:.2f} (expected={exp_mix:.2f}, ratio={norm_mix:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, style="italic", color="#555555")
    return obs_mix, exp_mix, norm_mix, Z, labels


def _fig_mixed_dendrograms(
    sub_feats: dict,
    source_order: list,
    source_colors: dict,
    output_path: str,
    plot_dpi: int,
    save_pdf_copy: bool,
    display_root: str | None,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
) -> dict:
    """Mixed dendrograms 2x2 (gen: Train+Test, Train+Gen, Gen+Test, Train+Gen+Test). Returns mixing_stats."""
    tr = sub_feats.get("Training")
    ge = sub_feats.get("Generated")
    te = sub_feats.get("Test")
    if ge is None:
        ge = sub_feats.get("Train recon")
        te2 = sub_feats.get("Test recon")
        if te is not None and te2 is not None:
            configs = [
                ("Training", tr, "Test", te, None, None),
                ("Training", tr, "Train recon", ge, None, None),
                ("Test", te, "Test recon", te2, None, None),
                ("Training", tr, "Test", te, "Train recon", ge),
            ]
            if tr is None:
                configs = [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in configs if c[1] is not None and c[3] is not None]
        else:
            configs = []
    else:
        configs = [
            ("Training", tr, "Test", te, None, None),
            ("Training", tr, "Generated", ge, None, None),
            ("Generated", ge, "Test", te, None, None),
            ("Training", tr, "Generated", ge, "Test", te),
        ]
    if not configs:
        return {}
    n_panels = len(configs)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 7 * n_rows))
    axes = np.atleast_2d(axes)
    stats = {}
    for idx, (na, fa, nb, fb, nc, fc) in enumerate(configs):
        if fa is None or fb is None:
            continue
        ax = axes.flat[idx]
        obs, exp, ratio, Z_mix, lbl_mix = _mixed_dendrogram_panel(
            ax, fa, na, fb, nb, fc, nc, source_colors, k_mixing, n_clusters, linkage_method,
        )
        key = f"{na}+{nb}" + (f"+{nc}" if nc else "")
        stats[key] = {"obs": obs, "exp": exp, "ratio": ratio}
    patches = [mpatches.Patch(color=source_colors[s], label=s) for s in source_order if s in sub_feats]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=min(4, len(patches)), fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Hierarchical Clustering — Mixed-Source Dendrograms\n"
        "Leaf strip colour = source; structures should interleave if model captures same landscape",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    if save_pdf_copy:
        _save_pdf_copy(plt.gcf(), output_path, save_pdf=True, display_root=display_root)
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")
    return stats


def _fig_mixing_analysis(
    sub_feats: dict,
    mixing_stats: dict,
    source_order: list,
    source_colors: dict,
    output_path: str,
    plot_dpi: int,
    save_pdf_copy: bool,
    display_root: str | None,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
) -> None:
    """Bar chart of mixing scores + cluster composition heatmaps."""
    keys = list(mixing_stats.keys())
    if not keys:
        return
    obs = [mixing_stats[k]["obs"] for k in keys]
    exp = [mixing_stats[k]["exp"] for k in keys]
    ratio = [mixing_stats[k]["ratio"] for k in keys]
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(len(keys))
    w = 0.35
    ax_bar.bar(x - w / 2, obs, w, label="Observed mixing", color="#4878d0", alpha=0.8)
    ax_bar.bar(x + w / 2, exp, w, label="Expected (random)", color="#aaaaaa", alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([k.replace("+", " + ") for k in keys], fontsize=10)
    ax_bar.set_ylabel("Source mixing score", fontsize=10)
    ax_bar.set_title(f"Source Mixing Scores  (k={k_mixing} nearest neighbours)", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.3)
    for xi, (o, e, r) in enumerate(zip(obs, exp, ratio)):
        ax_bar.text(xi, max(o, e) + 0.02, f"ratio={r:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")
    pair_configs = []
    for k in keys:
        parts = k.split("+")
        if len(parts) == 2 and parts[0] in sub_feats and parts[1] in sub_feats:
            pair_configs.append((parts[0], sub_feats[parts[0]], parts[1], sub_feats[parts[1]]))
    for i, ax_c in enumerate([fig.add_subplot(gs[1, i]) for i in range(3)]):
        if i >= len(pair_configs):
            ax_c.set_visible(False)
            continue
        na, fa, nb, fb = pair_configs[i]
        stacked = np.concatenate([fa, fb], axis=0)
        sizes = [len(fa), len(fb)]
        names = [na, nb]
        labels = _source_labels_array(sizes, names)
        D = _pairwise_rmse(stacked)
        cond = squareform(D, checks=False)
        Z = linkage(cond, method=linkage_method)
        comp, sources = _cluster_source_composition(Z, labels, n_clusters)
        row_sums = comp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        frac = comp / row_sums
        bot = np.zeros(n_clusters)
        for si, src in enumerate(sources):
            ax_c.bar(np.arange(n_clusters), frac[:, si], bottom=bot, color=source_colors.get(src, "#888888"), label=src, alpha=0.85)
            bot += frac[:, si]
        ax_c.set_xticks(np.arange(n_clusters))
        ax_c.set_xticklabels([f"C{j+1}" for j in range(n_clusters)], fontsize=8)
        ax_c.set_ylabel("Source fraction", fontsize=9)
        ax_c.set_ylim(0, 1.05)
        ax_c.set_title(f"{na} + {nb} — Cluster Composition", fontsize=9, fontweight="bold")
        ax_c.legend(fontsize=8, loc="upper right")
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)
    fig.suptitle("Quantitative Mixing Analysis", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    if save_pdf_copy:
        _save_pdf_copy(plt.gcf(), output_path, save_pdf=True, display_root=display_root)
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


def _fig_rmse_similarity(
    sub_feats: dict,
    source_order: list,
    output_path: str,
    plot_dpi: int,
    save_pdf_copy: bool,
    display_root: str | None,
) -> None:
    """Quantile-quantile plots of pairwise RMSE distributions between pairs of sources."""
    groups = [(name, sub_feats[name]) for name in source_order if name in sub_feats]
    n = len(groups)
    if n < 2:
        return
    n_pairs = n * (n - 1) // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 7))
    if n_pairs == 1:
        axes = [axes]
    panel = 0
    for i in range(n):
        for j in range(i + 1, n):
            name_a, feats_a = groups[i]
            name_b, feats_b = groups[j]
            ax = axes[panel]
            panel += 1
            Da = _pairwise_rmse(feats_a)
            Db = _pairwise_rmse(feats_b)
            tri_a = Da[np.triu_indices(len(feats_a), k=1)]
            tri_b = Db[np.triu_indices(len(feats_b), k=1)]
            n_q = min(500, len(tri_a), len(tri_b))
            qs = np.linspace(0, 100, n_q)
            qa = np.percentile(tri_a, qs)
            qb = np.percentile(tri_b, qs)
            corr = float(np.corrcoef(qa, qb)[0, 1])
            ax.scatter(qa, qb, s=10, alpha=0.6, c=np.linspace(0, 1, n_q), cmap="viridis")
            lim = max(qa.max(), qb.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.6, label="y=x")
            ax.set_xlabel(f"{name_a} pairwise RMSE (quantiles)", fontsize=10)
            ax.set_ylabel(f"{name_b} pairwise RMSE (quantiles)", fontsize=10)
            ax.set_title(f"{name_a} vs {name_b}   (Pearson r={corr:.3f})", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Distance-Landscape Similarity Between Populations", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    if save_pdf_copy:
        _save_pdf_copy(plt.gcf(), output_path, save_pdf=True, display_root=display_root)
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


def _write_clustering_figures(
    run_dir_this: str,
    sub_feats: dict,
    source_order: list,
    source_colors: dict,
    plot_cfg: dict,
    display_root: str | None,
    k_mixing: int,
    n_clusters: int,
    linkage_method: str,
    n_subsample: int,
) -> str:
    """Write pure_dendrograms, mixed_dendrograms, mixing_analysis, rmse_similarity. Optionally save data/ when save_data. Returns path to primary figure."""
    os.makedirs(run_dir_this, exist_ok=True)
    dpi = plot_cfg.get("plot_dpi", 150)
    save_pdf = plot_cfg.get("save_pdf_copy", False)
    save_data = plot_cfg.get("save_data", False)
    pure_path = os.path.join(run_dir_this, "pure_dendrograms.png")
    mixed_path = os.path.join(run_dir_this, "mixed_dendrograms.png")
    mixing_path = os.path.join(run_dir_this, "mixing_analysis.png")
    rmse_path = os.path.join(run_dir_this, "rmse_similarity.png")
    _fig_pure_dendrograms(sub_feats, source_order, source_colors, pure_path, dpi, save_pdf, display_root)
    mixing_stats = _fig_mixed_dendrograms(sub_feats, source_order, source_colors, mixed_path, dpi, save_pdf, display_root, k_mixing, n_clusters, linkage_method)
    _fig_mixing_analysis(sub_feats, mixing_stats, source_order, source_colors, mixing_path, dpi, save_pdf, display_root, k_mixing, n_clusters, linkage_method)
    _fig_rmse_similarity(sub_feats, source_order, rmse_path, dpi, save_pdf, display_root)
    if save_data:
        data_dir = os.path.join(run_dir_this, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "clustering_data.npz")
        out = {}
        for name in source_order:
            if name in sub_feats:
                key = name.lower().replace(" ", "_") + "_feats"
                out[key] = sub_feats[name]
        out["n_subsample"] = np.array(n_subsample, dtype=np.int64)
        out["k_mixing"] = np.array(k_mixing, dtype=np.int64)
        out["n_clusters"] = np.array(n_clusters, dtype=np.int64)
        out["linkage_method"] = np.array(linkage_method, dtype=object)
        if mixing_stats:
            keys = list(mixing_stats.keys())
            out["mixing_keys"] = np.array(keys, dtype=object)
            out["mixing_obs"] = np.array([mixing_stats[k]["obs"] for k in keys], dtype=np.float64)
            out["mixing_exp"] = np.array([mixing_stats[k]["exp"] for k in keys], dtype=np.float64)
            out["mixing_ratio"] = np.array([mixing_stats[k]["ratio"] for k in keys], dtype=np.float64)
        np.savez_compressed(data_path, **out)
        print(f"  Saved: {display_path(data_path, display_root)}")
    return mixed_path


def run_distmap_clustering_gen_analysis(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    frozen_vae,
    embed,
    latent_dim: int,
    device: torch.device,
    run_dir: str,
    plot_cfg: dict,
    *,
    num_samples: int,
    sample_variance: float,
    output_suffix: str = "",
    display_root: str | None = None,
    clustering_seed_feats_path: str | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
    feats_batch_size: int = 64,
) -> str:
    """
    Distmap clustering gen: load train/test feats from cache, generate structures, subsample gen feats, produce four figures.
    Saves to run_dir/analysis/distmap_clustering/gen/<run_name>/.
    """
    from .distmap.sample import generate_samples

    run_name = output_suffix.lstrip("_") if output_suffix else "default"
    run_dir_this = os.path.join(run_dir, "analysis", "distmap_clustering", "gen", run_name)
    if clustering_seed_feats_path and os.path.isfile(clustering_seed_feats_path):
        train_feats, test_feats = _load_clustering_cache(clustering_seed_feats_path)
    else:
        raise ValueError("distmap_clustering_gen requires clustering_seed_feats_path (seed-level cache).")
    num_atoms = coords_tensor.size(1)
    embed.eval()
    with torch.no_grad():
        z = generate_samples(num_samples, latent_dim, device, variance=sample_variance)
        D_ne = frozen_vae._decode_to_matrix(z)
        gen_coords = embed(D_ne)
    gen_feats_full = _feats_from_coords(gen_coords, device, num_atoms, feats_batch_size)
    ge_idx = _fps_subsample(gen_feats_full, n_subsample, seed=FPS_SEED + 2)
    gen_feats = gen_feats_full[ge_idx]
    sub_feats = {"Training": train_feats, "Test": test_feats, "Generated": gen_feats}
    return _write_clustering_figures(
        run_dir_this, sub_feats, SOURCE_ORDER_GEN, SOURCE_COLORS_GEN,
        plot_cfg, display_root, k_mixing, n_clusters, linkage_method, n_subsample,
    )


def run_distmap_clustering_gen_analysis_multi(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    frozen_vae,
    embed,
    latent_dim: int,
    device: torch.device,
    run_dir: str,
    plot_cfg: dict,
    *,
    num_samples_list: list[int],
    sample_variance: float,
    variance_suffix: str = "",
    display_root: str | None = None,
    clustering_seed_feats_path: str | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
    feats_batch_size: int = 64,
) -> list[str]:
    """Run distmap clustering gen for multiple num_samples with same variance."""
    from .distmap.sample import generate_samples

    if not clustering_seed_feats_path or not os.path.isfile(clustering_seed_feats_path):
        raise ValueError("distmap_clustering_gen multi requires clustering_seed_feats_path.")
    train_feats, test_feats = _load_clustering_cache(clustering_seed_feats_path)
    num_atoms = coords_tensor.size(1)
    embed.eval()
    sorted_n = sorted(set(num_samples_list))
    out_paths = []
    for n in sorted_n:
        with torch.no_grad():
            z = generate_samples(n, latent_dim, device, variance=sample_variance)
            D_ne = frozen_vae._decode_to_matrix(z)
            gen_coords = embed(D_ne)
        gen_feats_full = _feats_from_coords(gen_coords, device, num_atoms, feats_batch_size)
        ge_idx = _fps_subsample(gen_feats_full, n_subsample, seed=FPS_SEED + 2)
        gen_feats = gen_feats_full[ge_idx]
        sub_feats = {"Training": train_feats, "Test": test_feats, "Generated": gen_feats}
        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_this = os.path.join(run_dir, "analysis", "distmap_clustering", "gen", run_name)
        path = _write_clustering_figures(
            run_dir_this, sub_feats, SOURCE_ORDER_GEN, SOURCE_COLORS_GEN,
            plot_cfg, display_root, k_mixing, n_clusters, linkage_method, n_subsample,
        )
        out_paths.append(path)
    return out_paths


def run_distmap_clustering_recon_analysis(
    clustering_seed_feats_path: str,
    train_coords_np: np.ndarray,
    test_coords_np: np.ndarray,
    train_recon_coords: np.ndarray,
    test_recon_coords: np.ndarray,
    run_dir: str,
    plot_cfg: dict,
    *,
    display_root: str | None = None,
    recon_subdir: str = "",
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
    feats_batch_size: int = 64,
    device: torch.device | None = None,
) -> str:
    """
    Distmap clustering recon: use cached train/test feats; compute train_recon and test_recon feats, FPS subsample; four populations.
    Saves to run_dir/analysis/distmap_clustering/recon[/recon_subdir]/.
    """
    train_feats, test_feats = _load_clustering_cache(clustering_seed_feats_path)
    if device is None:
        device = utils.get_device()
    num_atoms = train_coords_np.shape[1]
    train_recon_t = torch.from_numpy(train_recon_coords).float().to(device)
    test_recon_t = torch.from_numpy(test_recon_coords).float().to(device)
    train_recon_feats_full = _feats_from_coords(train_recon_t, device, num_atoms, feats_batch_size)
    test_recon_feats_full = _feats_from_coords(test_recon_t, device, num_atoms, feats_batch_size)
    tr_recon_idx = _fps_subsample(train_recon_feats_full, n_subsample, seed=FPS_SEED + 3)
    te_recon_idx = _fps_subsample(test_recon_feats_full, n_subsample, seed=FPS_SEED + 4)
    train_recon_feats = train_recon_feats_full[tr_recon_idx]
    test_recon_feats = test_recon_feats_full[te_recon_idx]
    sub_feats = {
        "Training": train_feats,
        "Train recon": train_recon_feats,
        "Test": test_feats,
        "Test recon": test_recon_feats,
    }
    run_dir_recon = os.path.join(run_dir, "analysis", "distmap_clustering", "recon", recon_subdir) if recon_subdir else os.path.join(run_dir, "analysis", "distmap_clustering", "recon")
    return _write_clustering_figures(
        run_dir_recon, sub_feats, SOURCE_ORDER_RECON, SOURCE_COLORS_RECON,
        plot_cfg, display_root, k_mixing, n_clusters, linkage_method, n_subsample,
    )


def run_coord_clustering_gen_analysis(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    frozen_vae,
    embed,
    latent_dim: int,
    device: torch.device,
    run_dir: str,
    plot_cfg: dict,
    *,
    num_samples: int,
    sample_variance: float,
    output_suffix: str = "",
    display_root: str | None = None,
    coord_clustering_seed_feats_path: str | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
) -> str:
    """
    Coord clustering gen: load train/test coord feats from cache, generate structures, subsample gen feats, produce four figures.
    Saves to run_dir/analysis/coord_clustering/gen/<run_name>/.
    """
    from .distmap.sample import generate_samples

    run_name = output_suffix.lstrip("_") if output_suffix else "default"
    run_dir_this = os.path.join(run_dir, "analysis", "coord_clustering", "gen", run_name)
    if coord_clustering_seed_feats_path and os.path.isfile(coord_clustering_seed_feats_path):
        train_feats, test_feats = _load_clustering_cache(coord_clustering_seed_feats_path)
    else:
        raise ValueError("coord_clustering_gen requires coord_clustering_seed_feats_path (seed-level cache).")
    num_atoms = coords_tensor.size(1)
    embed.eval()
    with torch.no_grad():
        z = generate_samples(num_samples, latent_dim, device, variance=sample_variance)
        D_ne = frozen_vae._decode_to_matrix(z)
        gen_coords = embed(D_ne)
    gen_coords_np = gen_coords.cpu().numpy()
    gen_feats_full = _feats_from_coords_aligned(gen_coords_np)
    ge_idx = _fps_subsample(gen_feats_full, n_subsample, seed=FPS_SEED + 2)
    gen_feats = gen_feats_full[ge_idx]
    sub_feats = {"Training": train_feats, "Test": test_feats, "Generated": gen_feats}
    return _write_clustering_figures(
        run_dir_this, sub_feats, SOURCE_ORDER_GEN, SOURCE_COLORS_GEN,
        plot_cfg, display_root, k_mixing, n_clusters, linkage_method, n_subsample,
    )


def run_coord_clustering_gen_analysis_multi(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    frozen_vae,
    embed,
    latent_dim: int,
    device: torch.device,
    run_dir: str,
    plot_cfg: dict,
    *,
    num_samples_list: list[int],
    sample_variance: float,
    variance_suffix: str = "",
    display_root: str | None = None,
    coord_clustering_seed_feats_path: str | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
) -> list[str]:
    """Run coord clustering gen for multiple num_samples with same variance."""
    from .distmap.sample import generate_samples

    if not coord_clustering_seed_feats_path or not os.path.isfile(coord_clustering_seed_feats_path):
        raise ValueError("coord_clustering_gen multi requires coord_clustering_seed_feats_path.")
    train_feats, test_feats = _load_clustering_cache(coord_clustering_seed_feats_path)
    num_atoms = coords_tensor.size(1)
    embed.eval()
    sorted_n = sorted(set(num_samples_list))
    out_paths = []
    for n in sorted_n:
        with torch.no_grad():
            z = generate_samples(n, latent_dim, device, variance=sample_variance)
            D_ne = frozen_vae._decode_to_matrix(z)
            gen_coords = embed(D_ne)
        gen_coords_np = gen_coords.cpu().numpy()
        gen_feats_full = _feats_from_coords_aligned(gen_coords_np)
        ge_idx = _fps_subsample(gen_feats_full, n_subsample, seed=FPS_SEED + 2)
        gen_feats = gen_feats_full[ge_idx]
        sub_feats = {"Training": train_feats, "Test": test_feats, "Generated": gen_feats}
        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_this = os.path.join(run_dir, "analysis", "coord_clustering", "gen", run_name)
        path = _write_clustering_figures(
            run_dir_this, sub_feats, SOURCE_ORDER_GEN, SOURCE_COLORS_GEN,
            plot_cfg, display_root, k_mixing, n_clusters, linkage_method, n_subsample,
        )
        out_paths.append(path)
    return out_paths


def run_coord_clustering_recon_analysis(
    coord_clustering_seed_feats_path: str,
    train_coords_np: np.ndarray,
    test_coords_np: np.ndarray,
    train_recon_coords: np.ndarray,
    test_recon_coords: np.ndarray,
    run_dir: str,
    plot_cfg: dict,
    *,
    display_root: str | None = None,
    recon_subdir: str = "",
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    k_mixing: int = DEFAULT_K_MIXING,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = LINKAGE_METHOD,
) -> str:
    """
    Coord clustering recon: use cached train/test coord feats; compute train_recon and test_recon aligned feats, FPS subsample; four populations.
    Saves to run_dir/analysis/coord_clustering/recon[/recon_subdir]/.
    """
    train_feats, test_feats = _load_clustering_cache(coord_clustering_seed_feats_path)
    train_recon_feats_full = _feats_from_coords_aligned(train_recon_coords)
    test_recon_feats_full = _feats_from_coords_aligned(test_recon_coords)
    tr_recon_idx = _fps_subsample(train_recon_feats_full, n_subsample, seed=FPS_SEED + 3)
    te_recon_idx = _fps_subsample(test_recon_feats_full, n_subsample, seed=FPS_SEED + 4)
    train_recon_feats = train_recon_feats_full[tr_recon_idx]
    test_recon_feats = test_recon_feats_full[te_recon_idx]
    sub_feats = {
        "Training": train_feats,
        "Train recon": train_recon_feats,
        "Test": test_feats,
        "Test recon": test_recon_feats,
    }
    run_dir_recon = os.path.join(run_dir, "analysis", "coord_clustering", "recon", recon_subdir) if recon_subdir else os.path.join(run_dir, "analysis", "coord_clustering", "recon")
    return _write_clustering_figures(
        run_dir_recon, sub_feats, SOURCE_ORDER_RECON, SOURCE_COLORS_RECON,
        plot_cfg, display_root, k_mixing, n_clusters, linkage_method, n_subsample,
    )
