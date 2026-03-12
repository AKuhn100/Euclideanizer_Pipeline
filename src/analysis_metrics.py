"""
Analysis metric registry: pluggable metrics (rmsd, q) with a common interface.
Used by run.py to drive a single analysis loop over registered metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import rmsd
from . import q_analysis
from . import clustering


def _rmsd_cache_filename(analysis_cfg: dict, max_train: int | None = None, max_test: int | None = None) -> str:
    mt = max_train if max_train is not None else analysis_cfg["rmsd_max_train"]
    mc = max_test if max_test is not None else analysis_cfg["rmsd_max_test"]
    if mt is None and mc is None:
        return "test_to_train_rmsd.npz"
    return f"test_to_train_rmsd_{mt if mt is not None else 'all'}_{mc if mc is not None else 'all'}.npz"


def _rmsd_kwargs_for_cache(analysis_cfg: dict, max_train: int | None = None, max_test: int | None = None) -> dict:
    gen = analysis_cfg["rmsd_gen"]
    mt = max_train if max_train is not None else analysis_cfg["rmsd_max_train"]
    mc = max_test if max_test is not None else analysis_cfg["rmsd_max_test"]
    out = {"query_batch_size": gen["query_batch_size"]}
    if mt is not None:
        out["max_train"] = mt
    if mc is not None:
        out["max_test"] = mc
    return out


def _rmsd_get_or_compute(
    cache_path: str,
    coords_np,
    coords_tensor,
    training_split: float,
    split_seed: int,
    display_root: str | None,
    **kwargs: Any,
):
    return rmsd.get_or_compute_test_to_train_rmsd(
        coords_np, coords_tensor, training_split, split_seed, cache_path,
        query_batch_size=kwargs["query_batch_size"],
        display_root=display_root,
        max_train=kwargs.get("max_train"),
        max_test=kwargs.get("max_test"),
    )


def _rmsd_build_gen_plot_cfg(analysis_cfg: dict, plot_dpi: int) -> dict:
    gen = analysis_cfg["rmsd_gen"]
    return {
        "plot_dpi": plot_dpi,
        "save_pdf_copy": gen["save_pdf_copy"],
        "rmsd_num_samples": gen["num_samples"],
        "rmsd_sample_variance": gen["sample_variance"],
        "rmsd_query_batch_size": gen["query_batch_size"],
        "gen_decode_batch_size": gen["gen_decode_batch_size"],
        "save_data": gen["save_data"],
        "save_structures_gro": gen["save_structures_gro"],
        "save_gen_coords_in_npz": False,
    }


def _rmsd_build_recon_plot_cfg(analysis_cfg: dict, plot_dpi: int) -> dict:
    recon = analysis_cfg["rmsd_recon"]
    return {
        "save_data": recon["save_data"],
        "plot_dpi": plot_dpi,
        "save_pdf_copy": recon["save_pdf_copy"],
    }


def _q_cache_filename(analysis_cfg: dict, max_train: int | None = None, max_test: int | None = None) -> str:
    mt = max_train if max_train is not None else analysis_cfg["q_max_train"]
    mc = max_test if max_test is not None else analysis_cfg["q_max_test"]
    if mt is None and mc is None:
        return "q_test_to_train_500_200.npz"
    return f"q_test_to_train_{mt if mt is not None else 'all'}_{mc if mc is not None else 'all'}.npz"


def _q_kwargs_for_cache(analysis_cfg: dict, max_train: int | None = None, max_test: int | None = None) -> dict:
    gen = analysis_cfg["q_gen"]
    recon = analysis_cfg["q_recon"]
    mt = max_train if max_train is not None else analysis_cfg["q_max_train"]
    mc = max_test if max_test is not None else analysis_cfg["q_max_test"]
    return {
        "max_train": mt if mt is not None else 500,
        "max_test": mc if mc is not None else 200,
        "delta": recon["delta"],
        "query_batch_size": gen["query_batch_size"],
    }


def _q_get_or_compute(
    cache_path: str,
    coords_np,
    coords_tensor,
    training_split: float,
    split_seed: int,
    display_root: str | None,
    **kwargs: Any,
):
    return q_analysis.get_or_compute_test_to_train_q(
        coords_np, coords_tensor, training_split, split_seed, cache_path,
        max_train=kwargs["max_train"],
        max_test=kwargs["max_test"],
        delta=kwargs["delta"],
        query_batch_size=kwargs["query_batch_size"],
        display_root=display_root,
    )


def _q_build_gen_plot_cfg(analysis_cfg: dict, plot_dpi: int) -> dict:
    gen = analysis_cfg["q_gen"]
    return {
        "plot_dpi": plot_dpi,
        "save_pdf_copy": gen["save_pdf_copy"],
        "q_num_samples": gen["num_samples"],
        "q_sample_variance": gen["sample_variance"],
        "q_query_batch_size": gen["query_batch_size"],
        "gen_decode_batch_size": gen["gen_decode_batch_size"],
        "save_data": gen["save_data"],
        "save_structures_gro": gen["save_structures_gro"],
        "save_gen_coords_in_npz": False,
    }


def _q_build_recon_plot_cfg(analysis_cfg: dict, plot_dpi: int) -> dict:
    recon = analysis_cfg["q_recon"]
    return {
        "save_data": recon["save_data"],
        "plot_dpi": plot_dpi,
        "save_pdf_copy": recon["save_pdf_copy"],
    }


def _rmsd_gen_extra_kwargs(analysis_cfg: dict) -> dict:
    return {}


def _rmsd_recon_extra_kwargs(analysis_cfg: dict) -> dict:
    return {}


def _rmsd_precomputed_kwargs(tt, train_c, test_c):
    return {"precomputed_test_to_train": tt, "train_coords_np": train_c, "test_coords_np": test_c}


def _q_gen_extra_kwargs(analysis_cfg: dict) -> dict:
    gen = analysis_cfg["q_gen"]
    return {"delta": gen["delta"]}


def _q_recon_extra_kwargs(analysis_cfg: dict) -> dict:
    recon = analysis_cfg["q_recon"]
    return {"delta": recon["delta"]}


def _q_precomputed_kwargs(tt, train_c, test_c):
    return {"precomputed_test_to_train_max_q": tt, "train_coords_np": train_c, "test_coords_np": test_c}


def _make_clustering_cache_filename(prefix: str):
    """Returns cache_filename(analysis_cfg, max_train, max_test) for coord or distmap clustering. Config must contain all required keys."""

    def _fn(analysis_cfg: dict, max_train: int | None = None, max_test: int | None = None) -> str:
        gen = analysis_cfg[f"{prefix}_gen"]
        n = gen["n_subsample"]
        mt = max_train if max_train is not None else analysis_cfg[f"{prefix}_max_train"]
        mc = max_test if max_test is not None else analysis_cfg[f"{prefix}_max_test"]
        if mt is None and mc is None:
            return f"{prefix}_train_test_feats_n{n}.npz"
        return f"{prefix}_train_test_feats_n{n}_{mt if mt is not None else 'all'}_{mc if mc is not None else 'all'}.npz"

    return _fn


def _make_clustering_kwargs_for_cache(prefix: str, include_batch_size: bool = False):
    """Returns kwargs_for_cache(analysis_cfg, max_train, max_test). Config must contain all required keys."""

    def _fn(analysis_cfg: dict, max_train: int | None = None, max_test: int | None = None) -> dict:
        gen = analysis_cfg[f"{prefix}_gen"]
        mt = max_train if max_train is not None else analysis_cfg[f"{prefix}_max_train"]
        mc = max_test if max_test is not None else analysis_cfg[f"{prefix}_max_test"]
        out = {
            "n_subsample": gen["n_subsample"],
            "fps_seed": clustering.FPS_SEED,
        }
        if include_batch_size:
            out["batch_size"] = gen["feats_batch_size"]
        if mt is not None:
            out["max_train"] = mt
        if mc is not None:
            out["max_test"] = mc
        return out

    return _fn


def _make_clustering_get_or_compute(use_batch_size: bool):
    """Returns get_or_compute wrapper calling coord or distmap clustering feats. Kwargs come from config via kwargs_for_cache."""

    def _fn(
        cache_path: str,
        coords_np,
        coords_tensor,
        training_split: float,
        split_seed: int,
        display_root: str | None,
        **kwargs: Any,
    ):
        if use_batch_size:
            return clustering.get_or_compute_distmap_clustering_feats(
                cache_path, coords_np, coords_tensor, training_split, split_seed,
                n_subsample=kwargs["n_subsample"],
                batch_size=kwargs["batch_size"],
                fps_seed=kwargs["fps_seed"],
                display_root=display_root,
                max_train=kwargs.get("max_train"),
                max_test=kwargs.get("max_test"),
            )
        return clustering.get_or_compute_coord_clustering_feats(
            cache_path, coords_np, coords_tensor, training_split, split_seed,
            n_subsample=kwargs["n_subsample"],
            fps_seed=kwargs["fps_seed"],
            display_root=display_root,
            max_train=kwargs.get("max_train"),
            max_test=kwargs.get("max_test"),
        )

    return _fn


def _make_clustering_build_gen_plot_cfg(prefix: str):
    def _fn(analysis_cfg: dict, plot_dpi: int) -> dict:
        gen = analysis_cfg[f"{prefix}_gen"]
        return {
            "plot_dpi": plot_dpi,
            "save_pdf_copy": gen["save_pdf_copy"],
            "save_data": gen["save_data"],
            "gen_decode_batch_size": gen["gen_decode_batch_size"],
        }
    return _fn


def _make_clustering_build_recon_plot_cfg(prefix: str):
    def _fn(analysis_cfg: dict, plot_dpi: int) -> dict:
        recon = analysis_cfg[f"{prefix}_recon"]
        return {
            "save_data": recon["save_data"],
            "plot_dpi": plot_dpi,
            "save_pdf_copy": recon["save_pdf_copy"],
        }
    return _fn


def _make_clustering_precomputed_kwargs(seed_feats_key: str):
    """seed_feats_key: 'coord_clustering_seed_feats_path' or 'clustering_seed_feats_path'."""

    def _fn(tt, train_c, test_c):
        return {seed_feats_key: tt, "train_coords_np": train_c, "test_coords_np": test_c}
    return _fn


def _make_clustering_gen_extra_kwargs(prefix: str, include_batch_size: bool = False):
    def _fn(analysis_cfg: dict) -> dict:
        gen = analysis_cfg[f"{prefix}_gen"]
        out = {
            "n_subsample": gen["n_subsample"],
            "k_mixing": gen["k_mixing"],
            "linkage_method": gen["linkage_method"],
        }
        if include_batch_size:
            out["feats_batch_size"] = gen["feats_batch_size"]
        return out
    return _fn


def _make_clustering_recon_extra_kwargs(prefix: str, include_batch_size: bool = False):
    def _fn(analysis_cfg: dict) -> dict:
        recon = analysis_cfg[f"{prefix}_recon"]
        out = {
            "n_subsample": recon["n_subsample"],
            "k_mixing": recon["k_mixing"],
            "linkage_method": recon["linkage_method"],
        }
        if include_batch_size:
            out["feats_batch_size"] = recon["feats_batch_size"]
        return out
    return _fn


# Coord clustering: no batch_size in cache or extra kwargs.
_coord_clustering_cache_filename = _make_clustering_cache_filename("coord_clustering")
_coord_clustering_kwargs_for_cache = _make_clustering_kwargs_for_cache("coord_clustering", include_batch_size=False)
_coord_clustering_get_or_compute = _make_clustering_get_or_compute(use_batch_size=False)
_coord_clustering_build_gen_plot_cfg = _make_clustering_build_gen_plot_cfg("coord_clustering")
_coord_clustering_build_recon_plot_cfg = _make_clustering_build_recon_plot_cfg("coord_clustering")
_coord_clustering_precomputed_kwargs = _make_clustering_precomputed_kwargs("coord_clustering_seed_feats_path")
_coord_clustering_gen_extra_kwargs = _make_clustering_gen_extra_kwargs("coord_clustering", include_batch_size=False)
_coord_clustering_recon_extra_kwargs = _make_clustering_recon_extra_kwargs("coord_clustering", include_batch_size=False)

# Distmap clustering: includes feats_batch_size in cache and extra kwargs.
_distmap_clustering_cache_filename = _make_clustering_cache_filename("distmap_clustering")
_distmap_clustering_kwargs_for_cache = _make_clustering_kwargs_for_cache("distmap_clustering", include_batch_size=True)
_distmap_clustering_get_or_compute = _make_clustering_get_or_compute(use_batch_size=True)
_distmap_clustering_build_gen_plot_cfg = _make_clustering_build_gen_plot_cfg("distmap_clustering")
_distmap_clustering_build_recon_plot_cfg = _make_clustering_build_recon_plot_cfg("distmap_clustering")
_distmap_clustering_precomputed_kwargs = _make_clustering_precomputed_kwargs("clustering_seed_feats_path")
_distmap_clustering_gen_extra_kwargs = _make_clustering_gen_extra_kwargs("distmap_clustering", include_batch_size=True)
_distmap_clustering_recon_extra_kwargs = _make_clustering_recon_extra_kwargs("distmap_clustering", include_batch_size=True)


@dataclass(frozen=True)
class AnalysisMetricSpec:
    """Single analysis metric (rmsd or q) for the unified analysis loop."""
    id: str
    gen_key: str
    recon_key: str
    subdir: str
    figure_filename: str
    get_or_compute_test_to_train: Callable[..., tuple[Any, Any, Any]]
    run_gen_analysis: Callable[..., str]
    run_gen_analysis_multi: Callable[..., list[str]]
    run_recon_analysis: Callable[..., str]
    cache_filename: Callable[[dict, int | None, int | None], str]
    kwargs_for_cache: Callable[[dict, int | None, int | None], dict]
    build_gen_plot_cfg: Callable[[dict, int], dict]
    build_recon_plot_cfg: Callable[[dict, int], dict]
    precomputed_kwargs: Callable[[Any, Any, Any], dict]  # (tt, train_coords, test_coords) -> kwargs for run_* and run_recon
    gen_extra_kwargs: Callable[[dict], dict]   # e.g. {"delta": ...} for q; {} for rmsd
    recon_extra_kwargs: Callable[[dict], dict]


# Order: rmsd first, then q (preserves cache and log order).
ANALYSIS_METRICS: list[AnalysisMetricSpec] = [
    AnalysisMetricSpec(
        id="rmsd",
        gen_key="rmsd_gen",
        recon_key="rmsd_recon",
        subdir="rmsd",
        figure_filename="rmsd_distributions.png",
        get_or_compute_test_to_train=_rmsd_get_or_compute,
        run_gen_analysis=rmsd.run_min_rmsd_analysis,
        run_gen_analysis_multi=rmsd.run_min_rmsd_analysis_multi,
        run_recon_analysis=rmsd.run_min_rmsd_recon_analysis,
        cache_filename=_rmsd_cache_filename,
        kwargs_for_cache=_rmsd_kwargs_for_cache,
        build_gen_plot_cfg=_rmsd_build_gen_plot_cfg,
        build_recon_plot_cfg=_rmsd_build_recon_plot_cfg,
        precomputed_kwargs=_rmsd_precomputed_kwargs,
        gen_extra_kwargs=_rmsd_gen_extra_kwargs,
        recon_extra_kwargs=_rmsd_recon_extra_kwargs,
    ),
    AnalysisMetricSpec(
        id="q",
        gen_key="q_gen",
        recon_key="q_recon",
        subdir="q",
        figure_filename="q_distributions.png",
        get_or_compute_test_to_train=_q_get_or_compute,
        run_gen_analysis=q_analysis.run_q_analysis,
        run_gen_analysis_multi=q_analysis.run_q_analysis_multi,
        run_recon_analysis=q_analysis.run_q_recon_analysis,
        cache_filename=_q_cache_filename,
        kwargs_for_cache=_q_kwargs_for_cache,
        build_gen_plot_cfg=_q_build_gen_plot_cfg,
        build_recon_plot_cfg=_q_build_recon_plot_cfg,
        precomputed_kwargs=_q_precomputed_kwargs,
        gen_extra_kwargs=_q_gen_extra_kwargs,
        recon_extra_kwargs=_q_recon_extra_kwargs,
    ),
    AnalysisMetricSpec(
        id="coord_clustering",
        gen_key="coord_clustering_gen",
        recon_key="coord_clustering_recon",
        subdir="coord_clustering",
        figure_filename="mixed_dendrograms.png",
        get_or_compute_test_to_train=_coord_clustering_get_or_compute,
        run_gen_analysis=clustering.run_coord_clustering_gen_analysis,
        run_gen_analysis_multi=clustering.run_coord_clustering_gen_analysis_multi,
        run_recon_analysis=clustering.run_coord_clustering_recon_analysis,
        cache_filename=_coord_clustering_cache_filename,
        kwargs_for_cache=_coord_clustering_kwargs_for_cache,
        build_gen_plot_cfg=_coord_clustering_build_gen_plot_cfg,
        build_recon_plot_cfg=_coord_clustering_build_recon_plot_cfg,
        precomputed_kwargs=_coord_clustering_precomputed_kwargs,
        gen_extra_kwargs=_coord_clustering_gen_extra_kwargs,
        recon_extra_kwargs=_coord_clustering_recon_extra_kwargs,
    ),
    AnalysisMetricSpec(
        id="distmap_clustering",
        gen_key="distmap_clustering_gen",
        recon_key="distmap_clustering_recon",
        subdir="distmap_clustering",
        figure_filename="mixed_dendrograms.png",
        get_or_compute_test_to_train=_distmap_clustering_get_or_compute,
        run_gen_analysis=clustering.run_distmap_clustering_gen_analysis,
        run_gen_analysis_multi=clustering.run_distmap_clustering_gen_analysis_multi,
        run_recon_analysis=clustering.run_distmap_clustering_recon_analysis,
        cache_filename=_distmap_clustering_cache_filename,
        kwargs_for_cache=_distmap_clustering_kwargs_for_cache,
        build_gen_plot_cfg=_distmap_clustering_build_gen_plot_cfg,
        build_recon_plot_cfg=_distmap_clustering_build_recon_plot_cfg,
        precomputed_kwargs=_distmap_clustering_precomputed_kwargs,
        gen_extra_kwargs=_distmap_clustering_gen_extra_kwargs,
        recon_extra_kwargs=_distmap_clustering_recon_extra_kwargs,
    ),
]
