#!/usr/bin/env python3
"""
Build a static HTML preview of the planned Meta-Analysis (sufficiency) dashboard page.

Copies figures from a synthetic or pipeline output tree into dashboard_sandbox/outputs/
so the page works from file:// (self-contained assets).

Example (after running synthetic_plot_sandbox):

  python dashboard_sandbox/build_meta_analysis_page.py \\
    --input synthetic_plot_sandbox/outputs/fake_base
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path


def _natural_sort_key(s: str) -> tuple:
    parts = re.split(r"(\d+)", s)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _discover_seeds(suff_root: Path) -> list[Path]:
    if not suff_root.is_dir():
        return []
    out = []
    for p in suff_root.iterdir():
        if p.is_dir() and p.name.startswith("seed_"):
            out.append(p)
    return sorted(out, key=lambda x: _natural_sort_key(x.name))


def _heatmap_path(seed_dir: Path) -> Path | None:
    h = seed_dir / "heatmap" / "sufficiency_heatmap_rmsd_q.png"
    return h if h.is_file() else None


def _curves_path(seed_dir: Path) -> Path | None:
    p = seed_dir / "curves" / "sufficiency_median_recon_vs_split_by_max_data.png"
    return p if p.is_file() else None


def _distribution_pngs(seed_dir: Path) -> list[tuple[str, Path]]:
    dist = seed_dir / "distributions"
    if not dist.is_dir():
        return []
    rows: list[tuple[str, Path]] = []
    for sub in sorted(dist.iterdir(), key=lambda x: _natural_sort_key(x.name)):
        if not sub.is_dir() or not sub.name.startswith("max_data_"):
            continue
        tag = sub.name.replace("max_data_", "", 1)
        png = sub / "distributions_rmsd_q.png"
        if png.is_file():
            rows.append((tag, png))
    return rows


def _copy_tree(src: Path, dst_dir: Path, *, rel: Path) -> Path:
    dst = dst_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def build_page(*, input_root: Path, out_dir: Path) -> Path:
    suff = input_root / "meta_analysis" / "sufficiency"
    assets = out_dir / "assets" / "meta_analysis"
    assets.mkdir(parents=True, exist_ok=True)

    seeds = _discover_seeds(suff)
    sections_html: list[str] = []

    for seed_dir in seeds:
        sid = seed_dir.name  # seed_1
        m = re.match(r"seed_(\d+)$", sid)
        seed_label = f"Seed {m.group(1)}" if m else sid

        heat = _heatmap_path(seed_dir)
        heat_rel = ""
        if heat:
            rel = Path("sufficiency") / seed_dir.name / "heatmap" / heat.name
            _copy_tree(heat, assets, rel=rel)
            heat_rel = (assets / rel).relative_to(out_dir).as_posix()

        dist_rows = _distribution_pngs(seed_dir)
        dist_cards = []
        for tag, png in dist_rows:
            rel = Path("sufficiency") / seed_dir.name / "distributions" / f"max_data_{tag}" / png.name
            _copy_tree(png, assets, rel=rel)
            href = (assets / rel).relative_to(out_dir).as_posix()
            dist_cards.append(
                f'<div class="meta-dist-card">'
                f'<div class="meta-dist-card-title">Max Data = {html.escape(tag)}</div>'
                f'<img src="{html.escape(href)}" alt="Distributions Max Data {html.escape(tag)}" loading="lazy">'
                f"</div>"
            )

        nav_id = html.escape(sid.replace(" ", "_"), quote=True)
        heat_block = ""
        if heat_rel:
            heat_block = (
                f'<div class="block">'
                f'<span class="block-title">Median Heatmap (RMSD / Q)</span>'
                f'<img class="meta-heatmap-img" src="{html.escape(heat_rel)}" alt="Sufficiency Heatmap" loading="lazy">'
                f"</div>"
            )
        curves = _curves_path(seed_dir)
        curves_block = ""
        if curves:
            rel_c = Path("sufficiency") / seed_dir.name / "curves" / curves.name
            _copy_tree(curves, assets, rel=rel_c)
            curves_href = (assets / rel_c).relative_to(out_dir).as_posix()
            curves_block = (
                f'<div class="block">'
                f'<span class="block-title">Median Recon Vs Training Split (By Max Structures)</span>'
                f'<img class="meta-heatmap-img" src="{html.escape(curves_href)}" alt="Sufficiency curves" loading="lazy">'
                f"</div>"
            )
        dist_section = ""
        if dist_cards:
            dist_section = (
                '<h3 class="meta-subheading">Distributions By Max Data</h3>'
                '<div class="meta-dist-grid">' + "".join(dist_cards) + "</div>"
            )
        elif not heat_block and not curves_block:
            dist_section = '<p class="muted">No Distribution Figures Under This Seed.</p>'

        sections_html.append(
            f'<section class="run-card meta-seed-section" id="{nav_id}">'
            f'<h2 class="run-card-title">{html.escape(seed_label)}</h2>'
            f"{heat_block}{curves_block}{dist_section}"
            f"</section>"
        )

    nav_li = []
    for seed_dir in seeds:
        sid = seed_dir.name
        m = re.match(r"seed_(\d+)$", sid)
        label = f"Seed {m.group(1)}" if m else sid
        nav_id = html.escape(sid.replace(" ", "_"), quote=True)
        nav_li.append(f'<li><a href="#{nav_id}">{html.escape(label)}</a></li>')

    if not sections_html:
        body_main = (
            '<div class="empty-state">'
            "<span class=\"empty-state-title\">No Sufficiency Figures Found</span>"
            f"<p>Expected <code>meta_analysis/sufficiency/seed_*/</code> under <code>{html.escape(str(input_root))}</code>. "
            "Run <code>python synthetic_plot_sandbox/generate_synthetic_plots.py</code> or point <code>--input</code> at a pipeline output dir.</p>"
            "</div>"
        )
    else:
        nav = (
            '<nav class="meta-page-nav" aria-label="Seed Sections">'
            "<span class=\"meta-page-nav-label\">Jump To</span><ul>"
            + "".join(nav_li)
            + "</ul></nav>"
        )
        body_main = nav + "\n" + "\n".join(sections_html)

    title = "Meta-Analysis (Sandbox)"
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg-page: #1a1a1a;
      --bg-card: #252525;
      --border: #444;
      --text: #e0e0e0;
      --text-secondary: #b0b0b0;
      --text-muted: #888;
      --accent: #5a9;
      --accent-block: #6ab;
    }}
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0; background: var(--bg-page); color: var(--text); }}
    header {{ margin: 0 1.5rem; padding: 1.25rem 0 0.5rem; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 0.35rem 0; }}
    header p {{ color: var(--text-muted); font-size: 0.9rem; margin: 0; line-height: 1.45; }}
    .content {{ margin: 0 1.5rem 2.5rem; }}
    .run-card {{ background: var(--bg-card); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }}
    .run-card-title {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
    .block {{ margin-bottom: 1.25rem; }}
    .block:last-child {{ margin-bottom: 0; }}
    .block-title {{ display: block; font-size: 1rem; color: var(--text-secondary); margin: 0 0 0.6rem 0; padding: 0.4rem 0.6rem; font-weight: 600; background: #333; border-radius: 4px; border-left: 3px solid var(--accent-block); }}
    .meta-heatmap-img {{ width: 100%; max-width: 100%; height: auto; display: block; }}
    .meta-subheading {{ font-size: 1rem; color: var(--text-secondary); margin: 1.25rem 0 0.75rem 0; font-weight: 600; }}
    .meta-dist-grid {{ display: grid; grid-template-columns: 1fr; gap: 1rem; align-items: start; width: 100%; }}
    .meta-dist-card {{ background: #1e1e1e; border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; }}
    .meta-dist-card-title {{ font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 600; }}
    .meta-dist-card img {{ width: 100%; max-width: 100%; height: auto; display: block; }}
    .meta-page-nav {{ position: sticky; top: 0; z-index: 5; background: #222; border: 1px solid var(--border); border-radius: 8px; padding: 0.65rem 1rem; margin-bottom: 1.25rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1rem; }}
    .meta-page-nav-label {{ font-size: 0.85rem; color: var(--text-muted); font-weight: 600; }}
    .meta-page-nav ul {{ list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 0.35rem 0.75rem; }}
    .meta-page-nav a {{ color: var(--accent); text-decoration: none; font-size: 0.9rem; }}
    .meta-page-nav a:hover {{ text-decoration: underline; }}
    .meta-seed-section {{ scroll-margin-top: 5rem; }}
    .empty-state {{ padding: 2rem; text-align: center; background: #252525; border-radius: 8px; border: 1px dashed var(--border); color: var(--text-muted); }}
    .empty-state-title {{ display: block; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.5rem; }}
    code {{ font-size: 0.82rem; background: #2a2a2a; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    .muted {{ color: var(--text-muted); font-size: 0.9rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Static preview for the sufficiency meta-analysis view (heatmap, optional median-recon curves, distributions). Figures are copied next to this file under <code>assets/</code>; layout mirrors the production dashboard Meta-Analysis page.</p>
  </header>
  <div class="content">
{body_main}
  </div>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(html_doc, encoding="utf-8")
    return index


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_in = repo / "synthetic_plot_sandbox" / "outputs" / "fake_base"
    ap = argparse.ArgumentParser(description="Build static meta-analysis dashboard sandbox page.")
    ap.add_argument(
        "--input",
        type=Path,
        default=default_in,
        help=f"Output dir containing meta_analysis/sufficiency (default: {default_in})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "meta_analysis",
        help="Directory to write index.html and assets/",
    )
    args = ap.parse_args()
    root = args.input.resolve()
    out = args.out.resolve()
    if out.exists() and not out.is_dir():
        raise SystemExit(f"--out must be a directory: {out}")
    index = build_page(input_root=root, out_dir=out)
    print(f"Wrote {index}")
    print(f"Assets under {out / 'assets'}")


if __name__ == "__main__":
    main()
