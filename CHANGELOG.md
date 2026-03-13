# Changelog

## 2026-03-13

- **Dashboard: Radar grid view.** Added a "Radar grid" page to the Pipeline dashboard (`Pipeline/src/dashboard.py`). The view shows a grid of radar (spider) plots for all scored Euclideanizer runs, each labeled with its overall score, ordered best-to-worst (left to right, top to bottom). Hovering a cell shows a tooltip with that run's parameters (frozen DistMap + Euclideanizer). Updated `Pipeline/README.md` and `Pipeline/STYLE_GUIDE.md` to document the new view.
