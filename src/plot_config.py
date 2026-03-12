"""
Shared plot configuration for the pipeline: colors, fonts, and font sizes.
Use these constants so all plots have consistent styling.

Conceptual groups (six): Experimental, Train, Test, Gen, Train Recon, Test Recon.
For test-to-train comparison and experimental reference we use COLOR_EXP so Train and Test
are shown as "experimental structures" (one color) in RMSD/Q panels and clustering.
"""
# ---------------------------------------------------------------------------
# Colors (canonical palette; clustering SOURCE_COLORS align with these)
# ---------------------------------------------------------------------------
COLOR_TRAIN = "#00A8CC"      # blue (used where train is distinct from "experimental" ref)
COLOR_GEN = "#D42248"        # green
COLOR_TEST = "#00B84E"       # orange (used where test is distinct from "experimental" ref)
COLOR_TRAIN_RECON = "#80DDED"   # darker green
COLOR_TEST_RECON = "#80DDA8"    # lighter orange
# Experimental: train/test reference and test-to-train comparison (RMSD, Q, clustering)
COLOR_EXP = "#0C3E96"

# Neutral grays (dendrograms, expected-value bars, fallback source, text)
COLOR_GRAY_LIGHT = "#aaaaaa"
COLOR_GRAY_MID = "#888888"
COLOR_GRAY_DARK = "#555555"
COLOR_GRAY_TEXT = "#333333"

# Ordered list for three-panel figures: panel 0, 1, 2
# Gen: Test→Train (experimental), Gen→Train (gen), Gen→Test (gen — same as Gen→Train)
# Recon: Test→Train (experimental), Train Recon, Test Recon
# Train/Test reference and test-to-train comparison use COLOR_EXP (experimental structures).
GEN_PANEL_COLORS = (COLOR_EXP, COLOR_GEN, COLOR_GEN)
RECON_PANEL_COLORS = (COLOR_EXP, COLOR_TRAIN_RECON, COLOR_TEST_RECON)

# Colormaps for distance maps (training video uses its own)
CMAP_DM = "viridis"
CMAP_DM_R = "viridis_r"  # reversed (e.g. avg maps: high = warm)


# ---------------------------------------------------------------------------
# Font and font sizes (use in every plotting call for consistency)
# ---------------------------------------------------------------------------
FONT_FAMILY = "sans-serif"

# Figure-level title (suptitle)
FONT_SIZE_SUPTITLE = 14
# Panel/subplot title (set_title)
FONT_SIZE_TITLE = 11
# Axis labels (set_xlabel, set_ylabel)
FONT_SIZE_AXIS = 10
# Tick labels
FONT_SIZE_TICK = 9
# Legend
FONT_SIZE_LEGEND = 9
# Secondary text, dense-panel legends
FONT_SIZE_SMALL = 8
# Very dense grids (e.g. bond_length_by_genomic_distance legend)
FONT_SIZE_TINY = 6
