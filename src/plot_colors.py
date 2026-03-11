"""
Shared color palette for pipeline plots and analysis.
Use these constants so Train / Test / Gen / Exp / Recon colors are consistent everywhere.
"""
# Canonical palette (aligned with clustering SOURCE_COLORS)
COLOR_TRAIN = "#4878d0"      # blue
COLOR_GEN = "#6acc65"        # green
COLOR_TEST = "#ee854a"       # orange
COLOR_TRAIN_RECON = "#5a9c50"   # darker green
COLOR_TEST_RECON = "#f0a060"    # lighter orange
# Exp: use same blue as Train for "experimental reference" in gen plots; training video may override for dark theme
COLOR_EXP = "#4878d0"

# Ordered list for three-panel figures: panel 0, 1, 2
# Gen: Test→Train (train ref), Gen→Train (gen), Gen→Test (test)
GEN_PANEL_COLORS = (COLOR_TRAIN, COLOR_GEN, COLOR_TEST)
# Recon: Test→Train (train ref), Train recon, Test recon
RECON_PANEL_COLORS = (COLOR_TRAIN, COLOR_TRAIN_RECON, COLOR_TEST_RECON)

# Colormaps for distance maps and related plots (training video uses its own).
CMAP_DM = "viridis"
CMAP_DM_R = "viridis_r"  # reversed (e.g. avg maps: high = warm)
