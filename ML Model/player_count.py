"""Pixel-level player detection / counting within predicted team regions.

The per-cell model tells us WHICH
team occupies WHICH cells; this module decides HOW MANY players make up each
team's region by segmenting jersey/player pixels and splitting touching players
with a distance-transform watershed.

Pipeline per team:
  1. Build an 800x600 mask from the team's predicted grid cells.
  2. Inside that mask, keep non-grass (player/jersey) pixels.
  3. Morphologically consolidate, then split into instances via watershed on the
     distance transform (peaks = player centers).
  4. Filter instances by a minimum area; each surviving blob is one player.

There is NO instance-level ground truth in the dataset (labels are per-cell team
ids only), so counting is validated qualitatively via the overlay in inference.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from create_dataset import (
    CELL_HEIGHT,
    CELL_WIDTH,
    GRID_COLS,
    GRID_ROWS,
    _GRASS_HUE_HI,
    _GRASS_HUE_LO,
    _GRASS_MIN_SAT,
)

# ------------------------------------------------------------
# Tunable parameters (chosen for 800x600 frames; qualitative).
# ------------------------------------------------------------
# Players in these frames stand side-by-side (separated HORIZONTALLY). One player
# -- even with an arm extended -- is a single continuous "hump" in the vertical
# projection (foreground-pixel count per column); two players show a valley/gap
# between them. So we count humps in the column profile rather than width-split
# (which mistakes one wide player for two) or 2D watershed (which over-splits a
# tall player into head/torso/legs).
_CELL_AREA = CELL_WIDTH * CELL_HEIGHT                 # 7500 px
MIN_PLAYER_AREA = int(0.5 * _CELL_AREA)               # ignore blobs smaller than ~0.5 cell
MORPH_KERNEL = 9                                      # consolidation kernel size
PROFILE_SMOOTH = 31                                   # column-profile smoothing window (px)
MIN_COL_HEIGHT = 28                                   # a column counts as "player" above this
PROFILE_REL_THR = 0.18                                # ...or above this fraction of the peak
MIN_RUN_WIDTH = 38                                    # min width (px) of a player hump
MERGE_GAP = 18                                        # merge humps separated by < this gap


@dataclass
class PlayerInstance:
    """One detected player within a team region."""
    team: int
    index: int                  # 1-based index within the team
    bbox: tuple[int, int, int, int]   # x, y, w, h
    centroid: tuple[int, int]         # cx, cy
    area: int

    @property
    def top_center(self) -> tuple[int, int]:
        x, y, w, _ = self.bbox
        return (x + w // 2, y)


def team_cell_mask(grid: np.ndarray, team: int) -> np.ndarray:
    """800x600 boolean mask of pixels covered by ``team``'s predicted cells."""
    grid = np.asarray(grid, dtype=int).reshape(GRID_ROWS, GRID_COLS)
    mask = np.zeros((GRID_ROWS * CELL_HEIGHT, GRID_COLS * CELL_WIDTH), dtype=bool)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r, c] == team:
                y0, x0 = r * CELL_HEIGHT, c * CELL_WIDTH
                mask[y0:y0 + CELL_HEIGHT, x0:x0 + CELL_WIDTH] = True
    return mask


def _foreground_in_region(image_bgr: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    """Non-grass (player/jersey) pixels inside the team's cell mask, cleaned."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    grass = (hue >= _GRASS_HUE_LO) & (hue <= _GRASS_HUE_HI) & (sat > _GRASS_MIN_SAT)
    region = (cell_mask & ~grass).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, k, iterations=2)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, k, iterations=1)
    return region


def _instance_from_pixels(team: int, blob: np.ndarray) -> PlayerInstance | None:
    """Build a PlayerInstance from a boolean pixel mask (None if empty)."""
    ys, xs = np.nonzero(blob)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return PlayerInstance(
        team=team, index=0,
        bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
        centroid=(int(xs.mean()), int(ys.mean())),
        area=int(xs.size),
    )


def _profile_runs(region: np.ndarray) -> list[tuple[int, int]]:
    """Column ranges [(x0, x1), ...] of player 'humps' in the vertical projection.

    profile[x] = number of foreground pixels in column x. A contiguous stretch of
    columns above the threshold is one player; small gaps are merged and narrow
    humps dropped.
    """
    profile = region.sum(axis=0).astype(np.float64) / 255.0
    if PROFILE_SMOOTH > 1:
        kernel = np.ones(PROFILE_SMOOTH) / PROFILE_SMOOTH
        profile = np.convolve(profile, kernel, mode="same")
    peak = float(profile.max())
    if peak <= 0:
        return []
    thr = max(MIN_COL_HEIGHT, PROFILE_REL_THR * peak)
    above = profile > thr

    runs: list[list[int]] = []
    in_run = False
    for x, flag in enumerate(above):
        if flag and not in_run:
            runs.append([x, x]); in_run = True
        elif flag:
            runs[-1][1] = x
        else:
            in_run = False

    # merge runs separated by a tiny gap, then drop narrow ones
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if (b - a + 1) >= MIN_RUN_WIDTH]


def detect_team_players(
    image_bgr: np.ndarray,
    grid: np.ndarray,
    team: int,
    min_area: int = MIN_PLAYER_AREA,
) -> list[PlayerInstance]:
    """Detect and localize the individual players of one team in the image.

    Each 'hump' in the column profile of the team's player pixels is one player.
    """
    cell_mask = team_cell_mask(grid, team)
    if not cell_mask.any():
        return []
    region = _foreground_in_region(image_bgr, cell_mask)

    instances: list[PlayerInstance] = []
    for x0, x1 in _profile_runs(region):
        band = np.zeros_like(region, dtype=bool)
        band[:, x0:x1 + 1] = region[:, x0:x1 + 1] > 0
        if int(band.sum()) < min_area:
            continue
        inst = _instance_from_pixels(team, band)
        if inst is not None:
            instances.append(inst)

    # Fallback: team predicted but nothing large enough segmented -> 1 player.
    if not instances and int(cell_mask.sum()) >= min_area:
        inst = _instance_from_pixels(team, cell_mask)
        if inst is not None:
            instances.append(inst)

    # order left-to-right and assign 1-based indices
    instances.sort(key=lambda p: p.centroid[0])
    for i, inst in enumerate(instances, start=1):
        inst.index = i
    return instances


def count_players(
    image_bgr: np.ndarray,
    grid: np.ndarray,
    min_area: int = MIN_PLAYER_AREA,
) -> dict[int, list[PlayerInstance]]:
    """Return {team: [PlayerInstance, ...]} for every team present in the grid."""
    grid = np.asarray(grid, dtype=int).reshape(GRID_ROWS, GRID_COLS)
    teams = sorted(int(t) for t in np.unique(grid) if t > 0)
    return {t: detect_team_players(image_bgr, grid, t, min_area) for t in teams}