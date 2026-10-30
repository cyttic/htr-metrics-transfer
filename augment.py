# -*- coding: utf-8 -*-
"""Augmentation families x strength levels for the augmentation / training-free-metrics study.

ONE source of truth: the training notebooks (online augmentation) and the metric computation (K augmented draws of the training
set) both call `augment()` from this file, so the metrics describe exactly the distribution the recognizer was trained on.

Input / output: a grayscale uint8 line image (H x W, white background, dark ink) ALREADY resized to the model line height
(64 px), so every magnitude below is in pixels of a 64 px high line. Level 0 = identity. For level L >= 1 every image is
augmented (p = 1); each magnitude is drawn uniformly from [0, MAX[L]] (with a random sign where it has one), so a higher
level widens the distribution rather than shifting it.

Families
  geom     rotation U(-a, a) deg + horizontal shear U(-s, s); the canvas is enlarged so no ink is cut, then resized back to 64 px
           high (a rotated long line gets taller, so its letters shrink: part of what rotating a cropped line does)
  elastic  smooth random displacement field (Gaussian-blurred noise, sigma 4 px), RMS displacement U(0, A) px per axis
  stroke   ink thickening with an elliptical kernel of size r+1, r ~ U{-1..k}; r = -1 thins by one px (more thinning erases
           HHD's ~3 px strokes), r = 0 leaves the line unchanged
  photo    Gaussian blur sigma U(0, b) + ink-contrast loss (ink lightened by up to c) + Gaussian pixel noise std U(0, n)
"""
import math

import cv2
import numpy as np

LEVELS = (0, 1, 2, 3, 4)
FAMILIES = ("geom", "elastic", "stroke", "photo")
MAX = {
    "geom":    {"rot": (0, 0.75, 1.5, 2.25, 3.0), "shear": (0, 0.10, 0.20, 0.35, 0.50)},
    "elastic": {"amp": (0, 0.75, 1.5, 2.25, 3.0), "sigma": 4.0},
    "stroke":  {"k": (0, 1, 2, 3, 4)},
    "photo":   {"blur": (0, 0.5, 1.0, 1.5, 2.0), "contrast": (0, 0.15, 0.30, 0.45, 0.60), "noise": (0, 0.03, 0.06, 0.10, 0.15)},
}


def condition_name(family, level):
    return "none" if level == 0 or family in (None, "none") else f"{family}{level}"


def parse_condition(name):
    """'none' -> (None, 0); 'geom3' -> ('geom', 3)"""
    if name == "none": return None, 0
    return name[:-1], int(name[-1])


def _geom(g, L, rng):
    a = math.radians(rng.uniform(-1, 1) * MAX["geom"]["rot"][L]); s = rng.uniform(-1, 1) * MAX["geom"]["shear"][L]
    h0, w0 = g.shape
    px = int(math.ceil(abs(s) * h0 + abs(math.sin(a)) * h0)) + 2; py = int(math.ceil(abs(math.sin(a)) * w0 / 2)) + 1
    g = cv2.copyMakeBorder(g, py, py, px, px, cv2.BORDER_CONSTANT, value=255); h, w = g.shape
    cx, cy = w / 2, h / 2
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ np.array([[1, s], [0, 1]])
    M = np.hstack([R, (np.array([cx, cy]) - R @ np.array([cx, cy]))[:, None]])
    g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rows = np.where((g < 200).any(1))[0]                       # crop the vertical padding back to the ink (+ the original margin)
    if len(rows): g = g[max(0, rows[0] - 2):rows[-1] + 3]
    return cv2.resize(g, (max(1, round(g.shape[1] * h0 / g.shape[0])), h0), interpolation=cv2.INTER_AREA) if g.shape[0] != h0 else g


def _elastic(g, L, rng):
    amp = rng.uniform(0, MAX["elastic"]["amp"][L]); sig = MAX["elastic"]["sigma"]
    h, w = g.shape
    f = [cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sig) for _ in range(2)]
    dx, dy = (f[i] * (amp / max(1e-6, float(f[i].std()))) for i in (0, 1))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(g, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _stroke(g, L, rng):
    k = MAX["stroke"]["k"][L]; r = int(rng.integers(-1, k + 1))
    if r == 0: return g
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(r) + 1, abs(r) + 1))
    return cv2.erode(g, ker) if r > 0 else cv2.dilate(g, ker)   # white background: erode = bolder ink, dilate = thinner ink


def _photo(g, L, rng):
    P = MAX["photo"]; x = g.astype(np.float32)
    b = rng.uniform(0, P["blur"][L])
    if b > 0.05: x = cv2.GaussianBlur(x, (0, 0), b)
    c = rng.uniform(0, P["contrast"][L]); x = 255 - (255 - x) * (1 - c)
    n = rng.uniform(0, P["noise"][L])
    if n > 0: x = x + rng.standard_normal(x.shape).astype(np.float32) * (255 * n)
    return np.clip(x, 0, 255).astype(np.uint8)


_FN = {"geom": _geom, "elastic": _elastic, "stroke": _stroke, "photo": _photo}


def augment(gray, family, level, rng):
    """gray: uint8 H x W (64 px high, white background). Returns a uint8 image of the same height (geom may widen it)."""
    if level == 0 or family in (None, "none"): return gray
    return _FN[family](gray, level, rng)


def to_line(image, height=64, max_width=None):
    """PIL image -> grayscale uint8 array, `height` px high (aspect kept; optionally squeezed to max_width)."""
    g = np.array(image.convert("L")); h, w = g.shape
    nw = max(1, round(w * height / h))
    if max_width and nw > max_width: nw = max_width
    return cv2.resize(g, (nw, height), interpolation=cv2.INTER_AREA if h > height else cv2.INTER_CUBIC)
