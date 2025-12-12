#!/usr/bin/env python3
"""
CORONAL CROSS-SECTIONAL MEDICAL ILLUSTRATION ENGINE
Parks Classification Style - Frank Netter Aesthetic
ISEF 2026 MRI-Crohn Atlas Project

Professional anatomical illustrations showing:
- Coronal (vertical) cross-section through anal canal
- Parks Classification fistula pathing
- Levator ani, sphincter complex, ischioanal fossa
- Hand-drawn Netter aesthetic with granulation texture

Inspired by Frank Netter's Atlas of Human Anatomy.
"""

import json
import re
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (PathPatch, Polygon, FancyBboxPatch,
                                Rectangle, Ellipse, Circle, Wedge)
from matplotlib.path import Path as MplPath
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# RENDERING CONFIGURATION
# =============================================================================

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# =============================================================================
# NETTER COLOR PALETTE - CORONAL VIEW
# =============================================================================

COLORS = {
    # Canvas & Frame
    'canvas': '#FFFAF0',           # Floral White (medical textbook)
    'border': '#2C2C2C',           # Near black frame

    # Smooth Muscle (Internal Anal Sphincter)
    'ias_fill': '#F5DEB3',         # Wheat - pale beige smooth muscle
    'ias_edge': '#D2B48C',         # Tan outline
    'ias_hatch': '#C4A882',        # Muted tan for fine hatching

    # Striated Muscle (External Anal Sphincter)
    'eas_fill': '#CD5C5C',         # Indian red - striated muscle
    'eas_edge': '#8B3A3A',         # Dark indian red outline
    'eas_stripe': '#A94442',       # Darker stripe for striation

    # Levator Ani
    'levator_fill': '#E8B4B8',     # Pale rose - levator muscle
    'levator_edge': '#C77C7C',     # Darker rose outline
    'levator_stripe': '#D19999',   # Striation marks

    # Ischioanal Fossa (Fat)
    'fat_fill': '#FFEAA7',         # Yellow adipose tissue
    'fat_edge': '#F9CA24',         # Golden outline
    'fat_lobule': '#FFE066',       # Fat lobule highlights

    # Anal Canal & Mucosa
    'mucosa': '#FFB6C1',           # Light pink mucosa
    'mucosa_edge': '#DB7093',      # Pale violet red
    'lumen': '#1C1C1C',            # Dark void
    'dentate': '#FF6B6B',          # Bright red dentate line

    # Rectum
    'rectum_fill': '#FFC0CB',      # Pink rectal wall
    'rectum_edge': '#FF69B4',      # Hot pink outline

    # Fistula Tract
    'fistula_fill': '#8B0000',     # Dark red tract lumen
    'fistula_wall': '#A52A2A',     # Brown-red granulation tissue
    'fistula_edge': '#DC143C',     # Crimson inflammatory edge
    'fistula_granulation': '#B22222', # Fire brick granulation

    # Abscess
    'abscess_fill': '#DAA520',     # Goldenrod pus
    'abscess_core': '#FFD700',     # Bright gold center
    'abscess_edge': '#B8860B',     # Dark goldenrod rim
    'abscess_wall': '#CD853F',     # Peru inflammatory wall

    # Seton
    'seton_main': '#1E90FF',       # Dodger blue
    'seton_highlight': '#87CEFA',  # Light sky blue

    # Text
    'text_dark': '#1A1A1A',
    'text_medium': '#4A4A4A',
    'text_light': '#696969',
    'text_white': '#FFFFFF',

    # Severity badges
    'badge_mild': '#2E8B57',
    'badge_moderate': '#DAA520',
    'badge_severe': '#CD5C5C',

    # Leader lines
    'leader': '#666666',
}

# =============================================================================
# ANATOMICAL DIMENSIONS (Coronal View)
# =============================================================================

# All coordinates in normalized units (-1 to 1 on y-axis)
ANATOMY = {
    # Vertical positions (y-axis, 0 = midpoint)
    'rectum_top': 0.85,
    'levator_y': 0.35,
    'dentate_y': -0.05,
    'anal_verge': -0.75,

    # Canal dimensions
    'canal_width': 0.12,           # Half-width of anal canal
    'canal_inner': 0.08,           # Inner lumen half-width

    # Sphincter thicknesses
    'ias_thickness': 0.06,         # IAS layer thickness
    'eas_thickness': 0.14,         # EAS layer thickness

    # Levator
    'levator_angle': 35,           # Degrees from horizontal
    'levator_thickness': 0.08,

    # Ischioanal fossa
    'fossa_width': 0.55,           # Total width to edge

    # Scale
    'scale_1cm': 0.15,             # 1cm in normalized units
}

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "output" / "visualizations"


# =============================================================================
# ORGANIC SHAPE GENERATION
# =============================================================================

def add_organic_noise(points: np.ndarray, amplitude: float = 0.01,
                      frequency: int = 3, seed: int = None) -> np.ndarray:
    """Add hand-drawn wobble to a path."""
    if seed is not None:
        np.random.seed(seed)

    n = len(points)
    noise = amplitude * (np.random.rand(n) - 0.5)
    noise = gaussian_filter1d(noise, sigma=max(1, n // 20), mode='wrap')

    # Add to perpendicular direction
    result = points.copy()
    for i in range(n):
        if i > 0 and i < n - 1:
            # Get tangent direction
            dx = points[i+1, 0] - points[i-1, 0]
            dy = points[i+1, 1] - points[i-1, 1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Perpendicular
                result[i, 0] += -dy / length * noise[i]
                result[i, 1] += dx / length * noise[i]

    return result


def generate_organic_boundary(points: List[Tuple[float, float]],
                              irregularity: float = 0.015,
                              seed: int = None) -> np.ndarray:
    """Convert control points to smooth organic boundary."""
    if seed is not None:
        np.random.seed(seed)

    pts = np.array(points)

    # Interpolate to higher resolution
    if len(pts) >= 4:
        # Use cubic spline
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0.01, per=False)
        u_fine = np.linspace(0, 1, 100)
        smooth_x, smooth_y = interpolate.splev(u_fine, tck)
        smooth = np.column_stack([smooth_x, smooth_y])
    else:
        smooth = pts

    # Add organic noise
    return add_organic_noise(smooth, amplitude=irregularity, seed=seed)


def generate_wavy_line(start: Tuple[float, float], end: Tuple[float, float],
                       amplitude: float = 0.02, frequency: int = 5,
                       seed: int = None) -> np.ndarray:
    """Generate wavy line between two points."""
    if seed is not None:
        np.random.seed(seed)

    n_points = 50
    t = np.linspace(0, 1, n_points)

    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])

    # Direction vector
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        # Perpendicular direction
        perp_x, perp_y = -dy / length, dx / length

        # Sine wave with decay at ends
        wave = amplitude * np.sin(frequency * np.pi * t)
        envelope = np.sin(np.pi * t)  # Smooth decay at endpoints
        wave *= envelope

        # Add random perturbation
        noise = amplitude * 0.3 * (np.random.rand(n_points) - 0.5)
        noise = gaussian_filter1d(noise, sigma=3)
        wave += noise

        x += perp_x * wave
        y += perp_y * wave

    return np.column_stack([x, y])


def generate_fistula_tunnel(start: Tuple[float, float], end: Tuple[float, float],
                            width: float = 0.025, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate fistula tract with irregular granulation tissue edges."""
    if seed is not None:
        np.random.seed(seed)

    # Central path with organic curves
    n_points = 60
    t = np.linspace(0, 1, n_points)

    # Base path
    x_center = start[0] + t * (end[0] - start[0])
    y_center = start[1] + t * (end[1] - start[1])

    # Add meandering
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        perp_x, perp_y = -dy / length, dx / length

        # Multiple frequency meander
        meander = 0.02 * np.sin(3 * np.pi * t)
        meander += 0.01 * np.sin(7 * np.pi * t + np.random.rand() * np.pi)

        # Envelope to keep endpoints fixed
        envelope = np.sin(np.pi * t) ** 0.5
        meander *= envelope

        x_center += perp_x * meander
        y_center += perp_y * meander

    center = np.column_stack([x_center, y_center])

    # Generate irregular edges (granulation tissue)
    left_edge = []
    right_edge = []

    for i in range(n_points):
        if i > 0 and i < n_points - 1:
            dx = center[i+1, 0] - center[i-1, 0]
            dy = center[i+1, 1] - center[i-1, 1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                perp_x, perp_y = -dy / length, dx / length
            else:
                perp_x, perp_y = 0, 1
        else:
            perp_x, perp_y = 0, 1

        # Variable width with granulation bumps
        w = width * (0.7 + 0.6 * np.random.rand())

        # Granulation irregularity
        bump = width * 0.3 * np.random.rand()

        left_edge.append([
            center[i, 0] + perp_x * (w + bump),
            center[i, 1] + perp_y * (w + bump)
        ])
        right_edge.append([
            center[i, 0] - perp_x * (w + bump * 0.5),
            center[i, 1] - perp_y * (w + bump * 0.5)
        ])

    # Smooth edges
    left = np.array(left_edge)
    right = np.array(right_edge)

    left[:, 0] = gaussian_filter1d(left[:, 0], sigma=2)
    left[:, 1] = gaussian_filter1d(left[:, 1], sigma=2)
    right[:, 0] = gaussian_filter1d(right[:, 0], sigma=2)
    right[:, 1] = gaussian_filter1d(right[:, 1], sigma=2)

    return left, right


def generate_abscess_blob(cx: float, cy: float, base_radius: float,
                          seed: int = None) -> np.ndarray:
    """Generate irregular abscess cavity shape."""
    if seed is not None:
        np.random.seed(seed)

    n_control = np.random.randint(10, 14)
    angles = np.linspace(0, 2 * np.pi, n_control, endpoint=False)

    # Angular jitter
    angles += 0.4 * (np.random.rand(n_control) - 0.5) * (2 * np.pi / n_control)
    angles = np.sort(angles)

    # Radial variation
    radii = base_radius * (0.6 + 0.8 * np.random.rand(n_control))

    # Add lobes
    radii *= 1 + 0.3 * np.sin(3 * angles + np.random.rand() * np.pi)

    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)

    # Close and smooth
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    try:
        tck, u = interpolate.splprep([x, y], s=0, per=True)
        u_fine = np.linspace(0, 1, 80)
        smooth_x, smooth_y = interpolate.splev(u_fine, tck)
        return np.column_stack([smooth_x, smooth_y])
    except:
        return np.column_stack([x, y])


# =============================================================================
# ANATOMICAL STRUCTURE DRAWING - CORONAL VIEW
# =============================================================================

def draw_ischioanal_fossa(ax: plt.Axes, side: str = 'both') -> None:
    """Draw ischioanal fossa as yellow adipose tissue."""

    # Fat lobule texture function
    def draw_fat_side(x_sign: int, seed: int):
        # Main fossa boundary
        x_inner = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                           ANATOMY['eas_thickness'])
        x_outer = x_sign * ANATOMY['fossa_width']

        y_top = ANATOMY['levator_y'] - 0.05
        y_bottom = ANATOMY['anal_verge'] + 0.1

        # Create irregular boundary
        top_pts = [
            (x_inner, y_top),
            (x_sign * 0.25, y_top + 0.02),
            (x_sign * 0.35, y_top - 0.03),
            (x_outer, y_top - 0.08),
        ]

        bottom_pts = [
            (x_outer, y_bottom + 0.05),
            (x_sign * 0.35, y_bottom),
            (x_sign * 0.25, y_bottom - 0.02),
            (x_inner, y_bottom + 0.03),
        ]

        # Smooth boundaries
        top_boundary = generate_organic_boundary(top_pts, irregularity=0.01, seed=seed)
        bottom_boundary = generate_organic_boundary(bottom_pts[::-1], irregularity=0.01, seed=seed+1)

        # Combine into polygon
        fossa_pts = np.vstack([top_boundary, bottom_boundary])

        # Main fill
        ax.fill(fossa_pts[:, 0], fossa_pts[:, 1],
                facecolor=COLORS['fat_fill'], edgecolor=COLORS['fat_edge'],
                linewidth=0.8, alpha=0.9, zorder=1)

        # Add fat lobule texture
        np.random.seed(seed + 100)
        n_lobules = 12
        for i in range(n_lobules):
            lx = x_inner + (x_outer - x_inner) * np.random.rand()
            if x_sign < 0:
                lx = x_outer + (x_inner - x_outer) * np.random.rand()
            ly = y_bottom + (y_top - y_bottom) * np.random.rand()
            lr = 0.015 + 0.02 * np.random.rand()

            # Check if inside fossa roughly
            if abs(lx) > abs(x_inner) * 0.9:
                lobule = generate_abscess_blob(lx, ly, lr, seed=seed + i)
                ax.fill(lobule[:, 0], lobule[:, 1],
                       facecolor=COLORS['fat_lobule'], edgecolor='none',
                       alpha=0.4, zorder=1)

    if side in ['both', 'right']:
        draw_fat_side(1, seed=1000)
    if side in ['both', 'left']:
        draw_fat_side(-1, seed=2000)


def draw_levator_ani(ax: plt.Axes) -> None:
    """Draw levator ani muscle angling upward from sphincter complex."""

    for x_sign in [1, -1]:
        # Levator originates from sphincter, angles up to pelvis
        angle_rad = math.radians(ANATOMY['levator_angle'])

        # Inner attachment point (at sphincter)
        x_inner = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                           ANATOMY['eas_thickness'] * 0.7)
        y_inner = ANATOMY['levator_y']

        # Outer point (toward pelvis)
        x_outer = x_sign * (ANATOMY['fossa_width'] + 0.05)
        y_outer = y_inner + abs(x_outer - x_inner) * math.tan(angle_rad)

        # Create muscle band
        thickness = ANATOMY['levator_thickness']

        # Top edge
        top_pts = [
            (x_inner - x_sign * 0.02, y_inner + thickness/2),
            ((x_inner + x_outer) / 2, (y_inner + y_outer) / 2 + thickness/2 + 0.01),
            (x_outer, y_outer + thickness/2),
        ]

        # Bottom edge
        bot_pts = [
            (x_outer, y_outer - thickness/2),
            ((x_inner + x_outer) / 2, (y_inner + y_outer) / 2 - thickness/2 - 0.01),
            (x_inner - x_sign * 0.02, y_inner - thickness/2),
        ]

        seed = 3000 if x_sign > 0 else 3100
        top_smooth = generate_organic_boundary(top_pts, irregularity=0.008, seed=seed)
        bot_smooth = generate_organic_boundary(bot_pts[::-1], irregularity=0.008, seed=seed+1)

        muscle_pts = np.vstack([top_smooth, bot_smooth])

        # Main fill
        ax.fill(muscle_pts[:, 0], muscle_pts[:, 1],
                facecolor=COLORS['levator_fill'], edgecolor=COLORS['levator_edge'],
                linewidth=1, zorder=5)

        # Add striation marks
        n_striations = 8
        for i in range(n_striations):
            t = (i + 0.5) / n_striations
            x1 = x_inner + t * (x_outer - x_inner)
            y1 = y_inner + t * (y_outer - y_inner)

            # Perpendicular to muscle
            dx = x_outer - x_inner
            dy = y_outer - y_inner
            length = math.sqrt(dx**2 + dy**2)
            perp_x = -dy / length * thickness * 0.35
            perp_y = dx / length * thickness * 0.35

            ax.plot([x1 - perp_x, x1 + perp_x], [y1 - perp_y, y1 + perp_y],
                    color=COLORS['levator_stripe'], linewidth=0.6, alpha=0.5, zorder=6)


def draw_external_sphincter(ax: plt.Axes) -> None:
    """Draw external anal sphincter with striated muscle texture."""

    # EAS wraps around the IAS from levator down to anal verge
    y_top = ANATOMY['levator_y'] + 0.02
    y_bottom = ANATOMY['anal_verge']

    for x_sign in [1, -1]:
        x_inner = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'])
        x_outer = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                           ANATOMY['eas_thickness'])

        # Create sphincter shape (thicker at middle)
        pts = [
            # Top
            (x_inner + x_sign * 0.01, y_top),
            (x_outer - x_sign * 0.02, y_top),
            # Upper bulge
            (x_outer + x_sign * 0.01, y_top - 0.15),
            # Middle (thickest)
            (x_outer + x_sign * 0.02, (y_top + y_bottom) / 2),
            # Lower bulge
            (x_outer + x_sign * 0.01, y_bottom + 0.15),
            # Bottom
            (x_outer - x_sign * 0.02, y_bottom + 0.05),
            (x_inner + x_sign * 0.01, y_bottom + 0.03),
            # Inner edge
            (x_inner, y_bottom + 0.1),
            (x_inner - x_sign * 0.01, (y_top + y_bottom) / 2),
            (x_inner, y_top - 0.05),
        ]

        seed = 4000 if x_sign > 0 else 4100
        boundary = generate_organic_boundary(pts, irregularity=0.01, seed=seed)

        # Main fill
        ax.fill(boundary[:, 0], boundary[:, 1],
                facecolor=COLORS['eas_fill'], edgecolor=COLORS['eas_edge'],
                linewidth=1.2, zorder=8)

        # Striated muscle texture (horizontal lines)
        n_stripes = 18
        for i in range(n_stripes):
            y_stripe = y_bottom + (y_top - y_bottom) * (i + 0.5) / n_stripes

            # Width varies with position
            stripe_inner = x_inner + x_sign * 0.005
            stripe_outer = x_outer - x_sign * 0.01

            # Add slight curve
            mid_bulge = x_sign * 0.008 * math.sin(math.pi * i / n_stripes)

            stripe = generate_wavy_line(
                (stripe_inner, y_stripe),
                (stripe_outer + mid_bulge, y_stripe),
                amplitude=0.003, frequency=3, seed=seed + i
            )
            ax.plot(stripe[:, 0], stripe[:, 1],
                    color=COLORS['eas_stripe'], linewidth=0.5, alpha=0.4, zorder=9)


def draw_internal_sphincter(ax: plt.Axes) -> None:
    """Draw internal anal sphincter as smooth muscle layer."""

    y_top = ANATOMY['levator_y'] - 0.05
    y_bottom = ANATOMY['dentate_y'] + 0.02

    for x_sign in [1, -1]:
        x_inner = x_sign * ANATOMY['canal_width']
        x_outer = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'])

        # IAS is thinner, smoother layer
        pts = [
            (x_inner, y_top),
            (x_outer, y_top + 0.01),
            (x_outer + x_sign * 0.008, (y_top + y_bottom) / 2),
            (x_outer, y_bottom - 0.01),
            (x_inner, y_bottom),
            (x_inner - x_sign * 0.005, (y_top + y_bottom) / 2),
        ]

        seed = 5000 if x_sign > 0 else 5100
        boundary = generate_organic_boundary(pts, irregularity=0.006, seed=seed)

        ax.fill(boundary[:, 0], boundary[:, 1],
                facecolor=COLORS['ias_fill'], edgecolor=COLORS['ias_edge'],
                linewidth=0.8, zorder=10)

        # Fine diagonal hatching for smooth muscle
        n_hatch = 20
        for i in range(n_hatch):
            y1 = y_bottom + (y_top - y_bottom) * i / n_hatch
            y2 = y_bottom + (y_top - y_bottom) * (i + 0.8) / n_hatch

            ax.plot([x_inner + x_sign * 0.005, x_outer - x_sign * 0.005],
                    [y1, y2],
                    color=COLORS['ias_hatch'], linewidth=0.3, alpha=0.4, zorder=11)


def draw_anal_canal(ax: plt.Axes) -> None:
    """Draw anal canal with mucosal lining and dentate line."""

    y_top = ANATOMY['rectum_top'] - 0.05
    y_bottom = ANATOMY['anal_verge']
    x_half = ANATOMY['canal_inner']

    # Mucosal wall (pink lining)
    for x_sign in [1, -1]:
        mucosa_pts = [
            (x_sign * x_half, y_top),
            (x_sign * (x_half + 0.02), y_top - 0.2),
            (x_sign * ANATOMY['canal_width'], ANATOMY['dentate_y']),
            (x_sign * ANATOMY['canal_width'], y_bottom + 0.1),
            (x_sign * (x_half + 0.01), y_bottom),
        ]

        seed = 6000 if x_sign > 0 else 6100
        boundary = generate_organic_boundary(mucosa_pts, irregularity=0.005, seed=seed)

        ax.fill(boundary[:, 0], boundary[:, 1],
                facecolor=COLORS['mucosa'], edgecolor=COLORS['mucosa_edge'],
                linewidth=0.6, zorder=12)

    # Central lumen (dark void)
    lumen_pts = np.array([
        [-x_half, y_top],
        [-x_half * 0.8, (y_top + y_bottom) / 2],
        [-x_half * 0.6, y_bottom],
        [x_half * 0.6, y_bottom],
        [x_half * 0.8, (y_top + y_bottom) / 2],
        [x_half, y_top],
    ])

    # Smooth it
    try:
        tck, u = interpolate.splprep([lumen_pts[:, 0], lumen_pts[:, 1]], s=0.01, per=False)
        u_fine = np.linspace(0, 1, 60)
        smooth_x, smooth_y = interpolate.splev(u_fine, tck)
        lumen_smooth = np.column_stack([smooth_x, smooth_y])
    except:
        lumen_smooth = lumen_pts

    ax.fill(lumen_smooth[:, 0], lumen_smooth[:, 1],
            facecolor=COLORS['lumen'], edgecolor='none', zorder=13)


def draw_dentate_line(ax: plt.Axes) -> None:
    """Draw dentate line as jagged landmark."""

    y = ANATOMY['dentate_y']
    x_range = ANATOMY['canal_width']

    # Jagged dentate line
    n_teeth = 12
    teeth_x = []
    teeth_y = []

    for i in range(n_teeth + 1):
        x = -x_range + 2 * x_range * i / n_teeth
        teeth_x.append(x)

        if i % 2 == 0:
            teeth_y.append(y + 0.015 + 0.008 * np.random.rand())
        else:
            teeth_y.append(y - 0.01 - 0.005 * np.random.rand())

    ax.plot(teeth_x, teeth_y, color=COLORS['dentate'], linewidth=2,
            zorder=14, solid_capstyle='round')

    # Add subtle glow
    ax.plot(teeth_x, teeth_y, color=COLORS['dentate'], linewidth=4,
            alpha=0.3, zorder=13)


def draw_rectum(ax: plt.Axes) -> None:
    """Draw rectal ampulla at top of canal."""

    y_top = ANATOMY['rectum_top'] + 0.08
    y_bottom = ANATOMY['rectum_top'] - 0.1

    # Rectum widens above anal canal
    pts = [
        (-ANATOMY['canal_inner'], y_bottom),
        (-ANATOMY['canal_inner'] * 1.5, y_bottom + 0.05),
        (-ANATOMY['canal_inner'] * 2, y_top - 0.03),
        (-ANATOMY['canal_inner'] * 1.8, y_top),
        (ANATOMY['canal_inner'] * 1.8, y_top),
        (ANATOMY['canal_inner'] * 2, y_top - 0.03),
        (ANATOMY['canal_inner'] * 1.5, y_bottom + 0.05),
        (ANATOMY['canal_inner'], y_bottom),
    ]

    boundary = generate_organic_boundary(pts, irregularity=0.01, seed=7000)

    # Wall
    ax.fill(boundary[:, 0], boundary[:, 1],
            facecolor=COLORS['rectum_fill'], edgecolor=COLORS['rectum_edge'],
            linewidth=1, zorder=6)

    # Lumen
    inner_scale = 0.6
    inner_pts = boundary * np.array([inner_scale, 1])
    ax.fill(inner_pts[:, 0], inner_pts[:, 1],
            facecolor=COLORS['lumen'], edgecolor='none', zorder=7)


# =============================================================================
# PATHOLOGY DRAWING - PARKS CLASSIFICATION
# =============================================================================

def draw_fistula_tract(ax: plt.Axes, tract_type: str, clock: int = 6,
                       seed: int = None) -> None:
    """
    Draw fistula tract following Parks Classification:
    - Intersphincteric: Between IAS and EAS
    - Transsphincteric: Through EAS into ischioanal fossa
    - Suprasphincteric: Above levator then down
    - Extrasphincteric: Outside sphincter complex entirely
    """
    if seed is None:
        seed = hash(tract_type) % (2**31)

    np.random.seed(seed)

    # Determine side from clock position
    x_sign = 1 if clock in [1, 2, 3, 4, 5, 11, 12] else -1
    if clock in [6, 12]:
        x_sign = 1 if np.random.rand() > 0.5 else -1

    # Internal opening at dentate line
    internal_x = x_sign * ANATOMY['canal_width'] * 0.9
    internal_y = ANATOMY['dentate_y']

    # Tract path depends on classification
    if tract_type == 'intersphincteric':
        # Stays between IAS and EAS
        mid_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] * 0.5)
        mid_y = ANATOMY['dentate_y'] - 0.15

        external_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] + 0.02)
        external_y = ANATOMY['anal_verge'] + 0.15

        control_pts = [
            (internal_x, internal_y),
            (mid_x, mid_y),
            (external_x, external_y),
        ]

    elif tract_type == 'transsphincteric':
        # Traverses through EAS into ischioanal fossa
        through_sphincter_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                                        ANATOMY['eas_thickness'] * 0.5)
        through_sphincter_y = ANATOMY['dentate_y'] - 0.1

        in_fossa_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                               ANATOMY['eas_thickness'] + 0.08)
        in_fossa_y = ANATOMY['dentate_y'] - 0.2

        external_x = x_sign * (ANATOMY['fossa_width'] - 0.1)
        external_y = ANATOMY['anal_verge'] + 0.1

        control_pts = [
            (internal_x, internal_y),
            (through_sphincter_x, through_sphincter_y),
            (in_fossa_x, in_fossa_y),
            (external_x, external_y),
        ]

    elif tract_type == 'suprasphincteric':
        # Goes up, over levator, then down
        up_x = x_sign * ANATOMY['canal_width'] * 0.8
        up_y = ANATOMY['levator_y'] + 0.05

        over_levator_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                                   ANATOMY['eas_thickness'] + 0.05)
        over_levator_y = ANATOMY['levator_y'] + 0.1

        down_x = x_sign * (ANATOMY['fossa_width'] - 0.1)
        down_y = ANATOMY['dentate_y']

        external_x = x_sign * ANATOMY['fossa_width']
        external_y = ANATOMY['anal_verge'] + 0.15

        control_pts = [
            (internal_x, internal_y),
            (up_x, up_y),
            (over_levator_x, over_levator_y),
            (down_x, down_y),
            (external_x, external_y),
        ]

    elif tract_type == 'extrasphincteric':
        # Outside sphincter complex, often from supralevator
        supra_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                           ANATOMY['eas_thickness'] + 0.1)
        supra_y = ANATOMY['levator_y'] + 0.15

        external_x = x_sign * ANATOMY['fossa_width']
        external_y = (ANATOMY['levator_y'] + ANATOMY['anal_verge']) / 2

        # This type has internal opening higher up (in rectum)
        internal_x = x_sign * ANATOMY['canal_inner'] * 1.2
        internal_y = ANATOMY['rectum_top'] - 0.05

        control_pts = [
            (internal_x, internal_y),
            (supra_x, supra_y),
            (external_x, external_y),
        ]

    else:  # superficial
        # Simple, superficial tract
        external_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['eas_thickness'] + 0.05)
        external_y = ANATOMY['anal_verge'] + 0.05

        control_pts = [
            (internal_x, internal_y),
            (external_x, external_y),
        ]

    # Generate tunnel with granulation tissue
    start = control_pts[0]
    end = control_pts[-1]

    # For complex paths, go through intermediate points
    if len(control_pts) > 2:
        all_left = []
        all_right = []

        for i in range(len(control_pts) - 1):
            left, right = generate_fistula_tunnel(
                control_pts[i], control_pts[i + 1],
                width=0.02 + 0.01 * np.random.rand(),
                seed=seed + i * 100
            )
            all_left.append(left)
            all_right.append(right)

        # Combine segments
        left_combined = np.vstack(all_left)
        right_combined = np.vstack(all_right)
    else:
        left_combined, right_combined = generate_fistula_tunnel(
            start, end, width=0.025, seed=seed
        )

    # Create closed polygon for tunnel
    tunnel_pts = np.vstack([left_combined, right_combined[::-1]])

    # Granulation tissue (inflammatory wall)
    ax.fill(tunnel_pts[:, 0], tunnel_pts[:, 1],
            facecolor=COLORS['fistula_wall'], edgecolor=COLORS['fistula_edge'],
            linewidth=1.2, alpha=0.9, zorder=20)

    # Central lumen (darker)
    center_left = left_combined * 0.9 + right_combined * 0.1
    center_right = right_combined * 0.9 + left_combined * 0.1
    center_pts = np.vstack([center_left, center_right[::-1]])

    ax.fill(center_pts[:, 0], center_pts[:, 1],
            facecolor=COLORS['fistula_fill'], edgecolor='none',
            alpha=0.8, zorder=21)

    # Granulation texture
    n_bumps = 15
    for i in range(n_bumps):
        idx = int(len(left_combined) * np.random.rand())
        if idx < len(left_combined):
            bx = left_combined[idx, 0] + 0.01 * (np.random.rand() - 0.5)
            by = left_combined[idx, 1] + 0.01 * (np.random.rand() - 0.5)
            br = 0.005 + 0.008 * np.random.rand()

            circ = Circle((bx, by), br, facecolor=COLORS['fistula_granulation'],
                         edgecolor='none', alpha=0.5, zorder=22)
            ax.add_patch(circ)

    # Internal opening marker
    ax.scatter([control_pts[0][0]], [control_pts[0][1]],
               c=COLORS['fistula_fill'], s=60, zorder=25,
               edgecolors='white', linewidths=2, marker='o')

    # External opening marker
    ax.scatter([control_pts[-1][0]], [control_pts[-1][1]],
               c=COLORS['fistula_fill'], s=50, zorder=25,
               edgecolors='white', linewidths=1.5, marker='o')


def draw_abscess(ax: plt.Axes, location: str = 'ischioanal', clock: int = 6,
                 size: str = 'medium', seed: int = None) -> None:
    """Draw abscess cavity at specified location."""
    if seed is None:
        seed = hash(location) % (2**31)

    size_map = {'small': 0.06, 'medium': 0.10, 'large': 0.15}
    radius = size_map.get(size, 0.10)

    x_sign = 1 if clock in [1, 2, 3, 4, 5, 11, 12] else -1

    if location == 'ischioanal':
        cx = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                      ANATOMY['eas_thickness'] + 0.12)
        cy = (ANATOMY['levator_y'] + ANATOMY['anal_verge']) / 2

    elif location == 'supralevator':
        cx = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                      ANATOMY['eas_thickness'] + 0.08)
        cy = ANATOMY['levator_y'] + 0.15

    elif location == 'intersphincteric':
        cx = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] * 0.5)
        cy = ANATOMY['dentate_y'] - 0.15
        radius *= 0.6  # Smaller in this space

    elif location == 'perianal':
        cx = x_sign * (ANATOMY['canal_width'] + ANATOMY['eas_thickness'] + 0.08)
        cy = ANATOMY['anal_verge'] + 0.05

    else:
        cx = x_sign * 0.3
        cy = 0

    # Generate abscess shape
    blob = generate_abscess_blob(cx, cy, radius, seed=seed)

    # Inflammatory rim (outer glow)
    rim = generate_abscess_blob(cx, cy, radius * 1.3, seed=seed + 1)
    ax.fill(rim[:, 0], rim[:, 1], facecolor=COLORS['abscess_wall'],
            alpha=0.35, edgecolor='none', zorder=18)

    # Main abscess body
    ax.fill(blob[:, 0], blob[:, 1], facecolor=COLORS['abscess_fill'],
            edgecolor=COLORS['abscess_edge'], linewidth=1.2, alpha=0.85, zorder=19)

    # Central pus (brighter)
    inner_blob = generate_abscess_blob(cx, cy, radius * 0.4, seed=seed + 2)
    ax.fill(inner_blob[:, 0], inner_blob[:, 1], facecolor=COLORS['abscess_core'],
            edgecolor='none', alpha=0.6, zorder=19)


def draw_seton(ax: plt.Axes, tract_type: str, clock: int = 6, seed: int = None) -> None:
    """Draw seton through fistula tract."""
    if seed is None:
        seed = 9000

    np.random.seed(seed)
    x_sign = 1 if clock in [1, 2, 3, 4, 5, 11, 12] else -1

    # Simplified seton path
    internal_y = ANATOMY['dentate_y']
    internal_x = x_sign * ANATOMY['canal_width'] * 0.6

    if tract_type == 'transsphincteric':
        external_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] +
                              ANATOMY['eas_thickness'] + 0.05)
        external_y = ANATOMY['dentate_y'] - 0.2
    else:
        external_x = x_sign * (ANATOMY['canal_width'] + ANATOMY['ias_thickness'] + 0.03)
        external_y = ANATOMY['anal_verge'] + 0.12

    # Generate seton path
    path = generate_wavy_line((internal_x, internal_y), (external_x, external_y),
                              amplitude=0.01, frequency=3, seed=seed)

    # Shadow
    ax.plot(path[:, 0] + 0.005, path[:, 1] - 0.005,
            color='#104E8B', linewidth=4, alpha=0.3, zorder=23)

    # Main seton
    ax.plot(path[:, 0], path[:, 1], color=COLORS['seton_main'],
            linewidth=3.5, solid_capstyle='round', zorder=24)

    # Highlight
    ax.plot(path[:, 0], path[:, 1], color=COLORS['seton_highlight'],
            linewidth=1.5, alpha=0.7, zorder=25)

    # End markers
    for px, py in [(internal_x, internal_y), (external_x, external_y)]:
        ax.scatter([px], [py], c=COLORS['seton_main'], s=50, zorder=26,
                  edgecolors=COLORS['seton_highlight'], linewidths=2)


# =============================================================================
# LABELING SYSTEM
# =============================================================================

def draw_anatomical_labels(ax: plt.Axes) -> None:
    """Draw professional anatomical labels with leader lines."""

    labels = [
        # (text, x_pos, y_pos, ha, pointer_to_x, pointer_to_y)
        ('Rectum', 0, ANATOMY['rectum_top'] + 0.12, 'center', None, None),
        ('Levator Ani', 0.48, ANATOMY['levator_y'] + 0.15, 'left',
         0.35, ANATOMY['levator_y'] + 0.08),
        ('Ext. Sphincter', 0.48, 0.05, 'left',
         ANATOMY['canal_width'] + ANATOMY['ias_thickness'] + ANATOMY['eas_thickness'] * 0.5, 0),
        ('Int. Sphincter', -0.48, 0.05, 'right',
         -(ANATOMY['canal_width'] + ANATOMY['ias_thickness'] * 0.5), 0.05),
        ('Ischioanal\nFossa', -0.42, -0.35, 'right',
         -0.32, -0.35),
        ('Dentate Line', 0.35, ANATOMY['dentate_y'] - 0.02, 'left',
         ANATOMY['canal_width'] * 0.8, ANATOMY['dentate_y']),
    ]

    for label_data in labels:
        text = label_data[0]
        x, y = label_data[1], label_data[2]
        ha = label_data[3]

        # Label text
        ax.text(x, y, text, ha=ha, va='center',
                fontsize=9, fontweight='bold', color=COLORS['text_dark'],
                fontfamily='sans-serif', zorder=50)

        # Leader line if specified
        if len(label_data) > 4 and label_data[4] is not None:
            ptr_x, ptr_y = label_data[4], label_data[5]

            # Calculate line start point (near text)
            if ha == 'left':
                line_start_x = x - 0.02
            elif ha == 'right':
                line_start_x = x + 0.02
            else:
                line_start_x = x

            ax.annotate('', xy=(ptr_x, ptr_y), xytext=(line_start_x, y),
                       arrowprops=dict(arrowstyle='-', color=COLORS['leader'],
                                      linewidth=0.8, shrinkA=2, shrinkB=2),
                       zorder=49)


# =============================================================================
# PROFESSIONAL ELEMENTS
# =============================================================================

def draw_frame_border(ax: plt.Axes) -> None:
    """Draw professional frame border."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    border = Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                       fill=False, edgecolor=COLORS['border'],
                       linewidth=1.5, zorder=100)
    ax.add_patch(border)


def draw_scale_bar(ax: plt.Axes) -> None:
    """Draw 1cm scale bar."""
    x_start = 0.35
    y_pos = -0.90
    bar_len = ANATOMY['scale_1cm']

    ax.plot([x_start, x_start + bar_len], [y_pos, y_pos],
            color=COLORS['border'], linewidth=2.5, solid_capstyle='butt', zorder=50)

    # End caps
    cap_h = 0.02
    ax.plot([x_start, x_start], [y_pos - cap_h, y_pos + cap_h],
            color=COLORS['border'], linewidth=2, zorder=50)
    ax.plot([x_start + bar_len, x_start + bar_len], [y_pos - cap_h, y_pos + cap_h],
            color=COLORS['border'], linewidth=2, zorder=50)

    ax.text(x_start + bar_len / 2, y_pos - 0.04, "1 cm",
            ha='center', va='top', fontsize=9, color=COLORS['text_dark'],
            fontfamily='sans-serif', fontweight='bold', zorder=50)


def draw_legend(ax: plt.Axes, has_abscess: bool, has_seton: bool,
                tract_type: str) -> None:
    """Draw legend."""
    legend_items = []

    # Anatomical structures
    levator_patch = mpatches.Patch(facecolor=COLORS['levator_fill'],
                                   edgecolor=COLORS['levator_edge'],
                                   linewidth=0.8, label='Levator Ani')
    legend_items.append(levator_patch)

    eas_patch = mpatches.Patch(facecolor=COLORS['eas_fill'],
                               edgecolor=COLORS['eas_edge'],
                               linewidth=0.8, label='Ext. Sphincter')
    legend_items.append(eas_patch)

    ias_patch = mpatches.Patch(facecolor=COLORS['ias_fill'],
                               edgecolor=COLORS['ias_edge'],
                               linewidth=0.8, label='Int. Sphincter')
    legend_items.append(ias_patch)

    fat_patch = mpatches.Patch(facecolor=COLORS['fat_fill'],
                               edgecolor=COLORS['fat_edge'],
                               linewidth=0.8, label='Adipose')
    legend_items.append(fat_patch)

    # Fistula tract
    fistula_patch = mpatches.Patch(facecolor=COLORS['fistula_wall'],
                                   edgecolor=COLORS['fistula_edge'],
                                   linewidth=0.8, label=f'Fistula ({tract_type})')
    legend_items.append(fistula_patch)

    if has_abscess:
        abscess_patch = mpatches.Patch(facecolor=COLORS['abscess_fill'],
                                       edgecolor=COLORS['abscess_edge'],
                                       linewidth=0.8, label='Abscess')
        legend_items.append(abscess_patch)

    if has_seton:
        seton_line = Line2D([0], [0], color=COLORS['seton_main'], linewidth=3,
                           label='Seton', marker='o', markersize=5,
                           markerfacecolor=COLORS['seton_main'],
                           markeredgecolor=COLORS['seton_highlight'])
        legend_items.append(seton_line)

    legend = ax.legend(handles=legend_items, loc='upper left',
                      bbox_to_anchor=(-0.02, 1.02),
                      fontsize=7, framealpha=0.95,
                      facecolor=COLORS['canvas'],
                      edgecolor=COLORS['text_light'],
                      borderpad=0.4, labelspacing=0.3)
    legend.get_frame().set_linewidth(0.5)


def draw_header(ax: plt.Axes, case_id: str, severity: str,
                vai_score: Optional[int], magnifi_score: Optional[int],
                title: str = None, tract_type: str = 'transsphincteric') -> None:
    """Draw professional header with Parks classification."""

    # Main title
    display_title = title if title else f"Perianal Fistula - {tract_type.capitalize()}"
    ax.text(0.5, 1.10, display_title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            color=COLORS['text_dark'], fontfamily='sans-serif')

    # Subtitle
    ax.text(0.5, 1.05, "(Coronal Cross-Section - Parks Classification)",
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, color=COLORS['text_medium'],
            fontfamily='sans-serif', style='italic')

    # Severity badge
    badge_colors = {
        'Remission': COLORS['badge_mild'],
        'Mild': COLORS['badge_mild'],
        'Moderate': COLORS['badge_moderate'],
        'Severe': COLORS['badge_severe'],
    }
    badge_color = badge_colors.get(severity, COLORS['text_light'])

    bbox_props = dict(boxstyle='round,pad=0.3,rounding_size=0.15',
                      facecolor=badge_color, edgecolor='none', alpha=0.9)
    ax.text(0.98, 1.10, severity.upper(), transform=ax.transAxes,
            ha='right', va='top', fontsize=9, fontweight='bold',
            color=COLORS['text_white'], bbox=bbox_props, fontfamily='sans-serif')

    # Score box
    if vai_score is not None or magnifi_score is not None:
        parts = []
        if vai_score is not None:
            parts.append(f"VAI: {vai_score}")
        if magnifi_score is not None:
            parts.append(f"MAGNIFI-CD: {magnifi_score}")

        bbox_props = dict(boxstyle='round,pad=0.25', facecolor=COLORS['canvas'],
                         edgecolor=COLORS['text_light'], linewidth=0.5)
        ax.text(0.5, -0.08, "  |  ".join(parts), transform=ax.transAxes,
                ha='center', va='top', fontsize=10, color=COLORS['text_dark'],
                bbox=bbox_props, fontfamily='sans-serif')


# =============================================================================
# MAIN VISUALIZATION FUNCTION - CORONAL VIEW
# =============================================================================

def visualize_case(case_data: Dict, output_path: Optional[Path] = None) -> plt.Figure:
    """
    Generate coronal cross-sectional medical illustration.
    Parks Classification style with Netter aesthetic.
    """
    # Extract case data
    case_id = case_data.get('id', case_data.get('case_id', 'unknown'))
    report_text = case_data.get('report_text', '')
    ground_truth = case_data.get('ground_truth', {})
    severity = case_data.get('severity', 'Unknown')
    title = case_data.get('title', None)

    vai_score = ground_truth.get('expected_vai_score', case_data.get('scored_vai'))
    magnifi_score = ground_truth.get('expected_magnifi_score', case_data.get('scored_magnifi'))

    has_abscess = ground_truth.get('collections_abscesses', False)
    seton_present = 'seton' in report_text.lower() if report_text else False

    # Determine tract type
    tract_type = determine_tract_type(case_data)

    # Extract clock position for laterality
    clock_positions = extract_clock_positions(report_text)
    primary_clock = clock_positions[0] if clock_positions else 6

    # Abscess location inference
    abscess_location = 'ischioanal'
    if report_text:
        text_lower = report_text.lower()
        if 'supralevator' in text_lower:
            abscess_location = 'supralevator'
        elif 'intersphincteric' in text_lower and 'abscess' in text_lower:
            abscess_location = 'intersphincteric'
        elif 'perianal' in text_lower:
            abscess_location = 'perianal'

    # Abscess size
    abscess_size = 'medium'
    if report_text:
        if any(x in report_text.lower() for x in ['large', '4.', '5.']):
            abscess_size = 'large'
        elif any(x in report_text.lower() for x in ['small', 'tiny']):
            abscess_size = 'small'

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 11), facecolor=COLORS['canvas'])
    ax.set_facecolor(COLORS['canvas'])

    ax.set_aspect('equal')
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.98, 0.98)
    ax.axis('off')

    # Unique seed for reproducibility
    case_seed = hash(case_id) % (2**31)
    np.random.seed(case_seed)

    # ===================
    # DRAW ANATOMY (back to front)
    # ===================

    # 1. Ischioanal fossa (fat) - lowest layer
    draw_ischioanal_fossa(ax)

    # 2. Levator ani muscle
    draw_levator_ani(ax)

    # 3. Rectum
    draw_rectum(ax)

    # 4. External anal sphincter
    draw_external_sphincter(ax)

    # 5. Internal anal sphincter
    draw_internal_sphincter(ax)

    # 6. Anal canal with mucosa
    draw_anal_canal(ax)

    # 7. Dentate line
    draw_dentate_line(ax)

    # ===================
    # DRAW PATHOLOGY
    # ===================

    # 8. Abscess (if present)
    if has_abscess:
        draw_abscess(ax, location=abscess_location, clock=primary_clock,
                    size=abscess_size, seed=case_seed + 500)

    # 9. Fistula tract
    draw_fistula_tract(ax, tract_type=tract_type, clock=primary_clock,
                      seed=case_seed + 100)

    # 10. Seton (if present)
    if seton_present:
        draw_seton(ax, tract_type=tract_type, clock=primary_clock,
                  seed=case_seed + 600)

    # ===================
    # PROFESSIONAL ELEMENTS
    # ===================

    draw_anatomical_labels(ax)
    draw_frame_border(ax)
    draw_scale_bar(ax)
    draw_legend(ax, has_abscess, seton_present, tract_type)
    draw_header(ax, case_id, severity, vai_score, magnifi_score,
               title, tract_type)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08)

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor=COLORS['canvas'], edgecolor='none',
                    pad_inches=0.1)
        print(f"  [CORONAL 300DPI] {output_path.name}")

    return fig


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_tract_type(case_data: Dict) -> str:
    """Determine Parks classification from case data."""
    fistula_type = case_data.get('ground_truth', {}).get('fistula_type', '')
    report = case_data.get('report_text', '').lower()

    combined = (fistula_type + ' ' + report).lower()

    if 'inter' in combined and 'sphincteric' in combined:
        return 'intersphincteric'
    if 'trans' in combined and 'sphincteric' in combined:
        return 'transsphincteric'
    if 'supra' in combined and 'sphincteric' in combined:
        return 'suprasphincteric'
    if 'extra' in combined and 'sphincteric' in combined:
        return 'extrasphincteric'
    if 'superficial' in combined or 'subcutaneous' in combined:
        return 'superficial'

    return 'transsphincteric'


def extract_clock_positions(report_text: str) -> List[int]:
    """Extract clock positions from report text."""
    clocks = []
    patterns = [
        r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*o[''']?clock",
        r"(\d{1,2})\s*o[''']?clock",
        r"at\s*(\d{1,2})\s*o[''']?clock",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, report_text.lower()):
            for g in match.groups():
                if g:
                    hour = int(g)
                    if 1 <= hour <= 12 and hour not in clocks:
                        clocks.append(hour)

    if not clocks:
        if 'posterior' in report_text.lower():
            clocks.append(6)
        elif 'anterior' in report_text.lower():
            clocks.append(12)
        elif 'right' in report_text.lower():
            clocks.append(3)
        elif 'left' in report_text.lower():
            clocks.append(9)

    return clocks if clocks else [6]


def load_test_cases(json_path: Path) -> List[Dict]:
    """Load test cases from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'test_cases' in data:
        return data['test_cases']
    elif 'cases' in data:
        return data['cases']
    elif isinstance(data, list):
        return data
    raise ValueError(f"Unknown JSON structure in {json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate coronal cross-sectional visualizations."""
    print("=" * 70)
    print("CORONAL CROSS-SECTIONAL MEDICAL ILLUSTRATION ENGINE")
    print("Parks Classification - Frank Netter Style")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent.parent / "data"
    test_cases_path = data_dir / "parser_validation" / "mega_test_cases.json"

    if not test_cases_path.exists():
        for alt in [data_dir / "training" / "master_cases.json"]:
            if alt.exists():
                test_cases_path = alt
                break

    print(f"\nLoading: {test_cases_path}")

    try:
        cases = load_test_cases(test_cases_path)
        print(f"Cases loaded: {len(cases)}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    case_lookup = {c.get('id', c.get('case_id', '')): c for c in cases}

    # Target cases for demonstration
    target_cases = [
        ('existing_rp_008', 'Mild'),       # Superficial
        ('existing_rp_001', 'Moderate'),   # Intersphincteric with abscess
        ('existing_rp_002', 'Severe'),     # Transsphincteric with abscess
    ]

    # Find seton case
    seton_case = None
    for c in cases:
        if 'seton' in c.get('report_text', '').lower():
            seton_case = c
            break

    print("\n" + "-" * 70)
    print("Generating CORONAL illustrations...")
    print("-" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    for case_id, expected in target_cases:
        case = case_lookup.get(case_id)
        if not case:
            print(f"\n[MISSING] {case_id}")
            continue

        sev = case.get('severity', 'Unknown')
        tract = determine_tract_type(case)
        print(f"\n[{sev.upper()}] {case_id}: {case.get('title', 'N/A')}")
        print(f"    Parks Classification: {tract}")

        try:
            out = OUTPUT_DIR / f"coronal_{case_id}_{sev.lower()}.png"
            fig = visualize_case(case, out)
            plt.close(fig)
            generated.append(out)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if seton_case:
        cid = seton_case.get('id', 'seton')
        sev = seton_case.get('severity', 'Unknown')
        tract = determine_tract_type(seton_case)
        print(f"\n[SETON] {cid}")
        print(f"    Parks Classification: {tract}")
        try:
            out = OUTPUT_DIR / f"coronal_{cid}_seton.png"
            fig = visualize_case(seton_case, out)
            plt.close(fig)
            generated.append(out)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print(f"Generated {len(generated)} coronal illustrations")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    print("\nCoronal View Features:")
    print("  - Vertical cross-section through anal canal")
    print("  - Parks Classification fistula pathing")
    print("  - Levator ani muscle with proper angulation")
    print("  - Ischioanal fossa with adipose tissue texture")
    print("  - Striated EAS vs smooth IAS differentiation")
    print("  - Dentate line landmark")
    print("  - Granulation tissue fistula tunnels")
    print("  - Professional anatomical labels")
    print("  - Sans-serif medical typography")

    return generated


if __name__ == "__main__":
    main()
