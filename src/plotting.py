"""
plotting.py - offline analysis/visualization for the .npz/.json files produced
by Simulation._save_snapshot / _start_recording/_stop_recording (see
simulation.py, "Export" panel section). Reads exported data only - never
touches a live Simulation instance.

Rendering mirrors the live pygame view as closely as possible: the same
color ramps (colors.py), equipotential lines, vector field and portal
orientation arrows.

Two entry points:
    plot_field(path, field=...)          - potential / |E| magnitude, with
                                            every pinned object drawn (fill +
                                            label + portal orientation arrow),
                                            optional isolines/vectors.
    plot_trajectories(recording_path)    - for a recording_*.npz/.json, draws
                                            the same background as
                                            plot_field, then each tracked
                                            MaterialObject's mask outline and
                                            each TestCharge's position every
                                            `every_n_frames` frames, colored
                                            by a hue gradient that advances
                                            with time.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from colors import COLOR_SCHEMES

_FIELD_KEYS = {
    "potential": "potential",
    "gradient": "E_magnitude",
    "grad_x": "grad_x",
    "grad_y": "grad_y",
}

_VECTOR_COLOR = (60 / 255, 60 / 255, 70 / 255)   # matches simulation.py:_render_vectors
_ISOLINE_ALPHA = 80 / 255                        # matches simulation.py:_render_isolines
_FILL_ALPHA = 210 / 255                          # matches simulation.py:_build_portal_render_cache
_ARROW_LEN = 6.0                                 # grid units, matches _render_portal_arrows

_FINAL_PLOTS_DIR = Path("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/final_plots")
_FINAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load(path) -> Tuple[np.lib.npyio.NpzFile, dict]:
    npz_path = Path(path)
    json_path = npz_path.with_suffix(".json")
    data = np.load(npz_path)
    meta = json.loads(json_path.read_text())
    return data, meta


def _field_rgb(data: np.ndarray, scheme: str) -> np.ndarray:
    """Same (H,W,3) uint8 RGB the live sim would show for this data/scheme."""
    if scheme not in COLOR_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}, expected one of {list(COLOR_SCHEMES)}")
    mapper = COLOR_SCHEMES[scheme]()
    return mapper(data)


def _mpl_colormap_and_norm(scheme: str):
    """Builds a matplotlib Colormap + Normalize matching a colors.py
    GradientColorMapper's own control points exactly, so a colorbar drawn
    with them lines up with the imshow render (which uses the mapper
    directly, not matplotlib's colormap machinery)."""
    if scheme not in COLOR_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}, expected one of {list(COLOR_SCHEMES)}")
    mapper = COLOR_SCHEMES[scheme]()
    vmin, vmax = float(mapper.values[0]), float(mapper.values[-1])
    span = vmax - vmin if vmax > vmin else 1.0
    positions = (mapper.values - vmin) / span
    colors = mapper.colors / 255.0
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{scheme}_mpl", list(zip(positions, colors)))
    return cmap, mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def _draw_isolines(ax, data: np.ndarray, isoline_count: int) -> None:
    d_min, d_max = float(np.min(data)), float(np.max(data))
    if d_max - d_min < 1e-9:
        return
    levels = np.linspace(d_min, d_max, isoline_count + 2)[1:-1]
    ax.contour(data, levels=levels, colors="white", alpha=_ISOLINE_ALPHA, linewidths=1.0)


def _draw_vectors(ax, grad_x: np.ndarray, grad_y: np.ndarray, step: int, magnitude: float = 0.5) -> None:
    H, W = grad_x.shape
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    gx = -grad_x[ys][:, xs]
    gy = -grad_y[ys][:, xs]
    xs_grid, ys_grid = np.meshgrid(xs, ys)
    ax.quiver(xs_grid, ys_grid, gx, gy, color=[_VECTOR_COLOR], angles="xy", scale=magnitude)

def _wire_format_coord(ax, arr: np.ndarray, field: str) -> None:
    """Cursor readout shows the underlying scalar field value, not the
    color-mapped RGB triplet imshow would report by default."""
    H, W = arr.shape

    def format_coord(x: float, y: float) -> str:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < H and 0 <= ix < W:
            return f"x={x:.1f} y={y:.1f} {field}={arr[iy, ix]:.4f}"
        return f"x={x:.1f} y={y:.1f}"

    ax.format_coord = format_coord

def _wire_format_coord(ax, arr: np.ndarray, field: str) -> None:
    """Cursor readout shows the underlying scalar field value, not the
    color-mapped RGB triplet imshow would report by default."""
    H, W = arr.shape

    def format_coord(x: float, y: float) -> str:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < H and 0 <= ix < W:
            return f"x={x:.1f} y={y:.1f} {field}={arr[iy, ix]:.4f}"
        return f"x={x:.1f} y={y:.1f}"

    ax.format_coord = format_coord


def _render_background(ax, data, field: str, scheme: str,
                       show_isolines: bool, isoline_count: int,
                       show_vectors: bool, vector_step: int) -> np.ndarray:
    if field not in _FIELD_KEYS:
        raise ValueError(f"unknown field {field!r}, expected one of {list(_FIELD_KEYS)}")

    arr = data[_FIELD_KEYS[field]]
    rgb = _field_rgb(arr, scheme)
    ax.imshow(rgb, origin="upper")
    _wire_format_coord(ax, arr, field)

    if show_isolines:
        _draw_isolines(ax, arr, isoline_count)
    if show_vectors:
        _draw_vectors(ax, data["grad_x"], data["grad_y"], vector_step)

    return rgb


def _luma_text_color(rgb: np.ndarray, x: float, y: float) -> str:
    H, W = rgb.shape[:2]
    ix = int(np.clip(round(x), 0, W - 1))
    iy = int(np.clip(round(y), 0, H - 1))
    r, g, b = rgb[iy, ix]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luma > 140 else "white"


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float, float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, h = xmax - xmin, ymax - ymin
    return xmin, xmax, ymin, ymax, cx, cy, w, h


def _aspect_figsize(width_units: float, height_units: float,
                    base_width: float = 8.0,
                    min_height: float = 3.0, max_height: float = 8.0) -> Tuple[float, float]:
    """Figure size (inches) that keeps the plotted content close to its
    real data aspect ratio, instead of always using a fixed (8, 6) canvas.
    A fixed canvas leaves huge blank padding above/below very wide/short
    data (e.g. an 800x200 capacitor scene), making the saved image look far
    bigger than the actual content."""
    if width_units <= 0:
        return base_width, min_height
    height = base_width * (height_units / width_units)
    height = max(min_height, min(max_height, height))
    return base_width, height


def _draw_pinned_objects(ax, data, meta, background_rgb: np.ndarray,
                         label_below: bool = False) -> None:
    sim_params = meta.get("sim_params", {})
    sim_width = sim_params.get("sim_width", background_rgb.shape[1])
    sim_height = sim_params.get("sim_height", background_rgb.shape[0])
    margin = 2

    for entry in meta.get("pinned_objects", []):
        mask = data[entry["array_key"]]
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        xmin, xmax, ymin, ymax, cx, cy, w, h = bbox

        color = np.array(entry.get("color") or (255, 255, 255)) / 255.0
        ax.contourf(mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=_FILL_ALPHA)
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.0)

        facing_positive = entry.get("facing_positive")
        is_portal = facing_positive is not None

        if is_portal:
            axis = entry.get("normal_axis") or ("y" if w >= h else "x")
            if axis == "y":
                dx, dy = 0.0, (1.0 if facing_positive else -1.0)
            else:
                dx, dy = (1.0 if facing_positive else -1.0), 0.0

            ex, ey = cx + dx * _ARROW_LEN, cy + dy * _ARROW_LEN
            ax.annotate("", xy=(ex, ey), xytext=(cx, cy),
                       arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5))

        # Label sits just above the object, except a PotentialAnchor's
        # label goes just below instead when label_below is set.
        is_anchor = entry.get("type") == "PotentialAnchor"
        sign = 1.0 if (label_below and is_anchor) else -1.0
        lx, ly = cx, cy + sign * (h / 2 + 3)
        if not margin <= ly <= sim_height - margin:
            ly = cy - sign * (h / 2 + 3)

        lx = float(np.clip(lx, margin, sim_width - margin))
        ly = float(np.clip(ly, margin, sim_height - margin))

        label = entry.get("label") or entry.get("type")
        if label:
            text_color = _luma_text_color(background_rgb, lx, ly)
            ax.annotate(label, (lx, ly), color=text_color, fontsize=8,
                       ha="center", va="center")


def plot_field(path, field: str = "potential", scheme: str = "Default",
              ax=None, show: bool = True,
              show_isolines: bool = True, isoline_count: int = 10,
              show_vectors: bool = False, vector_step: int = 10,
              title: Optional[str] = None, xlabel: Optional[str] = None,
              save_path: Optional[str] = None,
              x_ranges: Optional[List[Tuple[float, float]]] = None,
              label_below: bool = False, show_colorbar: bool = False):
    """
    Plots a scalar field from a snapshot or recording export, matching the
    live simulation's rendering: real color ramp, equipotential lines,
    optional vector field, and every pinned/fixed object drawn with its
    fill, label, and (for portals) orientation arrow.

    Args:
        path:          path to a snapshot_*.npz or recording_*.npz file (its
                       sibling .json is loaded automatically).
        field:         "potential" | "gradient" (|E| magnitude) | "grad_x" | "grad_y"
        scheme:        any key from colors.COLOR_SCHEMES (e.g. "Default",
                       "Potential", "Plasma", "Electric", "Fire", "Extra").
        ax:            existing matplotlib Axes to draw into (creates a new
                       figure if None). Incompatible with x_ranges.
        show:          call plt.show() when a new figure was created.
        show_isolines: draw equipotential lines (matches Simulation's
                       isoline overlay).
        show_vectors:  draw the (-grad_x, -grad_y) vector field.
        title:         custom plot title (defaults to an auto-generated one).
        xlabel:        custom x-axis label (defaults to "x (grid)").
        save_path:     if given, save the figure to this path.
        x_ranges:      optional list of (x_min, x_max) grid-coordinate
                       windows to keep visible on the x-axis, e.g.
                       [(0, 300), (700, 800)] to skip an empty middle
                       region - one subplot per window, sharing the y-axis,
                       with diagonal break marks between them (broken
                       x-axis). Requires ax=None.
        label_below:   plot every pinned object's label just below it
                       instead of the default just above.
        show_colorbar: draw a colorbar next to the plot for the raw scalar
                       value <-> color mapping. Most fields here use a
                       fixed, known scale (e.g. potential always 0..1) so
                       this defaults to off, but fields without a fixed
                       range (e.g. "gradient") benefit from one.
    """
    data, meta = _load(path)

    if field not in _FIELD_KEYS:
        raise ValueError(f"unknown field {field!r}, expected one of {list(_FIELD_KEYS)}")
    H, W = data[_FIELD_KEYS[field]].shape

    if x_ranges is not None:
        if ax is not None:
            raise ValueError("x_ranges requires ax=None (it creates its own subplots)")
        return _plot_field_broken_x(data, meta, field, scheme, show,
                                    show_isolines, isoline_count,
                                    show_vectors, vector_step,
                                    title, xlabel, save_path, x_ranges,
                                    label_below, H, show_colorbar)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=_aspect_figsize(W, H))

    rgb = _render_background(ax, data, field, scheme,
                             show_isolines, isoline_count,
                             show_vectors, vector_step)
    _draw_pinned_objects(ax, data, meta, rgb, label_below=label_below)

    ax.set_title(title or f"{field} ({scheme}) - {meta.get('kind', '?')} @ {meta.get('timestamp', '?')}")
    ax.set_xlabel(xlabel or "x (grid)")
    ax.set_ylabel("y (grid)")

    if show_colorbar:
        cmap, norm = _mpl_colormap_and_norm(scheme)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        plt.colorbar(sm, cax=cax, label=field)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if created and show:
        plt.show()
    return ax


def _plot_field_broken_x(data, meta, field: str, scheme: str, show: bool,
                         show_isolines: bool, isoline_count: int,
                         show_vectors: bool, vector_step: int,
                         title: Optional[str], xlabel: Optional[str],
                         save_path: Optional[str],
                         x_ranges: List[Tuple[float, float]],
                         label_below: bool = False,
                         height_units: Optional[float] = None,
                         show_colorbar: bool = False):
    """Same rendering as plot_field, split into one subplot per x_ranges
    window (broken x-axis) so an uninteresting middle region can be
    skipped without distorting the aspect ratio of the parts shown."""
    widths = [x_max - x_min for x_min, x_max in x_ranges]
    figsize = _aspect_figsize(sum(widths), height_units) if height_units else (8, 6)
    fig, axes = plt.subplots(1, len(x_ranges), figsize=figsize, sharey=True,
                             gridspec_kw={"width_ratios": widths, "wspace": 0.06})
    axes = np.atleast_1d(axes)

    for ax in axes:
        rgb = _render_background(ax, data, field, scheme,
                                 show_isolines, isoline_count,
                                 show_vectors, vector_step)
        _draw_pinned_objects(ax, data, meta, rgb, label_below=label_below)

    for ax, (x_min, x_max) in zip(axes, x_ranges):
        ax.set_xlim(x_min, x_max)

    for ax in axes[1:]:
        ax.tick_params(left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    for ax in axes[:-1]:
        ax.spines["right"].set_visible(False)

    d = 0.015  # diagonal break-mark size, in axes-fraction units
    for left_ax, right_ax in zip(axes[:-1], axes[1:]):
        kwargs = dict(transform=left_ax.transAxes, color="k", clip_on=False, linewidth=1)
        left_ax.plot((1 - d, 1 + d), (-d, d), **kwargs)
        left_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs["transform"] = right_ax.transAxes
        right_ax.plot((-d, d), (-d, d), **kwargs)
        right_ax.plot((-d, d), (1 - d, 1 + d), **kwargs)

    axes[0].set_ylabel("y (grid)")
    fig.supxlabel(xlabel or "x (grid)")
    fig.suptitle(title or f"{field} ({scheme}) - {meta.get('kind', '?')} @ {meta.get('timestamp', '?')}")

    if show_colorbar:
        cmap, norm = _mpl_colormap_and_norm(scheme)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="4%", pad=0.15)
        plt.colorbar(sm, cax=cax, label=field)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return axes


def plot_trajectories(recording_path, every_n_frames: int = 5,
                      field: str = "potential", scheme: str = "Default",
                      show_isolines: bool = True, isoline_count: int = 10,
                      show_vectors: bool = False, vector_step: int = 10,
                      ax=None, show: bool = True, cmap: str = "plasma",
                      title: Optional[str] = None, xlabel: Optional[str] = None,
                      save_path: Optional[str] = None,
                      label_below: bool = False):
    """
    Plots the recorded MaterialObject mask outlines and TestCharge positions
    sampled every `every_n_frames` frames, colored by a hue gradient that
    advances with time (earliest = one end of `cmap`, latest = the other),
    over the same background rendering as `plot_field` (field/isolines/
    vectors/pinned objects, captured once at recording start).

    Args:
        recording_path:  path to a recording_*.npz file (its sibling .json is
                          loaded automatically).
        every_n_frames:  sample stride in frames; the very last frame is
                          always included even if it doesn't fall on stride.
        field, scheme, show_isolines, show_vectors: see plot_field.
        ax:              existing matplotlib Axes to draw into.
        show:            call plt.show() when a new figure was created.
        cmap:            matplotlib colormap name for the time-progression hue.
        title:           custom plot title (defaults to an auto-generated one).
        xlabel:          custom x-axis label (defaults to "x (grid)").
        save_path:       if given, save the figure to this path.
        label_below:     plot every pinned object's label just below it
                         instead of the default just above.
    """
    data, meta = _load(recording_path)
    if meta.get("kind") != "recording":
        raise ValueError("plot_trajectories expects a recording_*.npz/.json file")

    if field not in _FIELD_KEYS:
        raise ValueError(f"unknown field {field!r}, expected one of {list(_FIELD_KEYS)}")
    H, W = data[_FIELD_KEYS[field]].shape

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=_aspect_figsize(W, H))

    rgb = _render_background(ax, data, field, scheme,
                             show_isolines, isoline_count,
                             show_vectors, vector_step)
    _draw_pinned_objects(ax, data, meta, rgb, label_below=label_below)

    frame_count = meta["frame_count"]
    stride = max(1, every_n_frames)
    sample_idxs = list(range(0, frame_count, stride))
    if sample_idxs[-1] != frame_count - 1:
        sample_idxs.append(frame_count - 1)

    colormap = mpl.colormaps[cmap]
    colors = colormap(np.linspace(0, 1, len(sample_idxs)))

    for obj_meta in meta.get("material_objects", []):
        stack = data[obj_meta["array_key"]]  # (frame_count, H, W)
        for color, t in zip(colors, sample_idxs):
            frame_mask = stack[t]
            if not np.any(frame_mask):
                continue
            ax.contour(frame_mask.astype(float), levels=[0.5],
                      colors=[color], linewidths=1.2)

    for q_meta in meta.get("test_charges", []):
        pos = data[q_meta["array_key"]]  # (frame_count, 2)
        sampled = pos[sample_idxs]
        ax.scatter(sampled[:, 0], sampled[:, 1], c=colors, s=25,
                  edgecolors="black", linewidths=0.3)

    norm = plt.Normalize(vmin=0, vmax=max(frame_count - 1, 1))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    plt.colorbar(sm, cax=cax, label="frame index (time)")

    ax.set_title(title or f"Trajectories ({field}, {scheme}) - recording @ {meta.get('timestamp', '?')}")
    ax.set_xlabel(xlabel or "x (grid)")
    ax.set_ylabel("y (grid)")

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if created and show:
        plt.show()
    return ax


def _centroid_trajectory(mask_stack: np.ndarray) -> np.ndarray:
    """Per-frame centroid (x,y) of a (n_frames, H, W) boolean mask stack.
    NaN for any frame where the mask is empty."""
    n = mask_stack.shape[0]
    pos = np.full((n, 2), np.nan)
    for t in range(n):
        ys, xs = np.nonzero(mask_stack[t])
        if len(xs):
            pos[t, 0] = xs.mean()
            pos[t, 1] = ys.mean()
    return pos


def _portal_mask_union(data, meta) -> Optional[np.ndarray]:
    """OR of every portal's (static, captured-at-recording-start) mask, used
    to detect when a tracked object/charge is touching a portal."""
    masks = [data[e["array_key"]] for e in meta.get("pinned_objects", [])
             if e.get("facing_positive") is not None]
    if not masks:
        return None
    union = masks[0].copy()
    for m in masks[1:]:
        union |= m
    return union


def _contact_frames(portals_mask: Optional[np.ndarray],
                    mask_stack: Optional[np.ndarray] = None,
                    positions: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Per-frame bool: is this object touching a portal that frame?"""
    if portals_mask is None:
        return None
    if mask_stack is not None:
        return np.array([np.any(mask_stack[t] & portals_mask)
                         for t in range(mask_stack.shape[0])])
    H, W = portals_mask.shape
    contact = np.zeros(len(positions), dtype=bool)
    for t, (x, y) in enumerate(positions):
        if np.isnan(x) or np.isnan(y):
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < H and 0 <= ix < W:
            contact[t] = portals_mask[iy, ix]
    return contact


_VELOCITY_TITLES = {"speed": "|v| - velocity magnitude", "vx": "vx", "vy": "vy"}


def plot_velocities(recording_path, every_n_frames: int = 10, show: bool = True,
                    components: Tuple[str, ...] = ("speed", "vx", "vy"),
                    title: Optional[str] = None, xlabel: Optional[str] = None,
                    save_path: Optional[str] = None):
    """
    Plots velocity magnitude and/or its x/y components vs. frame index, one
    subplot per requested component, each with one line per tracked object
    (MaterialObject centroid or TestCharge position).

    No absolute dt is recorded, so velocity is a displacement (grid units /
    frame). Positions are subsampled every `every_n_frames` frames before
    differentiating (np.gradient against the actual sampled frame indices,
    so uneven spacing at the tail is handled correctly) - many frames have
    little/no movement, so differentiating every single frame gives a
    jittery curve; sampling first smooths it out.

    Any sample where the object is touching a portal (or whose gradient
    stencil includes a neighboring sample that is) is dropped - contact
    with a portal is a teleport event, and the resulting centroid jump is
    not a real velocity.

    Args:
        components: which subplots to draw, e.g. ("speed",) for just |v|,
                    or any subset/order of "speed", "vx", "vy".
        title:      if given, set as an overall figure suptitle (in addition
                    to the per-component subplot titles).
        xlabel:     custom x-axis label for the bottom subplot (defaults to
                    "frame").
        save_path:  if given, save the figure to this path.
    """
    data, meta = _load(recording_path)
    if meta.get("kind") != "recording":
        raise ValueError("plot_velocities expects a recording_*.npz/.json file")

    unknown = set(components) - set(_VELOCITY_TITLES)
    if unknown:
        raise ValueError(f"unknown component(s) {unknown}, expected subset of {list(_VELOCITY_TITLES)}")

    portals_mask = _portal_mask_union(data, meta)

    objects = []
    for obj_meta in meta.get("material_objects", []):
        stack = data[obj_meta["array_key"]]
        pos = _centroid_trajectory(stack)
        contact = _contact_frames(portals_mask, mask_stack=stack)
        objects.append((obj_meta.get("label") or f"MaterialObject {obj_meta['index']}", pos, contact))
    for q_meta in meta.get("test_charges", []):
        pos = data[q_meta["array_key"]]
        contact = _contact_frames(portals_mask, positions=pos)
        objects.append((f"TestCharge {q_meta['index']}", pos, contact))

    fig, axes = plt.subplots(len(components), 1, figsize=(9, 3 * len(components)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    stride = max(1, every_n_frames)

    for label, pos, contact in objects:
        n = len(pos)
        idxs = list(range(0, n, stride))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        frames = np.asarray(idxs, dtype=float)
        sampled = pos[idxs]

        vx = np.gradient(sampled[:, 0], frames)
        vy = np.gradient(sampled[:, 1], frames)
        speed = np.hypot(vx, vy)
        values = {"speed": speed, "vx": vx, "vy": vy}

        if contact is not None:
            contact_sampled = contact[idxs]
            bad = contact_sampled.copy()
            bad[:-1] |= contact_sampled[1:]
            bad[1:] |= contact_sampled[:-1]
            for arr in values.values():
                arr[bad] = np.nan

        for component, ax in zip(components, axes):
            ax.plot(frames, values[component], label=label)

    for component, ax in zip(components, axes):
        ax.set_title(_VELOCITY_TITLES[component])
        ax.set_ylabel("velocity (grid units / frame)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel(xlabel or "frame")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return axes


def _plot_speed_into_ax(ax, recording_path, every_n_frames: int = 50) -> None:
    """Adapts plot_velocities' speed-only computation to draw into an
    externally-provided Axes (plot_velocities always builds its own figure,
    so this is used when several recordings need to share one figure)."""
    data, meta = _load(recording_path)
    portals_mask = _portal_mask_union(data, meta)

    objects = []
    for obj_meta in meta.get("material_objects", []):
        stack = data[obj_meta["array_key"]]
        pos = _centroid_trajectory(stack)
        contact = _contact_frames(portals_mask, mask_stack=stack)
        objects.append((obj_meta.get("label") or f"MaterialObject {obj_meta['index']}", pos, contact))
    for q_meta in meta.get("test_charges", []):
        pos = data[q_meta["array_key"]]
        contact = _contact_frames(portals_mask, positions=pos)
        objects.append((f"TestCharge {q_meta['index']}", pos, contact))

    stride = max(1, every_n_frames)
    for label, pos, contact in objects:
        n = len(pos)
        idxs = list(range(0, n, stride))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        frames = np.asarray(idxs, dtype=float)
        sampled = pos[idxs]

        vx = np.gradient(sampled[:, 0], frames)
        vy = np.gradient(sampled[:, 1], frames)
        speed = np.hypot(vx, vy)

        if max_value is not None:
            bad_values = speed > max_value
            vx[bad_values] = np.nan
            vy[bad_values] = np.nan
            speed[bad_values] = np.nan

        if contact is not None:
            contact_sampled = contact[idxs]
            bad = contact_sampled.copy()
            bad[:-1] |= contact_sampled[1:]
            bad[1:] |= contact_sampled[:-1]
            speed[bad] = np.nan

        ax.plot(frames, speed, label=label)

    ax.set_ylabel("velocidad (unidades de grilla / frame)")
    ax.set_xlabel("frame")
    ax.legend()
    ax.grid(True)


def _plot_distance_grid(entries, base: Path, suptitle: str, save_name: str, show: bool):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (fname, vdist) in zip(axes.flat, entries):
        plot_field(base / fname, ax=ax, show=False, scheme="Extra",
                  show_isolines=True, show_vectors=True,
                  title=f"d = {vdist}")

    fig.suptitle(suptitle, fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.06,
                        wspace=0.05, hspace=0.15)
    fig.savefig(str(_FINAL_PLOTS_DIR / save_name), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def plot_close_portals_scene(show: bool = True):
    """Dos figuras de 2x2 subgraficas (ejes x/y compartidos): campo
    potencial para las distintas distancias VERTICALES entre portales
    guardadas en output/close_portals/ (0, 40, 80, 120), una para MOM y
    otra para SOR."""
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/close_portals")

    mom_entries = [
        ("snapshot_mom_0d_close.npz", 0),
        ("snapshot_mom_40d_close.npz", 40),
        ("snapshot_mom_80d_close.npz", 80),
        ("snapshot_mom_120d_close.npz", 120),
    ]
    fig_mom = _plot_distance_grid(
        mom_entries, base,
        "Campo potencial (MOM) para distintas distancias verticales entre portales",
        "campo_por_distancia_mom.png", show)

    sor_entries = [
        ("snapshot_sor_0d.npz", 0),
        ("snapshot_sor_40d_close.npz", 40),
        ("snapshot_sor_80d_close.npz", 80),
        ("snapshot_sor_120d_close.npz", 120),
    ]
    fig_sor = _plot_distance_grid(
        sor_entries, base,
        "Campo potencial (SOR) para distintas distancias verticales entre portales",
        "campo_por_distancia_sor.png", show)

    return fig_mom, fig_sor


def plot_equipotencial_field_mom(show: bool = True):
    """Figura independiente: campo potencial equipotencial, MOM."""
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials")
    ax = plot_field(base / "snapshot_mom_800_400.npz", show=show,
                    scheme="Extra", show_isolines=True, show_vectors=True,
                    title="Campo equipotencial - MOM",
                    save_path=str(_FINAL_PLOTS_DIR / "equipotential_field_mom.png"))
    return ax


def plot_equipotencial_field_sor(show: bool = True):
    """Figura independiente: campo potencial equipotencial, SOR."""
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials")
    ax = plot_field(base / "snapshot_sor_800_400.npz", show=show,
                    scheme="Extra", show_isolines=True, show_vectors=True,
                    title="Campo equipotencial - SOR",
                    save_path=str(_FINAL_PLOTS_DIR / "equipotential_field_sor.png"))
    return ax


def plot_velocities_comparison(show: bool = True):
    """Grafica de 1x2 subgraficas: velocidad absoluta de los objetos
    rastreados, MOM vs SOR, sobre la misma geometria (800x400)."""
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_speed_into_ax(axes[0], base / "recording_mom_800_400.npz", every_n_frames=50)
    axes[0].set_title("Velocidad absoluta - MOM")
    _plot_speed_into_ax(axes[1], base / "recording_sor_800_400.npz", every_n_frames=50)
    axes[1].set_title("Velocidad absoluta - SOR")
    fig.suptitle("Comparación de velocidad absoluta: MOM vs SOR")
    plt.tight_layout()
    fig.savefig(str(_FINAL_PLOTS_DIR / "velocities_comparison.png"), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def plot_oscillating_object_field(show: bool = True):
    """Figura independiente: campo potencial de la escena del objeto
    oscilante (conductor cargado oscilando entre los portales, MOM)."""
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/oscilatiing_object")
    ax = plot_field(base / "snapshot_mom_oscilationg.npz", show=show,
                    scheme="Extra", show_isolines=False, show_vectors=True, field="gradient",
                    title="Gradiente del campo - objeto oscilante (MOM)",
                    save_path=str(_FINAL_PLOTS_DIR / "campo_objeto_oscilante.png"),
                    show_colorbar=True)
    return ax


def plot_oscillating_object_trajectory(show: bool = True):
    """Figura independiente: trayectoria del conductor oscilante sobre el
    campo potencial de fondo (MOM)."""
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/oscilatiing_object")
    ax = plot_trajectories(base / "recording_mom_oscilationg.npz", every_n_frames=20,
                           scheme="Extra", show_isolines=True, show_vectors=True,
                           title="Trayectoria del objeto oscilante (MOM)",
                           save_path=str(_FINAL_PLOTS_DIR / "trayectoria_objeto_oscilante.png"),
                           show=show)
    return ax

def plot_capacitor(show = True):
    base = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/capacitor")
    plot_field(base/"snapshot_capacitor_MOM.npz",scheme="Extra", show_isolines=True, show_vectors=False, field="potential",
                        title="Campo equipotencial para un capacitor (MOM)",
                        save_path=str(_FINAL_PLOTS_DIR / "capacitor_mom_raw.png"),isoline_count=20,
                        x_ranges=[(0, 275), (600, 775)])
    plot_field(base/"snapshot_capacitor_MOM_corrected.npz",scheme="Extra", show_isolines=True, show_vectors=False, field="potential",
                        title="Campo equipotencial corregido para un capacitor (MOM)",
                        save_path=str(_FINAL_PLOTS_DIR / "capacitor_mom_corrected.png"),isoline_count=20,
                        x_ranges=[(0, 275), (600, 775)],label_below=True)


def plot_equipotencial_scene(show = True):

    #SOR 800x400 objects

    snapshot_path = "/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials/snapshot_sor_800_400.npz"
    recording_path = "/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials/recording_sor_800_400.npz"

    safe_path = Path(snapshot_path)
    # Comentado: ya generado, no queremos que se abra una ventana cada vez
    # que se corre el archivo. Descomentar si se necesita regenerar.
    # plot_field(snapshot_path, field="potential", scheme="Extra",
    #           show_isolines=True, show_vectors=True, show=False,title="Equipotential field with portals and fixed potentials SOR",save_path=str(safe_path.with_name("equipotential_field_sor.png")))
    # plot_trajectories(recording_path, every_n_frames=50, field="potential", show=False, scheme="Extra", show_isolines=True, title="Trajectories of tracked objects SOR", save_path=str(safe_path.with_name("trajectories_sor.png")))
    # plot_velocities(recording_path, show=False,components=["speed"],every_n_frames=50, title="Velocities of tracked objects SOR", save_path=str(safe_path.with_name("velocities_sor.png")))

    # MOM 800x400 objects
    snapshot_path = "/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials/snapshot_mom_800_400.npz"
    recording_path = "/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/equipotencials/recording_mom_800_400.npz"

    safe_path = Path(snapshot_path)

    # Comentado por el mismo motivo que el bloque SOR de arriba.
    # plot_field(snapshot_path, field="potential", scheme="Extra",
    #           show_isolines=True, show_vectors=True, show=False,title="Equipotential field with portals and fixed potentials MOM",save_path=str(safe_path.with_name("equipotential_field_mom.png")))
    # plot_trajectories(recording_path, every_n_frames=50, field="potential", show=False, scheme="Extra", show_isolines=True, title="Trajectories of tracked objects MOM", save_path=str(safe_path.with_name("trajectories_mom.png")))
    # plot_velocities(recording_path, show=False,components=["speed"],every_n_frames=50, title="Velocities of tracked objects MOM", save_path=str(safe_path.with_name("velocities_mom.png")))

    # Graficas finales (final_plots) - unicas que se muestran (show=True)
    #plot_close_portals_scene(show=show)
    #plot_equipotencial_field_mom(show=show)
    #plot_equipotencial_field_sor(show=show)
    #plot_velocities_comparison(show=show)
    plot_oscillating_object_field(show=show)
    #plot_oscillating_object_trajectory(show=show)
    plot_capacitor(show=show)

def _flux_from_mask(mask: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray,
                     portals_mask: np.ndarray) -> float:
    """Réplica offline de MaterialObject.compute_flux (portals.py), operando
    sobre máscaras ya evaluadas del snapshot en vez de un objeto vivo."""
    m = mask.astype(np.float64)
    if not np.any(m):
        return 0.0
    dmy, dmx = np.gradient(m)
    nx, ny = -dmx, -dmy
    norm = np.hypot(nx, ny)
    boundary = (norm > 1e-9) & (m > 0.5) & ~portals_mask
    if not np.any(boundary):
        return 0.0
    nx_b, ny_b = nx[boundary] / norm[boundary], ny[boundary] / norm[boundary]
    Ex, Ey = -grad_x[boundary], -grad_y[boundary]
    return float(np.sum(Ex * nx_b + Ey * ny_b))


def _portals_mask_from_meta(data, meta) -> np.ndarray:
    """Reconstruye la máscara de portales (para excluir del flujo), igual
    que Simulation._portals_mask pero desde el JSON exportado."""
    mask = np.zeros_like(data["potential"], dtype=bool)
    for obj in meta["pinned_objects"]:
        if obj["type"] in ("Portal", "FixedPotentialPortal"):
            mask |= data[obj["array_key"]]
    dilated = mask.copy()
    dilated[1:, :]  |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:]  |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    return dilated

def plot_triple_portal_scene(show: bool = True):
    """1x2: equipotencial del triple portal, MOM vs SOR."""
    base = Path("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/triple_portals")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_field(base / "snapshot_mom_triple.npz", scheme="Extra",
               show_isolines=True, show_vectors=True, ax=axes[0], show=False,
               title="Triple portal - MOM")
    plot_field(base / "snapshot_sor_triple.npz", scheme="Extra",
               show_isolines=True, show_vectors=True, ax=axes[1], show=False,
               title="Triple portal - SOR")
    plt.tight_layout()
    fig.savefig(str(_FINAL_PLOTS_DIR / "triple_portal_mom_vs_sor.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_material_flux_zero_test(show: bool = True):
    """1x2: sonda neutra aislada, Φ anotado sobre el campo, MOM vs SOR."""
    base = Path("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/flux_zero_test")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tag in zip(axes, ["mom", "sor"]):
        data, meta = _load(base / f"snapshot_{tag}_flux_zero.npz")
        portals_mask = _portals_mask_from_meta(data, meta)
        probe = next(o for o in meta["pinned_objects"] if o["type"] == "MaterialObject")
        flux = _flux_from_mask(data[probe["array_key"]], data["grad_x"], data["grad_y"], portals_mask)
        plot_field(base / f"snapshot_{tag}_flux_zero.npz", scheme="Extra", ax=ax, show=False,
                   title=f"{tag.upper()}:  \u03a6 = {flux:+.4e}")
    plt.tight_layout()
    fig.savefig(str(_FINAL_PLOTS_DIR / "flux_zero_test.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_triple_portal_flux(show: bool = True):
    """Φ por portal individual + Φ del grupo completo, MOM vs SOR,
    mostrando la discrepancia Φ1+Φ2+Φ3 vs Φ_total."""
    base = Path("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/triple_portal_flux")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tag in zip(axes, ["mom", "sor"]):
        data, meta = _load(base / f"snapshot_{tag}_triple_flux.npz")
        portals_mask = _portals_mask_from_meta(data, meta)
        probes = {o["label"]: data[o["array_key"]]
                  for o in meta["pinned_objects"] if o["type"] == "MaterialObject"}
        fluxes = {label: _flux_from_mask(m, data["grad_x"], data["grad_y"], portals_mask)
                  for label, m in probes.items()}
        suma = fluxes["\u03a6 portal 1"] + fluxes["\u03a6 portal 2"] + fluxes["\u03a6 portal 3"]
        total = fluxes.get("Φ grupo total", sum(fluxes.values()))
        ax.bar(list(fluxes.keys()), list(fluxes.values()))
        ax.set_title(f"{tag.upper()}   |\u03a3-total| = {abs(suma-total):.3e}")
        ax.tick_params(axis='x', rotation=20)
        ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(str(_FINAL_PLOTS_DIR / "triple_portal_flux_discrepancy.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig

    base = Path("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/object_at_portal")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tag in zip(axes, ["mom", "sor"]):
        data, meta = _load(base / f"snapshot_{tag}_at_portal.npz")
        portals_mask = _portals_mask_from_meta(data, meta)
        probe = next(o for o in meta["pinned_objects"] if o["type"] == "MaterialObject")
        flux = _flux_from_mask(data[probe["array_key"]], data["grad_x"], data["grad_y"], portals_mask)
        plot_field(base / f"snapshot_{tag}_at_portal.npz", scheme="Extra", ax=ax, show=False,
                   title=f"{tag.upper()}:  \u03a6 = {flux:+.4e}")
    plt.tight_layout()
    fig.savefig(str(_FINAL_PLOTS_DIR / "flux_at_portal.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
def _flux_series_from_recording(data, meta, obj_index: int = 0):
    """Φ(t) para el material_object obj_index a lo largo de toda la
    grabación. El campo (grad_x/grad_y) se capturó una sola vez al iniciar
    la grabación, así que se reutiliza el mismo para cada frame."""
    portals_mask = _portals_mask_from_meta(data, meta)
    obj_meta = meta["material_objects"][obj_index]
    stack = data[obj_meta["array_key"]]          # (n_frames, H, W)
    grad_x, grad_y = data["grad_x"], data["grad_y"]
    n = stack.shape[0]
    flux = np.full(n, np.nan)
    for t in range(n):
        flux[t] = _flux_from_mask(stack[t], grad_x, grad_y, portals_mask)
    return flux, obj_meta
def render_flux_video(recording_path, save_path=None, obj_index: int = 0,
                      field: str = "potential", scheme: str = "Extra",
                      show_isolines: bool = True, isoline_count: int = 10,
                      every_n_frames: int = 1, fps: int = 20, dpi: int = 120):
    """Video del objeto cayendo, con Φ recalculado y anotado cuadro a
    cuadro sobre el campo estático capturado al iniciar la grabación."""
    import matplotlib.animation as animation

    data, meta = _load(recording_path)
    if meta.get("kind") != "recording":
        raise ValueError("render_flux_video expects a recording_*.npz/.json file")

    flux, obj_meta = _flux_series_from_recording(data, meta, obj_index)
    portals_mask = _portals_mask_from_meta(data, meta)
    stack = data[obj_meta["array_key"]]
    n_frames = stack.shape[0]
    idxs = list(range(0, n_frames, max(1, every_n_frames)))
    if idxs[-1] != n_frames - 1:
        idxs.append(n_frames - 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    rgb = _render_background(ax, data, field, scheme, show_isolines,
                             isoline_count, False, 10)
    _draw_pinned_objects(ax, data, meta, rgb)
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")

    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=12,
                  va="top", ha="left", color="white",
                  bbox=dict(facecolor="black", alpha=0.6, pad=4))
    state = {"artist": None}

    def _clear():
        if state["artist"] is not None:
            try:
                state["artist"].remove()
            except (AttributeError, NotImplementedError):
                for coll in getattr(state["artist"], "collections", []):
                    coll.remove()
            state["artist"] = None

    def update(t):
        _clear()
        mask = stack[t]
        if np.any(mask):
            state["artist"] = ax.contourf(mask.astype(float), levels=[0.5, 1.5],
                                          colors=[(0.3, 0.9, 0.4)], alpha=0.8)
        touching = bool(np.any(mask & portals_mask))
        nota = " (contacto con portal)" if touching else ""
        txt.set_text(f"frame {t}\n\u03a6 = {flux[t]:+.3f}{nota}")
        return [txt]

    anim = animation.FuncAnimation(fig, update, frames=idxs, blit=False)

    if save_path is None:
        save_path = str(_FINAL_PLOTS_DIR / "falling_flux.mp4")
    try:
        anim.save(save_path, fps=fps, dpi=dpi, writer="ffmpeg")
    except Exception:
        save_path = str(Path(save_path).with_suffix(".gif"))
        anim.save(save_path, fps=fps, dpi=dpi, writer="pillow")

    plt.close(fig)
    print(f"[plotting] video guardado en {save_path}")
    return save_path
    
def plot_flux_trajectory(recording_path, obj_index: int = 0, every_n_frames: int = 5,
                         field: str = "potential", scheme: str = "Extra", cmap: str = "coolwarm",
                         show_isolines: bool = True, isoline_count: int = 10,
                         title: Optional[str] = None, save_path: Optional[str] = None,
                         show: bool = True):
    """Izquierda: trayectoria sobre el campo, coloreada por Φ en cada punto
    muestreado. Derecha: Φ vs frame (cuadros de contacto con portal
    descartados, igual que en plot_velocities)."""
    data, meta = _load(recording_path)
    if meta.get("kind") != "recording":
        raise ValueError("plot_flux_trajectory expects a recording_*.npz/.json file")

    flux, obj_meta = _flux_series_from_recording(data, meta, obj_index)
    portals_mask = _portals_mask_from_meta(data, meta)
    stack = data[obj_meta["array_key"]]
    pos = _centroid_trajectory(stack)
    contact = _contact_frames(portals_mask, mask_stack=stack)

    n = stack.shape[0]
    idxs = list(range(0, n, max(1, every_n_frames)))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    fig, (ax_traj, ax_flux) = plt.subplots(1, 2, figsize=(15, 6),
                                           gridspec_kw={"width_ratios": [1.3, 1]})

    rgb = _render_background(ax_traj, data, field, scheme, show_isolines,
                             isoline_count, False, 10)
    _draw_pinned_objects(ax_traj, data, meta, rgb)

    finite = flux[~np.isnan(flux)]
    vmax = np.max(np.abs(finite)) if finite.size else 1.0
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    colormap = mpl.colormaps[cmap]

    sampled_pos, sampled_flux = pos[idxs], flux[idxs]
    valid = ~np.isnan(sampled_pos[:, 0])
    sc = ax_traj.scatter(sampled_pos[valid, 0], sampled_pos[valid, 1],
                         c=sampled_flux[valid], cmap=colormap, norm=norm,
                         s=35, edgecolors="black", linewidths=0.4)
    divider = make_axes_locatable(ax_traj)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    plt.colorbar(sc, cax=cax, label="\u03a6 (flujo)")
    ax_traj.set_title("Trayectoria coloreada por flujo")
    ax_traj.set_xlabel("x (grid)")
    ax_traj.set_ylabel("y (grid)")

    flux_plot = flux.copy()
    bad = contact.copy()
    bad[:-1] |= contact[1:]
    bad[1:] |= contact[:-1]
    flux_plot[bad] = np.nan
    ax_flux.plot(np.arange(n), flux_plot, color="tab:red")
    ax_flux.axhline(0, color="black", linewidth=0.8)
    ax_flux.set_title("\u03a6 vs frame")
    ax_flux.set_xlabel("frame")
    ax_flux.set_ylabel("\u03a6 (flujo)")
    ax_flux.grid(True)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path is None:
        save_path = str(_FINAL_PLOTS_DIR / "flux_trajectory.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig

def plot_flux_at_portal(show: bool = True):
    base = Path("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/object_at_portal")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tag in zip(axes, ["mom", "sor"]):
        data, meta = _load(base / f"snapshot_{tag}_at_portal.npz")
        portals_mask = _portals_mask_from_meta(data, meta)
        probe = next(o for o in meta["pinned_objects"] if o["type"] == "MaterialObject")
        flux = _flux_from_mask(data[probe["array_key"]], data["grad_x"], data["grad_y"], portals_mask)
        plot_field(base / f"snapshot_{tag}_at_portal.npz", scheme="Extra", ax=ax, show=False,
                   title=f"{tag.upper()}:  \u03a6 = {flux:+.4e}")
    plt.tight_layout()
    fig.savefig(str(_FINAL_PLOTS_DIR / "flux_at_portal.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    plot_equipotencial_scene(show=True)
    
    '''
    plot_triple_portal_scene(show=False)
    plot_material_flux_zero_test(show=False)
    plot_triple_portal_flux(show=False)
    plot_flux_at_portal(show=False)
    render_flux_video("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/falling_flux/recording_mom_falling.npz",
                       save_path=str(_FINAL_PLOTS_DIR / "falling_flux_mom.mp4"))
    plot_flux_trajectory("/home/tomas/Documentos/portal-gravity-electoestatics-engine/output/falling_flux/recording_mom_falling.npz",
                          title="Flujo durante la caída - MOM",
                          save_path=str(_FINAL_PLOTS_DIR / "flux_trajectory_mom.png"))
    '''

