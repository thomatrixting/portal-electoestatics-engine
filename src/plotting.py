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
from typing import Optional, Tuple

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

_FINAL_PLOTS_DIR = Path("/mnt/ubuntu/home/thomas/Desktop/portals/portal-gravity-engine/output/final_plots")
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


def _draw_pinned_objects(ax, data, meta, background_rgb: np.ndarray) -> None:
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

            label_dx, label_dy = -dx, -dy
            pad = (h / 2 if axis == "y" else w / 2) + 3
            lx, ly = cx + label_dx * pad, cy + label_dy * pad
            if not (margin <= lx <= sim_width - margin and margin <= ly <= sim_height - margin):
                # opposite-of-arrow side runs off the grid (portal sits at
                # the sim boundary) - fall back to the arrow's own side,
                # which always has room since the arrow points inward.
                label_dx, label_dy = dx, dy
                lx, ly = cx + label_dx * pad, cy + label_dy * pad
        else:
            lx, ly = cx, cy - (h / 2 + 3)
            if ly < margin:
                ly = cy + (h / 2 + 3)

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
              save_path: Optional[str] = None):
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
                       figure if None).
        show:          call plt.show() when a new figure was created.
        show_isolines: draw equipotential lines (matches Simulation's
                       isoline overlay).
        show_vectors:  draw the (-grad_x, -grad_y) vector field.
        title:         custom plot title (defaults to an auto-generated one).
        xlabel:        custom x-axis label (defaults to "x (grid)").
        save_path:     if given, save the figure to this path.
    """
    data, meta = _load(path)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 6))

    rgb = _render_background(ax, data, field, scheme,
                             show_isolines, isoline_count,
                             show_vectors, vector_step)
    _draw_pinned_objects(ax, data, meta, rgb)

    ax.set_title(title or f"{field} ({scheme}) - {meta.get('kind', '?')} @ {meta.get('timestamp', '?')}")
    ax.set_xlabel(xlabel or "x (grid)")
    ax.set_ylabel("y (grid)")

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if created and show:
        plt.show()
    return ax


def plot_trajectories(recording_path, every_n_frames: int = 5,
                      field: str = "potential", scheme: str = "Default",
                      show_isolines: bool = True, isoline_count: int = 10,
                      show_vectors: bool = False, vector_step: int = 10,
                      ax=None, show: bool = True, cmap: str = "plasma",
                      title: Optional[str] = None, xlabel: Optional[str] = None,
                      save_path: Optional[str] = None):
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
    """
    data, meta = _load(recording_path)
    if meta.get("kind") != "recording":
        raise ValueError("plot_trajectories expects a recording_*.npz/.json file")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 6))

    rgb = _render_background(ax, data, field, scheme,
                             show_isolines, isoline_count,
                             show_vectors, vector_step)
    _draw_pinned_objects(ax, data, meta, rgb)

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
    plot_close_portals_scene(show=show)
    plot_equipotencial_field_mom(show=show)
    plot_equipotencial_field_sor(show=show)
    plot_velocities_comparison(show=show)


if __name__ == "__main__":
    plot_equipotencial_scene(show=True)
