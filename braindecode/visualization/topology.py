import itertools

import numpy as np
from mne.channels.layout import _find_topomap_coords
from mne.defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT
from mne.utils import _check_option, _check_sphere, _validate_type

try:
    from mne.io.pick import _MEG_CH_TYPES_SPLIT
except ImportError:
    _MEG_CH_TYPES_SPLIT = ()


def project_to_topomap(data, chs_info, res=64):
    """Project per-channel attribution values onto a 2-D scalp topomap grid.

    Converts a 1-D array of per-channel values into a ``(res, res)`` 2-D map
    by interpolating electrode values across the scalp surface using
    Clough-Tocher triangulation, matching the behaviour of MNE topoplots.

    Parameters
    ----------
    data : numpy.ndarray
        Per-channel values of shape ``(n_chans,)``.
    chs_info : list of dict
        MNE-style channel info list.  Each entry must have a ``'ch_name'`` key
        and a ``'loc'`` array whose first three elements are the 3-D Cartesian
        electrode position in metres.
    res : int, default=64
        Resolution of the output square grid (pixels per side).

    Returns
    -------
    numpy.ndarray
        Interpolated scalp map of shape ``(res, res)``.
    """
    import mne

    ch_names = [ch["ch_name"] for ch in chs_info]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")

    with info._unlock():
        for i, ch in enumerate(chs_info):
            if ch.get("loc") is not None:
                info["chs"][i]["loc"] = np.array(ch["loc"], dtype=np.float64)

    sphere = _check_sphere(None)
    picks = list(range(len(data)))
    pos2d = _find_topomap_coords(info, picks=picks, sphere=sphere)

    extrapolate = _check_extrapolate(_EXTRAPOLATE_DEFAULT, "eeg")
    outlines = _make_head_outlines(sphere, pos2d, "head", (0.0, 0.0))
    _, Xi, Yi, interp = _setup_interp(
        pos2d, res, extrapolate, sphere, outlines, _BORDER_DEFAULT
    )
    interp.set_values(data)
    Z = interp.set_locations(Xi, Yi)()
    return Z


def _check_extrapolate(extrapolate, ch_type):
    _check_option("extrapolate", extrapolate, ("box", "local", "head", "auto"))
    if extrapolate == "auto":
        extrapolate = "local" if ch_type in _MEG_CH_TYPES_SPLIT else "head"
    return extrapolate


def _make_head_outlines(sphere, pos, outlines, clip_origin):
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere

    if outlines in ("head", "skirt", None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        nose_x = np.array([-dx, 0, dx]) * radius + x
        nose_y = np.array([dy, 1.15, dy]) * radius + y
        ear_x = np.array(
            [
                0.497,
                0.510,
                0.518,
                0.5299,
                0.5419,
                0.54,
                0.547,
                0.532,
                0.510,
                0.489,
            ]
        ) * (radius * 2)
        ear_y = (
            np.array(
                [
                    0.0555,
                    0.0775,
                    0.0783,
                    0.0746,
                    0.0555,
                    -0.0055,
                    -0.0932,
                    -0.1313,
                    -0.1384,
                    -0.1199,
                ]
            )
            * (radius * 2)
            + y
        )

        if outlines is not None:
            outlines_dict = dict(
                head=(head_x, head_y),
                nose=(nose_x, nose_y),
                ear_left=(ear_x + x, ear_y),
                ear_right=(-ear_x + x, ear_y),
            )
        else:
            outlines_dict = dict()

        mask_scale = 1.25 if outlines == "skirt" else 1.0
        mask_scale = max(
            mask_scale,
            np.linalg.norm(pos, axis=1).max() * 1.01 / radius,
        )
        outlines_dict["mask_pos"] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict["clip_radius"] = (clip_radius,) * 2
        outlines_dict["clip_origin"] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if "mask_pos" not in outlines:
            raise ValueError("You must specify the coordinates of the image mask.")
    else:
        raise ValueError("Invalid value for `outlines`.")

    return outlines


def _setup_interp(pos, res, extrapolate, sphere, outlines, border):
    xlim = np.inf, -np.inf
    ylim = np.inf, -np.inf
    mask_ = np.c_[outlines["mask_pos"]]
    clip_radius = outlines["clip_radius"]
    clip_origin = outlines.get("clip_origin", (0.0, 0.0))
    xmin = np.min(np.r_[xlim[0], mask_[:, 0], clip_origin[0] - clip_radius[0]])
    xmax = np.max(np.r_[xlim[1], mask_[:, 0], clip_origin[0] + clip_radius[0]])
    ymin = np.min(np.r_[ylim[0], mask_[:, 1], clip_origin[1] - clip_radius[1]])
    ymax = np.max(np.r_[ylim[1], mask_[:, 1], clip_origin[1] + clip_radius[1]])
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = _GridData(pos, extrapolate, clip_origin, clip_radius, border)
    extent = (xmin, xmax, ymin, ymax)
    return extent, Xi, Yi, interp


class _GridData:
    """Unstructured (x, y) data interpolator via Clough-Tocher triangulation."""

    def __init__(self, pos, extrapolate, origin, radii, border):
        assert pos.ndim == 2 and pos.shape[1] == 2, pos.shape
        _validate_type(border, ("numeric", str), "border")
        if isinstance(border, str):
            _check_option("border", border, ("mean",), extra="when a string")

        outer_pts, mask_pts, tri = _get_extra_points(pos, extrapolate, origin, radii)
        self.n_extra = outer_pts.shape[0]
        self.mask_pts = mask_pts
        self.border = border
        self.tri = tri

    def set_values(self, v):
        from scipy.interpolate import CloughTocher2DInterpolator

        if isinstance(self.border, str):
            n_points = v.shape[0]
            v_extra = np.zeros(self.n_extra)
            indices, indptr = self.tri.vertex_neighbor_vertices
            rng = range(n_points, n_points + self.n_extra)
            used = np.zeros(len(rng), bool)
            for idx, extra_idx in enumerate(rng):
                ngb = indptr[indices[extra_idx] : indices[extra_idx + 1]]
                ngb = ngb[ngb < n_points]
                if len(ngb) > 0:
                    used[idx] = True
                    v_extra[idx] = v[ngb].mean()
            if not used.all() and used.any():
                v_extra[~used] = np.mean(v_extra[used])
        else:
            v_extra = np.full(self.n_extra, self.border, dtype=float)

        v = np.concatenate((v, v_extra))
        self.interpolator = CloughTocher2DInterpolator(self.tri, v)
        return self

    def set_locations(self, Xi, Yi):
        self.Xi = Xi
        self.Yi = Yi
        return self

    def __call__(self, *args):
        if len(args) == 0:
            args = [self.Xi, self.Yi]
        return self.interpolator(*args)


def _get_extra_points(pos, extrapolate, origin, radii):
    from scipy.spatial import Delaunay

    radii = np.array(radii, float)
    assert radii.shape == (2,)
    x, y = origin
    _check_option("extrapolate", extrapolate, ("head", "box", "local"))

    mask_pos = None
    if extrapolate == "box":
        extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
        diffs = extremes[1] - extremes[0]
        extremes[0] -= diffs
        extremes[1] += diffs
        eidx = np.array(
            list(itertools.product(*([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1])))
        )
        pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
        outer_pts = extremes[eidx, pidx]
        return outer_pts, mask_pos, Delaunay(np.concatenate((pos, outer_pts)))

    diffs = np.diff(pos, axis=0)
    with np.errstate(divide="ignore"):
        slopes = diffs[:, 1] / diffs[:, 0]
    colinear = (slopes == slopes[0]).all() or np.isinf(slopes).all()

    if colinear or pos.shape[0] < 4:
        dim = 1 if diffs[:, 1].sum() > diffs[:, 0].sum() else 0
        sorting = np.argsort(pos[:, dim])
        pos_sorted = pos[sorting, :]
        diffs_sorted = np.diff(pos_sorted, axis=0)
        distances = np.linalg.norm(diffs_sorted, axis=1)
        distance = np.median(distances)
    else:
        tri = Delaunay(pos, incremental=True)
        idx1, idx2, idx3 = tri.simplices.T
        distances = np.concatenate(
            [
                np.linalg.norm(pos[i1] - pos[i2], axis=1)
                for i1, i2 in zip([idx1, idx2], [idx2, idx3])
            ]
        )
        distance = np.median(distances)

    if extrapolate == "local":
        if colinear or pos.shape[0] < 4:
            edge_points = sorting[[0, -1]]
            line_len = np.diff(pos[edge_points, :], axis=0)
            unit_vec = line_len / np.linalg.norm(line_len) * distance
            unit_vec_par = unit_vec[:, ::-1] * [[-1, 1]]
            edge_pos = pos[edge_points] + np.concatenate([-unit_vec, unit_vec])
            new_pos = np.concatenate([pos + unit_vec_par, pos - unit_vec_par, edge_pos])
            if pos.shape[0] == 3:
                new_pos_diff = pos[..., np.newaxis] - new_pos.T[np.newaxis, :]
                new_pos_diff = np.linalg.norm(new_pos_diff, axis=1)
                good_extra = (new_pos_diff > 0.5 * distance).all(axis=0)
                new_pos = new_pos[good_extra]
            tri = Delaunay(np.concatenate([pos, new_pos]))
            return new_pos, new_pos, tri

        hull_pos = pos[tri.convex_hull]
        channels_center = pos.mean(axis=0)
        radial_dir = hull_pos - channels_center
        unit_radial_dir = radial_dir / np.linalg.norm(
            radial_dir, axis=-1, keepdims=True
        )
        hull_extended = hull_pos + unit_radial_dir * distance
        mask_pos = hull_pos + unit_radial_dir * distance * 0.5
        hull_diff = np.diff(hull_pos, axis=1)[:, 0]
        hull_distances = np.linalg.norm(hull_diff, axis=-1)

        mask_pos = np.unique(mask_pos.reshape(-1, 2), axis=0)
        mask_center = np.mean(mask_pos, axis=0)
        mask_pos -= mask_center
        mask_pos = mask_pos[np.argsort(np.arctan2(mask_pos[:, 1], mask_pos[:, 0]))]
        mask_pos += mask_center

        add_points = []
        eps = np.finfo("float").eps
        n_times_dist = np.round(0.25 * hull_distances / distance).astype(int)
        for n in range(2, n_times_dist.max() + 1):
            mask = n_times_dist == n
            mult = np.arange(1 / n, 1 - eps, 1 / n)[:, np.newaxis, np.newaxis]
            steps = hull_diff[mask][np.newaxis] * mult
            add_points.append(
                (hull_extended[mask, 0][np.newaxis] + steps).reshape((-1, 2))
            )

        hull_extended = np.unique(hull_extended.reshape((-1, 2)), axis=0)
        new_pos = np.concatenate([hull_extended] + add_points)
    else:
        assert extrapolate == "head"
        angle = np.arcsin(distance / np.mean(radii))
        n_pnts = max(12, int(np.round(2 * np.pi / angle)))
        points_l = np.linspace(0, 2 * np.pi, n_pnts, endpoint=False)
        use_radii = radii * 1.1 + distance
        points_x = np.cos(points_l) * use_radii[0] + x
        points_y = np.sin(points_l) * use_radii[1] + y
        new_pos = np.stack([points_x, points_y], axis=1)
        if colinear or pos.shape[0] == 3:
            tri = Delaunay(np.concatenate([pos, new_pos]))
            return new_pos, mask_pos, tri

    tri.add_points(new_pos)
    return new_pos, mask_pos, tri
