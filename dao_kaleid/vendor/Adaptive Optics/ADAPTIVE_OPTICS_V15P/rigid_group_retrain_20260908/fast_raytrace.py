"""Batch rays through the CURRENT corrected RayOptics path, without reinterpreting groups.

Spherical intersections, refraction, reflection and aperture tests follow the
installed RayOptics implementation. Unsupported paths use its scalar tracer.
Adapters are scoped to the sampling process; no optical prescription is edited.
"""
from contextlib import contextmanager
import numpy as np
from rayoptics.raytr import analyses, trace, raytrace
from rayoptics.elem.profiles import Spherical
from rayoptics.elem.surface import Circular, Rectangular

STATS = {'batch_calls': 0, 'batch_rays': 0, 'unsupported_paths': 0, 'failed_rays': 0}
ORIGINAL_GRID = analyses.trace_ray_grid


def _inside(ifc, p, fuzz):
    result = np.ones(len(p), bool)
    if not ifc.clear_apertures:
        return np.sqrt(p[:, 0]**2 + p[:, 1]**2) <= ifc.max_aperture + fuzz
    for aperture in ifc.clear_apertures:
        x, y = p[:, 0] - aperture.x_offset, p[:, 1] - aperture.y_offset
        if type(aperture) is Circular:
            result &= np.sqrt(x*x + y*y) <= aperture.radius + fuzz
        elif type(aperture) is Rectangular:
            result &= (abs(x) <= aperture.x_half_width + fuzz) & (abs(y) <= aperture.y_half_width + fuzz)
        else:
            result &= np.asarray([aperture.point_inside(px, py, fuzz) for px, py in p[:, :2]])
    return result


def _intersect(ifc, p, d, z_dir):
    cv = ifc.profile.cv
    cx2 = cv * np.einsum('ij,ij->i', p, p) - 2*p[:, 2]
    b = cv * np.einsum('ij,ij->i', d, p) - d[:, 2]
    s = cx2 / (z_dir*np.sqrt(b*b - cv*cx2) - b)
    point = p + s[:, None]*d
    normal = -cv*point
    normal[:, 2] += 1
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return s, point, normal


def trace_batch(seq, points, directions, wvl, *, check_apertures=False,
                intersect_obj=True, first_surf=1, last_surf=None, pt_inside_fuzz=1e-5):
    """Return RayPkg or None for each ray, in input order."""
    path = list(seq.path(wvl))
    if last_surf is None:
        last_surf = seq.get_num_surfaces()-2
    kwargs = dict(check_apertures=check_apertures, intersect_obj=intersect_obj,
                  first_surf=first_surf, last_surf=last_surf, pt_inside_fuzz=pt_inside_fuzz)
    if any(type(row[0].profile) is not Spherical or hasattr(row[0], 'phase_element') for row in path):
        STATS['unsupported_paths'] += 1
        from rayoptics.raytr.traceerror import TraceError
        result = []
        for p, d in zip(points, directions):
            try:
                result.append(trace.RayPkg(*raytrace.trace(seq, p, d, wvl, **kwargs)))
            except TraceError:
                result.append(None)
        return result
    STATS['batch_calls'] += 1
    STATS['batch_rays'] += len(points)
    p = np.asarray(points, float).copy()
    d = np.asarray(directions, float).copy()
    valid = np.ones(len(p), bool)
    opl = np.zeros(len(p))
    segments = []
    with np.errstate(invalid='ignore', divide='ignore'):
        if intersect_obj:
            _, p, normal = _intersect(path[0][0], p, d, path[0][4])
        else:
            normal = np.tile([0., 0., 1.], (len(p), 1))
        for surf in range(1, len(path)):
            before, after = path[surf-1], path[surf]
            rotation, translation = before[2]
            b4p = (p-translation) @ rotation.T
            b4d = d @ rotation.T
            pp_distance = -np.einsum('ij,ij->i', b4p, b4d)
            pp = b4p + pp_distance[:, None]*b4d
            distance, hit, new_normal = _intersect(after[0], pp, b4d, before[4])
            distance += pp_distance
            segments.append((p, d, distance, normal))
            if first_surf <= surf-1 < last_surf:
                opl += before[3]*distance
            if check_apertures and first_surf <= surf <= last_surf:
                valid &= _inside(after[0], hit, pt_inside_fuzz)
            mode = after[0].interact_mode
            cos_i = np.einsum('ij,ij->i', b4d, new_normal) / np.linalg.norm(new_normal, axis=1)
            if mode == 'transmit':
                ni, no = before[3], after[3]
                ncos = np.copysign(np.sqrt(no*no - ni*ni*(1-cos_i*cos_i)), cos_i)
                outgoing = (ni*b4d + (ncos-ni*cos_i)[:, None]*new_normal)/no
            elif mode == 'reflect':
                outgoing = b4d - 2*cos_i[:, None]*new_normal
            else:
                outgoing = b4d
            valid &= np.isfinite(hit).all(axis=1) & np.isfinite(outgoing).all(axis=1)
            p, d, normal = hit, outgoing, new_normal
        segments.append((p, d, np.zeros(len(p)), normal))
    valid &= np.isfinite(opl)
    STATS['failed_rays'] += int((~valid).sum())
    return [trace.RayPkg([[a[i], b[i], c[i], n[i]] for a, b, c, n in segments], float(opl[i]), wvl)
            if valid[i] else None for i in range(len(p))]


def trace_ray_grid(opt_model, grid_rng, fld, wvl, foc, append_if_none=True,
                   output_filter=None, rayerr_filter=None, **kwargs):
    # Keep scalar filter/error semantics for optional non-wavefront callers.
    if output_filter is not None or rayerr_filter is not None or kwargs.get('use_named_tuples', False):
        return ORIGINAL_GRID(opt_model, grid_rng, fld, wvl, foc, append_if_none,
                             output_filter, rayerr_filter, **kwargs)
    start = np.array(grid_rng[0]); stop = grid_rng[1]; num = grid_rng[2]
    step = np.array((stop-start)/(num-1))
    pupils = []
    for i in range(num):
        for j in range(num):
            pupils.append(start.copy())
            start[1] += step[1]
        start[0] += step[0]; start[1] = grid_rng[0][1]
    osp = opt_model['optical_spec']; seq = opt_model['seq_model']
    points, directions = [], []
    for pupil in pupils:
        if kwargs.get('apply_vignetting', False):
            pupil = fld.apply_vignetting(pupil)
        p, d = osp.ray_start_from_osp(pupil, fld, kwargs.get('pupil_type', 'rel pupil'))
        if not osp['fov'].is_wide_angle and d[2]*seq.z_dir[0] < 0:
            d = -d
        points.append(p); directions.append(d)
    packages = trace_batch(seq, points, directions, wvl,
        check_apertures=kwargs.get('check_apertures', False),
        intersect_obj=False if osp['fov'].is_wide_angle else kwargs.get('intersect_obj', True),
        first_surf=kwargs.get('first_surf', 1), last_surf=kwargs.get('last_surf', seq.get_num_surfaces()-2),
        pt_inside_fuzz=kwargs.get('pt_inside_fuzz', 1e-5))
    return [[[pupils[i*num+j][0], pupils[i*num+j][1], packages[i*num+j]]
             for j in range(num) if append_if_none or packages[i*num+j] is not None] for i in range(num)]


@contextmanager
def trepan_backend():
    original = analyses.trace_ray_grid
    analyses.trace_ray_grid = trace_ray_grid
    try:
        yield
    finally:
        analyses.trace_ray_grid = original


@contextmanager
def nikon_backend():
    from nikon35 import base
    corrected = base.corrected
    original = corrected.HighNAPlant._trace_opl_state

    def batch_opl(self, effective_q):
        corrected.apply_normalized_state(self.opm, effective_q)
        directions = []
        points = []
        for field in self.fields_mm:
            for px, py in self.pupil_nodes:
                ux = corrected.SAMPLE_SIDE_NA*float(px)/corrected.SAMPLE_INDEX
                uy = corrected.SAMPLE_SIDE_NA*float(py)/corrected.SAMPLE_INDEX
                points.append([field, 0., 0.])
                directions.append([ux, uy, np.sqrt(1-ux*ux-uy*uy)])
        packages = trace_batch(self.opm.seq_model, points, directions, corrected.WAVELENGTH_MM*1e6)
        # Retain the exact scalar failure reporting and counters on invalid rays.
        if any(pkg is None for pkg in packages):
            return original(self, effective_q)
        self.ray_attempt_count += len(packages)
        self.ray_trace_count += len(packages)
        assert all(len(pkg.ray) == self.expected_interfaces for pkg in packages)
        return np.array([p.op for p in packages]).reshape(len(self.fields_mm), -1), len(packages)

    corrected.HighNAPlant._trace_opl_state = batch_opl
    try:
        yield
    finally:
        corrected.HighNAPlant._trace_opl_state = original


class NikonSampler:
    """Reuse only the fixed prescription/reference, reset hidden state and RNG per sample."""
    def __init__(self):
        from nikon35 import Plant
        with nikon_backend():
            self.plant = Plant(noise_std=.015)

    def measure(self, q, seed):
        from nikon35 import BIAS
        plant = self.plant
        plant._hidden = np.asarray(q, float).copy()
        plant._plant.hidden_q = BIAS + plant._hidden
        plant._plant.cache.clear()
        plant._plant.request_count = plant._plant.trace_evaluation_count = 0
        plant._plant.ray_attempt_count = plant._plant.ray_trace_count = plant._plant.failed_trace_attempts = 0
        plant.rng = np.random.RandomState(int(seed))
        plant.records.clear(); plant.readings = 0
        with nikon_backend():
            return plant.measure(np.zeros(35))
