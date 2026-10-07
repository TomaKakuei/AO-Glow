"""Native-conjugate mod03 ideal-OPD adapter for local Kaleid integration tests.

All seven rigid groups have dx/dy/dz/tx/ty coordinates. A normalized unit means
1 um or 1 arcmin. Rotation is Ry @ Rx about each group's axial vertex midpoint.
OPD uses a fixed nominal reference sphere and fixed shared archived focus.
Only piston is subtracted; neither tilt nor defocus is fitted per observation.
This is a simulator interface, not a camera/SHWFS model or a trained controller.
"""
from bootstrap import ARCHIVE
import json
import numpy as np
from rayoptics.gui.roafile import open_roa as open_model
from rayoptics.raytr import raytrace
from fast_raytrace import trace_batch
from wavefront_check import at_focus, opd

GROUPS = ((3, 5), (6, 8), (9, 10), (11, 14), (15, 18), (19, 20), (21, 22))
DOFS = ('dx', 'dy', 'dz', 'tx', 'ty')
SCALE = np.array([.001, .001, .001, 1/60, 1/60])  # mm and degrees
FIELDS = ((0., 0.), (0., 1.5), (0., -1.5), (1.5, 0.), (-1.5, 0.),
          (0., 3.), (0., -3.), (3., 0.), (-3., 0.))
D_LINE = 587.5618


class TracePath:
    def __init__(self, path):
        self._path = path

    def path(self, wave):
        return iter(self._path)

    def get_num_surfaces(self):
        return len(self._path)


class Mod03Adapter:
    def __init__(self, grid_size=32, fields=FIELDS, wavelength_nm=D_LINE):
        self.model = open_model(ARCHIVE / 'US07199938-1-mod03.roa')
        self.sm = self.model.seq_model
        assert len(self.sm.ifcs) == 25
        self.wave = float(wavelength_nm)
        self.fields = np.asarray(fields, float)
        self.grid_size = int(grid_size)
        y, x = np.mgrid[-1:1:complex(grid_size), -1:1:complex(grid_size)]
        self.pupil_mask = x*x + y*y <= 1.
        self.pupils = np.vstack(([0., 0.], np.c_[x[self.pupil_mask], y[self.pupil_mask]]))
        self.nominal_origins = np.zeros((25, 3))
        # Omit the 1e10-mm infinity dummy; S1 is the finite launch plane.
        self.nominal_origins[2:, 2] = np.cumsum([g.thi for g in self.sm.gaps[1:]])
        self.n_image = self.sm.gaps[23].medium.rindex(self.wave)
        archive = json.loads((ARCHIVE / 'evidence/native_dense.json').read_text(encoding='utf-8'))
        self.focus_mm = next(r['focus_mm'] for r in archive['results'] if r['label'] == 'candidate')
        self.reference = []
        for field in self.fields:
            raw = self._trace(np.zeros(35), field)
            if not raw['valid'].all():
                raise RuntimeError('Nominal grid is clipped; choose and validate a new protocol explicitly')
            data = self._wave_data(raw, raw['chief_point'], raw['chief_opl'])
            center = np.asarray(at_focus(data, self.focus_mm)['center'])
            self.reference.append(dict(center=center, E=raw['chief_point'].copy(),
                                       chief_opl=raw['chief_opl']))
        self.nominal = self.observe(np.zeros(35))

    def poses(self, q):
        q = np.asarray(q, float)
        if q.shape != (35,) or not np.isfinite(q).all():
            raise ValueError('Expected 35 finite normalized coordinates')
        if np.max(abs(q)) > 1. + 1e-12:
            raise ValueError('Outside registered +/-1 um, +/-1 arcmin test box')
        state = q.reshape(7, 5) * SCALE
        origins = self.nominal_origins.copy()
        rotations = np.broadcast_to(np.eye(3), (25, 3, 3)).copy()
        for (a, b), pose in zip(GROUPS, state):
            tx, ty = np.deg2rad(pose[3:])
            cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
            rotation = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]]) @ np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
            pivot = (self.nominal_origins[a] + self.nominal_origins[b])/2
            origins[a:b+1] = (self.nominal_origins[a:b+1] - pivot) @ rotation.T + pivot + pose[:3]
            rotations[a:b+1] = rotation
        return origins, rotations

    def path(self, q):
        origins, rotations = self.poses(q)
        path = [list(row) for row in self.sm.path(self.wave)]
        for i in range(1, 24):
            path[i][2] = (rotations[i+1].T @ rotations[i],
                          rotations[i].T @ (origins[i+1] - origins[i]))
        return path[1:]

    def launch(self, field, pupils=None):
        pupils = self.pupils if pupils is None else np.asarray(pupils, float)
        slope = np.tan(np.deg2rad(field))
        direction = np.r_[slope, 1.]
        direction /= np.linalg.norm(direction)
        points = np.zeros((len(pupils), 3))
        points[:, :2] = 4.5*pupils - self.sm.gaps[1].thi*slope
        return points, np.tile(direction, (len(pupils), 1))

    def _trace(self, q, field, pupils=None, scalar=False):
        points, directions = self.launch(field, pupils)
        path = self.path(q)
        kwargs = dict(check_apertures=True, first_surf=0, last_surf=22,
                      intersect_obj=False, pt_inside_fuzz=1e-8)
        if scalar:
            packages = []
            from rayoptics.raytr import RayPkg
            from rayoptics.raytr.traceerror import TraceError
            for p, d in zip(points, directions):
                try:
                    packages.append(RayPkg(*raytrace.trace_raw(iter(path), p, d, self.wave, **kwargs)))
                except TraceError:
                    packages.append(None)
        else:
            packages = trace_batch(TracePath(path), points, directions, self.wave, **kwargs)
        valid = np.array([pkg is not None for pkg in packages])
        p = np.full((len(points), 3), np.nan)
        d = p.copy()
        opl = np.full(len(points), np.nan)
        for i, pkg in enumerate(packages):
            if pkg is not None:
                p[i], d[i] = pkg.ray[-2][:2]
                opl[i] = (pkg.op + np.dot(points[i], directions[i])) / self.n_image
        return dict(p=p[1:], d=d[1:], opl=opl[1:], valid=valid[1:],
                    chief_valid=bool(valid[0]), chief_point=p[0], chief_opl=opl[0])

    def _wave_data(self, raw, E, chief_opl):
        keep = raw['valid']
        p, d = raw['p'][keep], raw['d'][keep]
        z = self.sm.gaps[23].thi
        return dict(p=p, d=d, opl=raw['opl'][keep], E=E, chief_t=0., chief_opl=chief_opl,
                    image_z=z, image_xy=p[:, :2]+(z-p[:, 2, None])*d[:, :2]/d[:, 2, None],
                    image_slopes=d[:, :2]/d[:, 2, None])

    def observe(self, q):
        maps, masks, rms, throughput, centroid = [], [], [], [], []
        for field, reference in zip(self.fields, self.reference):
            raw = self._trace(q, field)
            if not raw['chief_valid'] or not raw['valid'].any():
                raise ValueError('Missing chief ray or empty pupil')
            data = self._wave_data(raw, reference['E'], reference['chief_opl'])
            values = opd(data, reference['center']) * self.n_image/(self.wave*1e-6)
            values -= values.mean()
            mask = np.zeros_like(self.pupil_mask)
            mask[self.pupil_mask] = raw['valid']
            phase = np.zeros(mask.shape)
            phase[mask] = values
            maps.append(phase); masks.append(mask)
            rms.append(float(np.sqrt(np.mean(values**2))))
            throughput.append(float(raw['valid'].mean()))
            xy = data['image_xy'] + self.focus_mm*data['image_slopes']
            centroid.append(xy.mean(axis=0))
        return dict(opd_waves=np.array(maps), valid=np.array(masks), wrms_waves=np.array(rms),
                    ray_survival_fraction=np.array(throughput), centroid_mm=np.array(centroid))

    def network_input(self, observation):
        # A new mod03 model must be trained on precisely this signed phase and
        # mask contract. Historical Trepan/Nikon weights are never used here.
        if not np.array_equal(observation['valid'], self.nominal['valid']):
            raise ValueError('Pupil mask changed: reject observation instead of hiding clipping in zero fill')
        return np.asarray(observation['opd_waves'] - self.nominal['opd_waves'], dtype=np.float32)


class Mod03Plant:
    """Controller-facing command/measurement interface; hidden state stays inside plant."""
    def __init__(self, adapter, hidden_state):
        self.adapter = adapter
        self._hidden = np.asarray(hidden_state, float).copy()
        adapter.poses(self._hidden)
        self.measurement_count = 0

    def measure(self, command):
        self.measurement_count += 1
        return self.adapter.observe(self._hidden + np.asarray(command, float))
