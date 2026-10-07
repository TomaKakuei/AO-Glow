"""Original reference-sphere formulas; no comparison experiment import."""
import numpy as np
D_LINE=587.5618

def opd(data, center, with_jac=False):
    q = data['p'] - center
    radius2 = np.sum((center - data['E']) ** 2)
    B = np.sum(q * data['d'], axis=1)
    A = np.sum(q * q, axis=1) - radius2
    disc = B * B - A
    assert np.min(disc) > 0
    root = np.sqrt(disc)
    denom = -B + root
    tsmall = np.divide(A, denom, out=-B - root, where=abs(denom) > 1e-12)
    tlarge = -B + root
    t = np.where(abs(tsmall - data['chief_t']) < abs(tlarge - data['chief_t']), tsmall, tlarge)
    w = data['opl'] + t - data['chief_opl'] - data['chief_t']
    if not with_jac:
        return w
    sphere_points = data['p'] + t[:, None] * data['d']
    jac = (sphere_points - data['E']) / np.sum((sphere_points - center) * data['d'], axis=1)[:, None]
    return (w, jac)

def at_focus(data, dz):
    center = np.array([*(data['image_xy'] + dz * data['image_slopes']).mean(0), data['image_z'] + dz])
    for _ in range(8):
        w, jac = opd(data, center, True)
        w -= w.mean()
        j = jac[:, :2] - jac[:, :2].mean(0)
        delta = np.linalg.lstsq(j, -w, rcond=None)[0]
        center[:2] += delta
        if np.linalg.norm(delta) < 1e-09:
            break
    w = opd(data, center)
    w -= w.mean()
    waves = w / (D_LINE * 1e-06)
    sigma = float(np.sqrt(np.mean(waves ** 2)))
    return dict(rms_waves=sigma, center=center.tolist(), rms_opd_nm=sigma * D_LINE, marechal_strehl_estimate=float(np.exp(-(2 * np.pi * sigma) ** 2)), uniform_phase_coherence=float(abs(np.mean(np.exp(2j * np.pi * waves))) ** 2))
