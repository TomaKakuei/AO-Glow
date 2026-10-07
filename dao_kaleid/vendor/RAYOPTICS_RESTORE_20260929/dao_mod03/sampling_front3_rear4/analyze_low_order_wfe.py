"""Posthoc pupil-mode decomposition of saved integration endpoints.

Reuses the registered optical engine and states. No controller, camera,
training, or historical dataset is rerun. Fits only surviving pupil samples.
"""
from optics import SamplingOptics
from job_support import RESULTS, atomic_json, sha
import json
import time
import numpy as np


def decompose(observation):
    size = observation['opd_waves'].shape[-1]
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    z_defocus = 2 * (x*x + y*y) - 1
    columns = {'piston': (np.ones_like(x),),
               'piston_defocus': (np.ones_like(x), z_defocus),
               'piston_tilt': (np.ones_like(x), x, y),
               'piston_tilt_defocus': (np.ones_like(x), x, y, z_defocus)}
    output = {}
    for name, basis in columns.items():
        rms, coefficients, orthogonality = [], [], []
        for phase, mask in zip(observation['opd_waves'], observation['valid']):
            values = phase[mask]
            design = np.column_stack([b[mask] for b in basis])
            coef, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
            assert rank == len(basis)
            residual = values - design @ coef
            error = float(np.max(abs(design.T @ residual)) / len(values))
            assert error < 1e-10 * max(1., float(np.max(abs(values))))
            rms.append(float(np.sqrt(np.mean(residual**2))))
            coefficients.append(coef.tolist())
            orthogonality.append(error)
        output[name] = dict(per_field_rms_waves=rms, mean_field_rms_waves=float(np.mean(rms)),
                            max_field_rms_waves=max(rms), coefficients_waves=coefficients,
                            fit_orthogonality_error=max(orthogonality))
    assert np.max(abs(np.asarray(output['piston']['per_field_rms_waves']) - observation['wrms_waves'])) < 1e-9
    return output


def main():
    start = time.perf_counter()
    engine = SamplingOptics()
    output = dict(wavelength_nm=engine.wave, fields_xy_deg=engine.fields.tolist(),
                  pupil_grid=engine.grid_size, focus_mm=engine.focus_mm,
                  scope='posthoc decomposition of the same two saved endpoints and nominal state',
                  fit='Unweighted least squares on surviving entrance-pupil samples; independent coefficients per field',
                  defocus_basis='2*(x*x+y*y)-1 on normalized entrance pupil; x/y tilt fitted only where explicitly labeled',
                  caveat='Pupil polynomial removal is not physical refocusing; per-field fits do not prove common-plane image quality',
                  nominal=decompose(engine.nominal), cases=[])
    maps = {'nominal_opd_waves': engine.nominal['opd_waves'], 'nominal_valid': engine.nominal['valid']}
    for case_id in (50000, 50001):
        source = RESULTS / f'kaleid_case_{case_id}.json'
        saved = json.loads(source.read_text(encoding='utf-8'))
        hidden = np.asarray(saved['hidden_fixture_for_posthoc_audit'])
        case = dict(case_id=case_id, source_sha256=sha(source))
        for name, q in [('initial', hidden), ('terminal', hidden + np.asarray(saved['terminal_command']))]:
            obs = engine.observe(q)
            replay_error = float(np.max(abs(obs['wrms_waves'] - np.asarray(saved[f'{name}_wrms_waves']))))
            assert replay_error < 1e-8, (case_id, name, replay_error)
            case[name] = dict(saved_wfe_replay_max_abs_error=replay_error, modes=decompose(obs))
            maps[f'{case_id}_{name}_opd_waves'] = obs['opd_waves']
            maps[f'{case_id}_{name}_valid'] = obs['valid']
        output['cases'].append(case)
    output['elapsed_seconds'] = time.perf_counter() - start
    atomic_json(RESULTS / 'low_order_wfe.json', output)
    np.savez_compressed(RESULTS / 'low_order_wfe_maps.npz', **maps)
    print('NOMINAL', json.dumps({k: [v['mean_field_rms_waves'], v['max_field_rms_waves']] for k,v in output['nominal'].items()}), flush=True)
    for case in output['cases']:
        for name in ('initial', 'terminal'):
            print(case['case_id'], name, json.dumps({k: [v['mean_field_rms_waves'], v['max_field_rms_waves']] for k,v in case[name]['modes'].items()}), flush=True)
    print('SECONDS', output['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
