"""Observation-only candidate: isolated learned acceptance, measured joint GN.

No simulator, hidden state, phase or endpoint scoring is imported here.
"""
import numpy as np
from learned_merit import compare_vectors
from control import compare, pair


def module_align(rig, model, branch, merit, budget=20):
    if rig.scope != branch or model.branch != branch or merit.branch != branch:
        raise ValueError('Isolated module and network must match')
    start = rig.measurements
    current = np.zeros(35)
    image = rig.read(current)
    history = []
    sl = slice(0, 15) if branch == 'front' else slice(15, 35)
    gain = 1.
    while rig.measurements-start+4 <= budget and rig.remaining >= 4:
        proposal = model.predict(branch, image)
        applied_gain = gain
        trial = current.copy()
        trial[sl] = np.clip(current[sl]-gain*proposal, -1., 1.)
        if not rig.allowed(trial):
            gain *= .5
            if gain < .0625:
                break
            continue
        before = [rig.read(current), rig.read(current)]
        after = [rig.read(trial), rig.read(trial)]
        if any(x is None for x in before+after):
            raise RuntimeError('Previously allowed module move became invalid')
        result = compare_vectors([merit.vector(x) for x in before],
                                 [merit.vector(x) for x in after])
        accepted = result['accepted']
        if accepted:
            current = trial
            image = after[-1]
            gain = min(1., gain*1.5)
        else:
            image = before[-1]
            gain *= .5
        rig.commit(current)
        history.append(dict(applied_gain=applied_gain, next_gain=gain, accepted=accepted,
                            measurements=rig.measurements, **{k:v for k,v in result.items() if k!='accepted'}))
        if gain < .0625:
            break
    other = slice(15, 35) if branch == 'front' else slice(0, 15)
    assert np.all(current[other] == 0)
    return dict(command=current, history=history, measurements=rig.measurements-start)


def joint_refine(rig, features, budget):
    if rig.scope != 'joint':
        raise ValueError('Joint controller only accepts an assembled rig')
    start = rig.measurements
    current = np.zeros(35)
    history = []
    directions = features.directions
    jac = features.jacobian@directions
    radius = .08
    failures = 0
    # Images are noise-whitened. A physical step prior of radius r gives
    # precision1/r^2, avoiding damping tied to the strongest optical direction.
    while rig.measurements-start+5 <= budget and rig.remaining >= 5:
        reading = rig.read(current)
        if reading is None:
            raise RuntimeError('Current assembled pose invalid')
        residual = features.vector(reading)
        gram = jac.T@jac
        damping = 1./radius**2
        step = np.linalg.solve(gram+damping*np.eye(len(gram)), -jac.T@residual)
        move = directions@step
        move *= min(1., radius/max(abs(move).max(), 1e-12))
        trial = np.clip(current+move, -1., 1.)
        result = compare(rig, current, trial, features)
        if result is None:
            radius *= .5
            if radius < .0005:
                break
            continue
        accepted = result['accepted']
        if accepted:
            dq = np.linalg.lstsq(directions, trial-current, rcond=None)[0]
            dy = result['after']['mean']-result['before']['mean']
            if dq@dq > 1e-10:
                jac += np.outer(dy-jac@dq, dq)/(dq@dq)
            current = trial
            failures = 0
            radius = min(.15, radius*1.25)
        else:
            failures += 1
            radius *= .5
        history.append(dict(stage='joint_proposal', accepted=accepted, measurements=rig.measurements,
                            improvement=result['improvement'], standard_error=result['standard_error'],
                            damping=damping, radius=radius))
        if failures >= 2 and rig.measurements-start+8 <= budget and rig.remaining >= 8:
            selected = np.argsort(abs(jac.T@residual))[-2:]
            refreshed = []
            for axis in selected:
                step_size = max(.015, radius)
                amount = step_size/max(abs(directions[:, axis]).max(), 1e-12)
                delta = directions[:, axis]*amount
                a = current+delta
                b = current-delta
                if not rig.allowed(a) or not rig.allowed(b):
                    continue
                plus = pair(rig, a, features)
                minus = pair(rig, b, features)
                rig.commit(current)
                if plus is None or minus is None:
                    continue
                jac[:, axis] = (plus['mean']-minus['mean'])/(2*amount)
                refreshed.append(int(axis))
            history.append(dict(stage='measured_direction_refresh', directions=refreshed, measurements=rig.measurements))
            failures = 0
            radius = max(radius, .025)
        if radius < .0005:
            break
    return dict(command=current, history=history, measurements=rig.measurements-start)
