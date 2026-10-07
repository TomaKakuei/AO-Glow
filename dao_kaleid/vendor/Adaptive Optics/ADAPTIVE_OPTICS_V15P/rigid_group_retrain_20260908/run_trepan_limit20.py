"""Trepan: paired proposals, bounded assembly gain search, then solver under 20 readings.

The original front-first replay and its source are retained in results/trepan_limit20.
Only online center scores enter control; nine-field audits remain evaluation-only.
"""
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from common import HERE, ROOT, ACTIVE, atomic_json, sha256
from fast_execution import execution

sys.path.insert(0, str(ROOT / 'kaleid_modular_repair_20260906'))
import restore_trepan as base
OUT = HERE / 'results/trepan_balanced20'
SEEDS = (3,34,36,96,100,142,136,188,187,246,272)
RUNTIME = None
MODELS = {}


class BudgetReached(Exception):
    pass


def run(job):
    global RUNTIME
    method, variant, seed = job
    path = OUT / f'{method}_{variant}_{seed}.json'
    identity = sha256(__file__)
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        assert cached['script_sha256'] == identity and cached['manifest_sha256'] == sha256(ACTIVE)
        return cached
    if RUNTIME is None:
        RUNTIME = base.OpticalRuntime(base.TRAINING_GRID, 'speckle')
    runtime = RUNTIME
    start = time.perf_counter()
    initial = base.initial_state_for_seed(seed)
    current_state = copy.deepcopy(initial)
    records = []
    forwards = {'front':0, 'rear':0}
    phases = []
    stop_reason = 'scheduled_endpoint'
    interrupted_phase = None
    best_observed = {'score': float('inf'), 'state': copy.deepcopy(initial), 'reading': None}

    def acquire(kind, state, **meta):
        if len(records) >= 20:
            raise BudgetReached()
        record = dict(reading=len(records)+1, kind=kind, state=base.state_json(state),
                      elapsed_seconds=time.perf_counter()-start, **meta)
        if kind == 'proposal':
            value = runtime.observe(state, noise_seed=meta['noise_seed'])
        else:
            value = float(runtime.center_wrms(state))
            record['center_wrms'] = value
            record['score_scope'] = 'actual_full_assembly'
            if np.isfinite(value) and value < best_observed['score']:
                best_observed.update(score=value, state=copy.deepcopy(state), reading=record['reading'])
        records.append(record)
        return value

    def predict(group, index, round_index, state):
        key = (method, group)
        if key not in MODELS:
            MODELS[key] = base.load(group, method)
        net, checkpoint, source = MODELS[key]
        obs = acquire('proposal', state, group=group, round=round_index+1,
                      noise_seed=seed*1000+(index*3+round_index+1)*100)
        with torch.inference_mode():
            out = net(torch.from_numpy(obs).unsqueeze(0).float().to(base.DEVICE))[0].cpu().numpy()
        forwards[group] += 1
        return out*np.asarray(checkpoint['target_std'])+np.asarray(checkpoint['target_mean'])

    with execution('trepan'):
        initial_mean, initial_fields = runtime.wide_wrms(initial)
        acquire('initial_diagnosis', initial, phase='initial')
        if variant in ('L1', 'L2'):
            for index, (group, anchors) in enumerate((('front',base.FRONT_ANCHORS),('rear',base.REAR_ANCHORS))):
                isolated = {a:initial[a] if a in anchors else base.NOMINAL for a in base.ALL_ANCHORS}
                prediction = predict(group,index,0,isolated)
                candidate = base.vec_to_state_group(prediction*(.7 if variant=='L2' else 1.),initial,anchors)
                for a in anchors:current_state[a]=candidate[a]
            acquire('applied_proposal', current_state, phase=variant)
            phases.append('single_grouped_proposal')
        else:
            try:
                if method != 'no_nn':
                    predictions = {}
                    # Complete both branches before spending any readings on gain search.
                    for index, (group, anchors) in enumerate((('front',base.FRONT_ANCHORS),('rear',base.REAR_ANCHORS))):
                        isolated = {a:initial[a] if a in anchors else base.NOMINAL for a in base.ALL_ANCHORS}
                        interrupted_phase = f'{group}_initial_proposal'
                        predictions[group] = predict(group,index,0,isolated)
                        candidate = base.vec_to_state_group(predictions[group],initial,anchors)
                        for a in anchors:current_state[a]=candidate[a]
                    joint_score = acquire('applied_proposal',current_state,phase='paired_unit_proposal')
                    phases.append('paired_unit_proposal')
                    # At most five new scores per branch leaves >=6 readings for
                    # cyclic/joint refinement. Both searches score the real assembly.
                    for group, anchors in (('front',base.FRONT_ANCHORS),('rear',base.REAR_ANCHORS)):
                        interrupted_phase = f'{group}_assembly_gain'
                        origin = copy.deepcopy(current_state)
                        samples = {1.:joint_score}
                        candidates = {1.:origin}
                        def objective_gain(gain):
                            candidate = copy.deepcopy(origin)
                            module = base.vec_to_state_group(gain*predictions[group],initial,anchors)
                            for a in anchors:candidate[a]=module[a]
                            score = acquire('candidate_score',candidate,phase='assembly_gain',group=group,gain=float(gain))
                            samples[float(gain)] = score
                            candidates[float(gain)] = candidate
                        for gain in (0.,.5,1.5,2.):objective_gain(gain)
                        # A local quadratic in WRMS squared proposes one measured
                        # interpolation. It cannot accept an unmeasured optimum.
                        grid = sorted(samples)
                        index = int(np.argmin([samples[g] for g in grid]))
                        left = min(max(index-1,0),len(grid)-3)
                        nearby = grid[left:left+3]
                        values = np.square([samples[g] for g in nearby])
                        if np.isfinite(values).all():
                            quadratic,linear,_ = np.polyfit(nearby,values,2)
                            if quadratic > 0:
                                vertex = float(np.clip(-linear/(2*quadratic),nearby[0],nearby[-1]))
                                if all(abs(vertex-g)>1e-8 for g in samples):objective_gain(vertex)
                        gain = min(samples,key=samples.get)
                        current_state = candidates[gain]
                        joint_score = samples[gain]
                        phases.append(dict(group=group,gain=float(gain),complete=True,
                                           new_score_calls=len(samples)-1,score_scope='actual_full_assembly'))
                if variant == 'gain_only':
                    interrupted_phase = None
                elif variant == 'L3':
                    interrupted_phase='fixed_cyclic'
                    best=acquire('candidate_score',current_state,phase='fixed_cyclic_initial')
                    for anchor in base.ALL_ANCHORS:
                        for axis in range(5):
                            for sign in (1.,-1.):
                                candidate=current_state.copy();v=np.asarray(candidate[anchor]).copy();v[axis]+=sign*.005
                                candidate[anchor]=base.SurfacePerturbation(*v)
                                score=acquire('candidate_score',candidate,phase='fixed_cyclic',anchor=int(anchor),axis=axis,sign=sign)
                                if score<best:current_state=candidate;best=score
                else:
                    interrupted_phase='joint_40'
                    x0=np.concatenate([np.asarray(initial[a])-np.asarray(current_state[a]) for a in base.ALL_ANCHORS])
                    def decode(x):return {a:base.SurfacePerturbation(*(np.asarray(initial[a])-x[i*5:(i+1)*5])) for i,a in enumerate(base.ALL_ANCHORS)}
                    best={'score':float('inf'),'state':current_state}
                    def objective_joint(x):
                        state=decode(x);score=acquire('candidate_score',state,phase='joint_40')
                        if score<best['score']:best.update(score=score,state=state)
                        return score
                    try:
                        fit=minimize(objective_joint,x0,method='L-BFGS-B',options={'maxiter':30,'ftol':1e-4,'eps':1e-2})
                        current_state=decode(fit.x);stop_reason=str(fit.message)
                    except BudgetReached:
                        current_state=best['state'];raise
            except BudgetReached:
                stop_reason='20 online acquisitions exhausted; best measured full-assembly state retained'
            current_state = copy.deepcopy(best_observed['state'])
        final,fields=runtime.wide_wrms(current_state)
    assert len(records)<=20
    if method=='slor':assert variant=='L1' and forwards=={'front':1,'rear':1}
    result=dict(system='Trepan2p',seed=seed,proposal=method,variant=variant,initial_metric=float(initial_mean),
                initial_field_wrms=list(map(float,initial_fields)),terminal_metric=float(final),field_wrms=list(map(float,fields)),
                success=int(final<.07),moves_used=len(records),budget=20,
                proposal_observations=sum(r['kind']=='proposal' for r in records),
                score_calls=sum(r['kind']=='candidate_score' for r in records),
                network_forwards=forwards if method!='sensitivity_svd' else {'front':0,'rear':0},
                predictor_calls=forwards,records=records,phases=phases,stop_reason=stop_reason,
                stopped_phase=interrupted_phase,terminal_state=base.state_json(current_state),
                selected_reading=best_observed['reading'] if variant not in ('L1','L2') else len(records),
                schedule='paired_proposals_then_assembly_gain_then_existing_solver',
                initial_state=base.state_json(initial),terminal_audit_fields=9,terminal_audit_count=1,
                audit_excluded_from_online_budget=True,script_sha256=identity,manifest_sha256=sha256(ACTIVE),
                elapsed_seconds=time.perf_counter()-start)
    atomic_json(path,result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[('arch_a',v) for v in ('L1','L2','gain_only','L3','L4')]+[(m,'L4') for m in ('arch_b','arch_c','sensitivity_svd','no_nn')]
    atomic_json(OUT/'protocol.json',dict(budget=20,unit='online acquisition including initial diagnosis, each branch observation and each candidate score',
                excluded='noise-free initial diagnostic audit and frozen terminal nine-field audit',
                schedule='both initial branch proposals, full-assembly unit score by reading 4, at most 5 additional gain scores per branch, then existing joint or cyclic solver',
                gain_grid=[0.,.5,1.,1.5,2.],gain_interpolation='one bounded local quadratic vertex per branch, fitted to squared center WRMS and measured before selection',
                exhaustion='best measured full-assembly state retained; initial and paired unit state are eligible; terminal audit never enters control',
                observation_scope='unchanged module-conditioned observations with the other branch nominal',
                feedback_scope='actual full assembly center WRMS; nine-field audit excluded from control',
                slor='excluded from this repair and rerun',seeds=SEEDS,entries=entries,
                source='user confirmed same counting convention as Nikon readings',script_sha256=sha256(__file__),manifest_sha256=sha256(ACTIVE)))
    results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(run,(m,v,s)) for m,v in entries for s in SEEDS]
        for f in as_completed(fs):
            row=f.result();results.append(row)
            print('LIMIT20',len(results),len(fs),row['proposal'],row['variant'],row['seed'],row['success'],row['moves_used'],row['stopped_phase'],flush=True)
            atomic_json(OUT/'progress.json',dict(completed=len(results),total=len(fs)))
    summaries=[]
    for m,v in entries:
        subset=[r for r in results if r['proposal']==m and r['variant']==v]
        summaries.append(dict(proposal=m,variant=v,cases=len(subset),successes=sum(r['success'] for r in subset),mean_terminal_metric=float(np.mean([r['terminal_metric'] for r in subset]))))
    atomic_json(OUT/'summary.json',dict(status='complete',results=summaries,rows=results,budget=20))


if __name__=='__main__':main()
