"""Mod03 Kaleid front-3/rear-4 adapter with measured gains and joint refinement."""
from optics import SamplingOptics
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize
from models import build_universal_model
from distribution import BRANCHES,candidate
from job_support import HERE,CHECKPOINTS,RESULTS,protocol,protocol_sha,atomic_json,sha


class Models:
    def __init__(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');self.models={}
        for branch,bodies in BRANCHES.items():
            path=CHECKPOINTS/f'{branch}_arch_a.pt'
            cp=torch.load(path,map_location='cpu',weights_only=False)
            if cp['protocol_sha256']!=protocol_sha():raise RuntimeError('Checkpoint protocol mismatch')
            model=build_universal_model('arch_a',9,[5]*len(bodies)).to(self.device).eval()
            model.load_state_dict(cp['state_dict']);self.models[branch]=(model,cp)
    def predict(self,branch,speckles):
        model,cp=self.models[branch]
        with torch.inference_mode():
            pred=model(torch.from_numpy(speckles.astype(np.float32)).unsqueeze(0).to(self.device))[0].cpu().numpy()
        return pred*np.asarray(cp['target_std'])+np.asarray(cp['target_mean'])


class CameraPlant:
    def __init__(self,engine,hidden,seed,budget):
        self.engine=engine;self._hidden=np.asarray(hidden,float).copy();self.seed=int(seed)
        self.current_command=np.zeros(35);self.measurements=0;self.requests=0;self.invalid=0;self.budget=budget;self.cache={}
        reference=engine.sample(np.zeros(35),self.seed)
        self.reference=reference['speckles'].astype(np.float32)/65535.
    def allowed(self,command):
        effective=self._hidden+np.asarray(command)
        if max(abs(effective))>1:return False
        return self.engine.geometry.path(effective,start=self._hidden+self.current_command)['safe']
    def trial(self,command):
        self.requests+=1
        if not self.allowed(command):self.invalid+=1;return 1e6,None
        key=tuple(np.asarray(command,float))
        if key in self.cache:return self.cache[key]
        if self.measurements>=self.budget:raise BudgetReached()
        self.measurements+=1
        try:obs=self.engine.sample(self._hidden+np.asarray(command),self.seed)
        except (ValueError,AssertionError):self.invalid+=1;return 1e6,None
        score=float(np.mean((obs['speckles'].astype(np.float32)/65535.-self.reference)**2))
        self.cache[key]=(score,obs)
        return score,obs
    def commit(self,command):
        if not self.allowed(command):raise RuntimeError('Unsafe selected command')
        self.current_command=np.asarray(command,float).copy()


class BudgetReached(RuntimeError):pass


def run_case(engine,models,index):
    cfg=protocol()['control'];path=RESULTS/f'kaleid_case_{index}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    # Disjoint from the 0..4999 training/validation records. Force both sections
    # to be present, with independent scales; the controller never sees this q.
    for attempt in range(2000):
        front,meta=candidate('front',index,attempt);rear,_=candidate('rear',index+197,attempt)
        hidden=np.r_[front[:15],rear[15:]]
        if not engine.geometry.path(hidden)['safe']:continue
        try:engine.sample(hidden,93003+index)
        except (ValueError,AssertionError):continue
        break
    else:raise RuntimeError('No valid held-out integration fixture')
    plant=CameraPlant(engine,hidden,93003+index,cfg['measurement_budget']);start=time.perf_counter()
    command=np.zeros(35);initial_score,initial_obs=plant.trial(command);score=initial_score;obs=initial_obs;history=[]
    for branch,bodies in BRANCHES.items():
        columns=np.array([5*g+d for g in bodies for d in range(5)])
        for round_index,target in enumerate(cfg['gain_rounds']):
            prediction=models.predict(branch,obs['speckles'])
            best=(score,command.copy(),obs,0.)
            for gain in [target+x for x in cfg['gain_offsets']]:
                trial=command.copy();trial[columns]=np.clip(trial[columns]-gain*prediction,-1,1)
                value,reading=plant.trial(trial)
                if value<best[0]:best=(value,trial,reading,gain)
            score,command,obs,gain=best;plant.commit(command)
            history.append(dict(stage='module_gain',branch=branch,round=round_index+1,selected_gain=gain,camera_merit=score,
                                measurements=plant.measurements))
    acquired=command.copy();acquired_score=score;best_joint=[score,command.copy(),obs]
    def objective(trial):
        value,reading=plant.trial(trial)
        if value<best_joint[0]:best_joint[:]=[value,np.asarray(trial).copy(),reading]
        return value
    optimizer_info={}
    try:
        fit=minimize(objective,command,method='L-BFGS-B',bounds=[(-1,1)]*35,
                     options={'maxiter':cfg['max_joint_iterations'],'eps':1e-3,'ftol':1e-8,'maxls':5})
        optimizer_info=dict(nit=int(fit.nit),nfev=int(fit.nfev),message=str(fit.message))
    except BudgetReached:
        optimizer_info=dict(message='Registered camera-measurement budget reached; best measured feasible command retained')
    score,command,obs=best_joint;plant.commit(command)
    result=dict(status='complete',case_id=index,protocol_sha256=protocol_sha(),
        algorithm='Kaleid front3/rear4 proposals + measured gain selection + bounded joint refinement',
        input='legacy noisy speckle camera only',feedback='camera residual to nominal calibration',
        noise_policy='same camera RNG seed across calibration and candidate acquisitions in this simulator fixture',
        hidden_state_read_by_controller=False,measurement_budget=cfg['measurement_budget'],measurements=plant.measurements,
        requests=plant.requests,invalid_rejected=plant.invalid,initial_camera_merit=initial_score,
        acquired_camera_merit=acquired_score,terminal_camera_merit=score,history=history,optimizer=optimizer_info,
        initial_wrms_waves=initial_obs['wrms_waves'].tolist(),terminal_wrms_waves=obs['wrms_waves'].tolist(),
        initial_survival=initial_obs['ray_survival'].tolist(),terminal_survival=obs['ray_survival'].tolist(),
        hidden_fixture_for_posthoc_audit=hidden.tolist(),acquired_command=acquired.tolist(),terminal_command=command.tolist(),
        elapsed_seconds=time.perf_counter()-start,
        scope='two held-out interface/controller fixtures; not manufacturing-yield or full robustness benchmark')
    atomic_json(path,result);print(f'KALEID case={index} camera_merit {initial_score:.6g}->{score:.6g} measurements={plant.measurements}',flush=True)
    return result


def integrate():
    models=Models();engine=SamplingOptics();records=[]
    for index in protocol()['control']['case_ids']:records.append(run_case(engine,models,index))
    manifest=dict(state='ready',version=protocol()['version'],protocol_sha256=protocol_sha(),
        front=dict(checkpoint=str(CHECKPOINTS/'front_arch_a.pt'),group_dofs=[5,5,5]),
        rear=dict(checkpoint=str(CHECKPOINTS/'rear_arch_a.pt'),group_dofs=[5,5,5,5]),
        source_group_order=BRANCHES,observation='9 raw noisy speckle frames, float32 ADU',
        checkpoints_sha256={b:sha(CHECKPOINTS/f'{b}_arch_a.pt') for b in BRANCHES},
        controller_entry=str(HERE/'control.py'),old_system_models_replaced=False,
        trained_and_integrated=True,full_recovery_performance_validated=False,
        completed_integration_cases=[r['case_id'] for r in records])
    atomic_json(HERE/'active_manifest.json',manifest)
    return manifest
