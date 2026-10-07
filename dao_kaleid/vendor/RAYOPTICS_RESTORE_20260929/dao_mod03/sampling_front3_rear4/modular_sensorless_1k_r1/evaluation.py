from settings import *
from model import build
from instrument import Instrument,load_features,measured_calibration
from control import module_align,joint_refine
from scoring import score


class Networks:
    def __init__(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');self.networks={}
        for branch in BRANCHES:
            cp=torch.load(CHECKPOINTS/f'{branch}_arch_a.pt',map_location='cpu',weights_only=False)
            assert cp['protocol_sha256']==protocol_sha()
            model=build(branch).to(self.device).eval();model.load_state_dict(cp['state_dict'])
            self.networks[branch]=(model,cp)

    def predict(self,branch,image):
        model,cp=self.networks[branch]
        x=torch.as_tensor(image.astype(np.float32),device=self.device)
        x=torch.asinh((x-cp['reference'].to(self.device))/cp['noise_std'].to(self.device)/3.)
        with torch.inference_mode():out=model(x.unsqueeze(0))[0].cpu().numpy()
        return out*cp['target_std']+cp['target_mean']


def fixture(camera,index):
    for attempt in range(2000):
        front,_=isolated_candidate('front',index,attempt)
        rear,_=isolated_candidate('rear',index+197,attempt)
        if not all(camera.engine.geometry.path(q)['safe'] for q in [front,rear,front+rear]):continue
        try:
            for q in [front,rear,front+rear]:
                if min(camera.engine.observe(q)['ray_survival_fraction'])<.2:raise ValueError('Insufficient pupil support')
        except ValueError:continue
        return front,rear
    raise RuntimeError('No safe modular fixture')


def run_case(camera,models,features,index):
    folder=RESULTS/'cases';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{index}.json'
    if path.exists():return json.loads(path.read_text())
    started=active_seconds();front,rear=fixture(camera,index)
    front_rig=Instrument(camera,front,6100000+index*7,20,'front')
    rear_rig=Instrument(camera,rear,7100000+index*7,20,'rear')
    f=module_align(front_rig,models,'front',features,20)
    r=module_align(rear_rig,models,'rear',features,20)
    assembled=front+f['command']+rear+r['command']
    if not camera.engine.geometry.path(assembled)['safe']:
        raise RuntimeError('Assembled corrected modules violate collision interlock; no unsafe joint refinement')
    budget=80-f['measurements']-r['measurements']
    joint_rig=Instrument(camera,assembled,8100000+index*7,budget,'joint')
    joint=joint_refine(joint_rig,features,budget)
    endpoint=assembled+joint['command']
    # All commands are frozen above. Only here may optical truth be scored.
    metrics={name:score(camera,q,256) for name,q in
             [('initial_assembly',front+rear),('after_module_assembly',assembled),('terminal',endpoint)]}
    coarse=score(camera,endpoint,128)['piston_tilt_defocus']['per_field_rms_waves']
    pt=metrics['terminal']['piston_tilt_defocus'];mean=pt['mean_field_rms_waves'];center=pt['per_field_rms_waves'][0]
    count=f['measurements']+r['measurements']+joint['measurements'];assert count<=80
    row=dict(case_id=index,protocol_sha256=protocol_sha(),front_hidden_posthoc_only=front.tolist(),
             rear_hidden_posthoc_only=rear.tolist(),front_command=f['command'].tolist(),rear_command=r['command'].tolist(),
             joint_command=joint['command'].tolist(),front_history=f['history'],rear_history=r['history'],joint_history=joint['history'],
             exposures=dict(front=f['measurements'],rear=r['measurements'],joint=joint['measurements'],total=count),
             isolated_training_and_deployment=True,other_branch_perturbed_during_module_alignment=False,
             noisy_sensorless=True,shwfs_used=False,ideal_phase_feedback=False,optical_metrics=metrics,
             offline_pupil_grid=256,terminal_128_grid_waves=coarse,
             terminal_128_to_256_max_abs_waves=float(np.max(abs(np.asarray(coarse)-pt['per_field_rms_waves']))),
             terminal_mean_waves=mean,terminal_center_waves=center,
             pass_mean_and_center=bool(mean<.08 and center<.08 and metrics['terminal']['minimum_pupil_support_fraction']>=.2),seconds=active_seconds()-started)
    atomic_json(path,row);print('MODULAR_CASE',index,'mean',mean,'center',center,'pass',row['pass_mean_and_center'],'exposures',count,flush=True)
    return row


def run(camera):
    models=Networks();features=measured_calibration(camera);rows=[]
    for index in protocol()['evaluation']['case_ids']:
        status('evaluation',case_id=index,completed_cases=len(rows),target_cases=30)
        rows.append(run_case(camera,models,features,index))
    summary=dict(case_count=len(rows),pass_count=sum(x['pass_mean_and_center'] for x in rows),
        mean_of_case_means=float(np.mean([x['terminal_mean_waves'] for x in rows])),
        mean_center=float(np.mean([x['terminal_center_waves'] for x in rows])),
        completed=True,assistant_reviewed=False,old_models_replaced=False,
        per_case=[{k:x[k] for k in ['case_id','terminal_mean_waves','terminal_center_waves','pass_mean_and_center','exposures']} for x in rows])
    atomic_json(RESULTS/'evaluation_summary.json',summary)
    return summary
