"""Feedback paths; decisions use observations, calibration and command interlocks."""
import copy
import json
import time
import numpy as np
from .paths import activate,BASE,DAO,legacy_bridge
from .predictors import Predictor
from .simulation import mod3_camera,execution

def mod3(seed=4101000,profile='latest',budget=80,module_budget=20,device=None,score_grid=64):
    activate('mod3')
    from evaluation import fixture
    from instrument import Instrument
    from control import MeasuredFeatures
    from repaired_control import module_align,joint_refine
    from learned_merit import LearnedMerit
    from scoring import score
    if not 5<=module_budget<=20 or not 2*module_budget+5<=budget<=80:raise ValueError('Use 5..20 per module and total <=80, reserving >=5 for joint feedback')
    camera=mod3_camera(device)
    front,rear=fixture(camera,seed)
    with np.load(BASE/'calibration_probe01/candidate_response.npz') as z:
        features=MeasuredFeatures(z['reference'],z['sigma'],z['basis'],z['corrected_jacobian'],z['directions'])
    with np.load(BASE/'results/calibration.npz') as z:prior=z['high_response']
    stages={}
    for branch,hidden,tag in [('front',front,6100000),('rear',rear,7100000)]:
        model=Predictor('mod3',branch,profile,device)
        merit=LearnedMerit(model,branch,prior)
        instrument=Instrument(camera,hidden,(tag+seed*7)%2**32,module_budget,branch)
        stages[branch]=module_align(instrument,model,branch,merit,module_budget)
    assembled=front+rear+stages['front']['command']+stages['rear']['command']
    if not camera.engine.geometry.path(assembled)['safe']:raise RuntimeError('Assembly rejected by mechanical interlock')
    remaining=budget-sum(stages[b]['measurements'] for b in ('front','rear'))
    joint=Instrument(camera,assembled,(8100000+seed*7)%2**32,remaining,'joint')
    stages['joint']=joint_refine(joint,features,remaining)
    endpoint=assembled+stages['joint']['command']
    # Freeze all commands first; only this posthoc block may score ideal optics.
    terminal=score(camera,endpoint,score_grid)
    return {'system':'mod3','profile':profile,'seed':seed,'stages':stages,
      'acquisitions':sum(v['measurements'] for v in stages.values()),'budget':budget,
      'noise_enabled':True,'ideal_truth_feedback':False,'shwfs_used':False,
      'terminal_offline_only':terminal,'offline_grid':score_grid,
      'scope':'Original repaired noisy-sensorless front/rear/assembled controller. Latest weights have isolated validation evidence, not a previously measured assembled recovery rate.'}

def nikon(seed=4,profile='closed_loop',budget=80,device=None):
    if not 5<=budget<=80:raise ValueError('Nikon L4 requires a 5..80-reading budget')
    activate('nikon');legacy_bridge(device)
    import nikon35
    import evaluate_nikon35_full as control
    model=Predictor('nikon',profile=profile,device=device)
    path=DAO/'rigid_group_retrain_20260908/results/control_nikon/selected_control.json'
    damping=json.loads(path.read_text(encoding='utf-8'))['damping_fraction'] if path.exists() else 1e-4
    hidden,groups=nikon35.sample(seed)
    with execution('nikon'):
        plant=nikon35.Plant(hidden,noise_seed=seed,noise_std=.015,score_center=True)
        q,history=control.controller(plant,model,'L4',damping,budget)
        terminal=plant.terminal(q,fields=(0.,-.05,.05))
    return {'system':'nikon','profile':profile,'seed':seed,'groups':groups,'command':q,
      'history':history,'readings':plant.records,'acquisitions':plant.readings,'budget':budget,
      'noise_enabled':True,'ideal_truth_feedback':False,'shwfs_used':False,
      'terminal_offline_only':terminal,'damping_fraction':damping,
      'scope':'Original Nikon L4 measured noisy node/center residual controller; original input retained.'}

class _TrepanRig:
    """Only noisy sensor images and command interlocks cross this boundary."""
    def __init__(self,runtime,hidden,anchors,seed,budget):
        self.__runtime=runtime;self.__hidden=copy.deepcopy(hidden)
        self.__anchors=tuple(anchors);self.__seed=seed;self.budget=budget;self.measurements=0
        self.records=[]
    @property
    def remaining(self):return self.budget-self.measurements
    def allowed(self,q):
        q=np.asarray(q)
        if q.shape!=(5*len(self.__anchors),) or not np.isfinite(q).all():return False
        # Conservative actuator travel in the legacy physical mm/degree units.
        scales=np.tile([.12,.12,.12,5/60,5/60],len(self.__anchors))
        if np.any(abs(q)>2*scales):return False
        import restore_trepan as t
        state=copy.deepcopy(self.__hidden)
        for i,a in enumerate(self.__anchors):
            state[a]=t.SurfacePerturbation(*(np.asarray(state[a])+q[5*i:5*i+5]))
        try:self.__runtime.input_engines[0].set_surface_perturbations(state,clear_others=True)
        except (ValueError,RuntimeError):return False
        # Reject any geometry clamp rather than quietly observing a different move.
        actual=self.__runtime.input_engines[0].get_surface_perturbations()
        for a in self.__anchors:
            applied=actual.get(a)
            vector=np.zeros(5) if applied is None else np.array([getattr(applied,k) for k in ('dx_mm','dy_mm','dz_mm','tilt_x_deg','tilt_y_deg')])
            if not np.allclose(vector,np.asarray(state[a]),rtol=0,atol=1e-12):return False
        return True
    def read(self,q):
        if self.remaining<=0:raise RuntimeError('Acquisition budget exceeded')
        if not self.allowed(q):return None
        import restore_trepan as t
        state=copy.deepcopy(self.__hidden)
        for i,a in enumerate(self.__anchors):state[a]=t.SurfacePerturbation(*(np.asarray(state[a])+q[5*i:5*i+5]))
        self.measurements+=1
        image=self.__runtime.observe(state,noise_seed=(self.__seed+100003*self.measurements)%2**32)
        self.records.append({'reading':self.measurements,'command':np.asarray(q).tolist()})
        return image

def _compare_vectors(before,after):
    from .sensor_control import compare_vectors
    return compare_vectors(before,after)

def trepan(seed=136,profile='closed_loop',budget=80,module_budget=20,device=None):
    """Portable sensorless feedback using original trained proposal weights.

    The historical Trepan paper controller uses exact center-WRMS gain scores.
    This entry instead accepts moves using reacquired noisy images; it does not
    inherit the historical paper controller's recovery claims.
    """
    if not 5<=module_budget<=20 or not 2*module_budget+8<=budget<=80:raise ValueError('Reserve >=8 joint readings after two 5..20-reading modules; total <=80')
    activate('trepan');legacy_bridge(device)
    import restore_trepan as t
    runtime=t.OpticalRuntime(t.TRAINING_GRID,'speckle')
    initial=t.initial_state_for_seed(seed);assembled=copy.deepcopy(initial);stages={}
    geometry_projections=[]
    def physical_state(requested,label):
        engine=runtime.input_engines[0]
        engine.set_surface_perturbations(requested,clear_others=True)
        applied=engine.get_surface_perturbations()
        result={a:t.SurfacePerturbation(*(getattr(applied[a],k) for k in t.SurfacePerturbation._fields)) if a in applied else t.NOMINAL for a in t.ALL_ANCHORS}
        if any(not np.allclose(result[a],requested[a],rtol=0,atol=1e-12) for a in t.ALL_ANCHORS):geometry_projections.append(label)
        return result
    with execution('trepan'):
        for number,(branch,anchors) in enumerate([('front',t.FRONT_ANCHORS),('rear',t.REAR_ANCHORS)]):
            hidden={a:initial[a] if a in anchors else t.NOMINAL for a in t.ALL_ANCHORS}
            hidden=physical_state(hidden,branch)
            rig=_TrepanRig(runtime,hidden,anchors,seed+number*100000,module_budget)
            model=Predictor('trepan',branch,profile,device)
            scales=np.tile([.12,.12,.12,5/60,5/60],len(anchors))
            q=np.zeros(len(scales));image=rig.read(q);gain=1.;history=[]
            if image is None:raise RuntimeError('Initial physical module state failed the geometry interlock')
            while rig.remaining>=4 and gain>=.0625:
                trial=q-gain*model(image)
                if not rig.allowed(trial):gain*=.5;continue
                before=[rig.read(q),rig.read(q)];after=[rig.read(trial),rig.read(trial)]
                if any(x is None for x in before+after):raise RuntimeError('Invalid reacquired sensor stack')
                result=_compare_vectors([model(x)/scales for x in before],[model(x)/scales for x in after])
                if result['accepted']:q=trial;image=after[-1]
                else:image=before[-1];gain*=.5
                history.append(dict(**result,gain=gain,measurements=rig.measurements))
            for i,a in enumerate(anchors):assembled[a]=t.SurfacePerturbation(*(np.asarray(hidden[a])+q[5*i:5*i+5]))
            stages[branch]={'command':q,'history':history,'measurements':rig.measurements}
        used=sum(v['measurements'] for v in stages.values())
        # Nominal references are measured with independent sensor noise.
        nominal={a:t.NOMINAL for a in t.ALL_ANCHORS}
        reference=np.mean([runtime.observe(nominal,noise_seed=(seed+990000+i)%2**32) for i in range(4)],axis=0)
        def feature(image):
            d=(np.asarray(image,float)-reference)/16384.
            return d.reshape(9,16,8,16,8).mean(axis=(2,4)).ravel()
        assembled=physical_state(assembled,'assembled')
        rig=_TrepanRig(runtime,assembled,t.ALL_ANCHORS,seed+200000,budget-used)
        q=np.zeros(40);radius=.005;rng=np.random.default_rng(seed);history=[]
        scales=np.tile([.12,.12,.12,5/60,5/60],8)
        while rig.remaining>=8:
            direction=rng.choice([-1.,1.],40)*scales
            plus=q+radius*direction;minus=q-radius*direction
            if not rig.allowed(plus) or not rig.allowed(minus):radius*=.5;break
            a=[feature(rig.read(plus)),feature(rig.read(plus))]
            b=[feature(rig.read(minus)),feature(rig.read(minus))]
            slope=(float(a[0]@a[1])-float(b[0]@b[1]))/(2*radius)
            trial=q-radius*np.sign(slope)*direction
            before=[feature(rig.read(q)),feature(rig.read(q))]
            after=[feature(rig.read(trial)),feature(rig.read(trial))]
            result=_compare_vectors(before,after)
            if result['accepted']:q=trial
            else:radius*=.5
            history.append(dict(**result,measurements=rig.measurements,radius=radius))
            if radius<1e-5:break
        endpoint=copy.deepcopy(assembled)
        for i,a in enumerate(t.ALL_ANCHORS):endpoint[a]=t.SurfacePerturbation(*(np.asarray(assembled[a])+q[5*i:5*i+5]))
        stages['joint']={'command':q,'history':history,'measurements':rig.measurements}
        mean,fields=runtime.wide_wrms(endpoint)
    return {'system':'trepan','profile':profile,'seed':seed,'stages':stages,'acquisitions':used+rig.measurements,
      'budget':budget,'offline_nominal_reference_acquisitions':4,'noise_enabled':True,'ideal_truth_feedback':False,'shwfs_used':False,
      'initial_geometry_projection_stages':geometry_projections,
      'terminal_offline_only':{'mean_wrms':mean,'field_wrms':fields},
      'scope':'Noisy sensorless deployment adapter with original lens proposal weights. Different measured acceptance/joint rule from the exact-WRMS historical controller; no original performance claim transferred.'}

def kla(seed=156,profile='closed_loop',budget=80,device=None):
    """Measured original KLA input and calibrated phase refinement, bounded reads."""
    if not 3<=budget<=80:raise ValueError('KLA adapter budget must be 3..80')
    activate('kla')
    from kla_coupled_phase import PhaseControl,SCALE
    from restore_kla import KLA_CASE_DEFINITIONS,make_kla_perturbations,PARAMS
    c=PhaseControl();c.calibration()
    definition=next((d for d in KLA_CASE_DEFINITIONS if d['seed']==seed),None)
    if definition is None:raise ValueError('Use one of the fixed original KLA seeds from doctor output')
    state=make_kla_perturbations(c.engine,definition);used=0;history=[]
    rng=np.random.get_state();np.random.seed(seed)
    try:
        for module in (2,3,4):
            if used+3>budget:break
            measured=[c.engine.get_field_opd_map(f,state,grid_size=64,add_noise=True) for f in c.engine.field_points_y];used+=1
            if any(m is None for m in measured):raise RuntimeError('Missing network pupil')
            model=Predictor('kla',f'module{module}',profile,device)
            predicted=model(np.asarray([m['opd_map'] for m in measured],np.float32))
            before=c.phase(state,averages=1)-c.reference;used+=1
            trial=copy.deepcopy(state)
            for i,lens in enumerate(c.engine.module_lenses[module]):
                trial[lens]={key:float(state[lens][key]-predicted[5*i+j]) for j,key in enumerate(PARAMS)}
            try:after=c.phase(trial,averages=1)-c.reference
            except RuntimeError:after=np.full_like(before,np.inf)
            used+=1;accepted=bool(after@after<before@before)
            if accepted:state=trial
            history.append({'phase':'module','module':module,'accepted':accepted,'acquisitions':used})
        u,s,vt=np.linalg.svd(c.jacobian,full_matrices=False)
        for damping in (.03,.01,.003,.001,.001,.001):
            if used+4>budget:break
            residual=c.phase(state,averages=1)-c.reference;used+=1
            direction=-vt.T@((s/(s*s+(damping*s[0])**2))*(u.T@residual))
            direction=np.clip(direction,-.5,.5)*SCALE;origin=c.vector(state)
            best=(float(residual@residual),0.,state)
            for gain in (1.,.5,.25):
                candidate=c.decode(origin+gain*direction)
                try:v=c.phase(candidate,averages=1)-c.reference;cost=float(v@v)
                except RuntimeError:cost=float('inf')
                used+=1
                if cost<best[0]:best=(cost,gain,candidate)
            state=best[2]
            history.append({'phase':'joint_measured_phase','damping':damping,'gain':best[1],'acquisitions':used})
        terminal=c.terminal(state,seed)
    finally:np.random.set_state(rng)
    return {'system':'kla','profile':profile,'seed':seed,'history':history,'commanded_terminal_state':state,
      'acquisitions':used,'budget':budget,'noise_enabled':True,'ideal_truth_feedback':False,'shwfs_used':False,
      'terminal_offline_only':terminal,
      'scope':'Bounded noisy original phase-map module/joint adapter. Exact historical adaptive phase/MTF polish source is also shipped; this bounded adapter is a separate deployment path.'}
