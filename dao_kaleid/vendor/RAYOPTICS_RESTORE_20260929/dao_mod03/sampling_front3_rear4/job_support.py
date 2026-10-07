from pathlib import Path
import hashlib
import json
import os
import time

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
CHECKPOINTS=HERE/'checkpoints'
RESULTS=HERE/'results'


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def protocol():
    return json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))


def protocol_sha():return sha(HERE/'protocol.json')


def status(stage,**values):
    path=HERE/'state.json'
    old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    old.update(values);old.update(stage=stage,updated_unix=time.time())
    atomic_json(path,old)


def freeze_protocol():
    from optics import SamplingOptics, ARCHIVE, DAO
    from distribution import BRANCHES, BIN_EDGES, FAMILIES
    from mechanics import SCALES
    path=HERE/'protocol.json'
    sources={name:sha(HERE/name) for name in ('mechanics.py','distribution.py','optics.py')}
    if path.exists():
        value=protocol()
        if value['sampling_source_hashes']!=sources:raise RuntimeError('Sampling sources changed after protocol freeze')
        return value
    e=SamplingOptics()
    value=dict(version='mod03_front3_rear4_100um_5arcmin_v1',
        design_sha256=sha(ARCHIVE/'US07199938-1-mod03.zmx'),
        branches={k:list(v) for k,v in BRANCHES.items()},
        group_numbering='Source ZMX order, G1=S3..5 through G7=S21..22; front means G1..G3 in this registration',
        samples_per_branch=5000,batch_size=25,train_count_per_branch=4000,validation_count_per_branch=1000,
        sampler_workers=2,maximum_attempts_per_record=2000,
        normalized_axis_scales_mm_mm_mm_deg_deg=SCALES.tolist(),
        component_bounds='each dx/dy/dz +/-100 um; each tx/ty +/-5 arcmin',
        cemented_internal_surfaces_rigid=True,collision=e.geometry.describe(),
        amplitude_bins_normalized=BIN_EDGES,accepted_quota_per_bin_per_branch=1000,
        families=FAMILIES,body_scale_rule='independent translation and tilt scales per body, independent signs and axis amplitudes',
        context='60% target branch only; 40% with independently drawn non-target bodies; no oracle isolation during control',
        fields_xy_deg=e.fields.tolist(),wavelength_nm=e.wave,fixed_focus_mm=e.focus_mm,
        observation='legacy Fresnel diffuser and minimal sCMOS speckle camera; 9 raw ADU images',
        pupil_grid=64,camera_grid=128,diffuser_sha256=sha(DAO/'artifacts/diffuser_mask.npy'),
        legacy_camera_source_sha256=sha(DAO/'generate_speckle_dataset.py'),
        legacy_fast_kernel_sha256=sha(DAO/'rigid_group_retrain_20260908/fast_raytrace.py'),
        endpoint_optimization='same original trace_batch numerical body, final per-ray RayPkg packing omitted',
        dtype_storage='uint16 lossless rounded camera ADU; training casts raw values to float32',
        invalid_optics_policy='reject failed reference-sphere intersection, missing chief, or field pupil survival below 20%; retain rejection counts and retry same registered stratum',
        training=dict(architecture='arch_a',epochs=35,batch_size=32,learning_rate=2e-4,weight_decay=1e-4,
                      loss='SmoothL1 beta .25 on standardized target coordinates',input_transform='raw_float32',
                      module_outputs=dict(front=15,rear=20),checkpoint_selection='minimum validation loss'),
        control=dict(algorithm='Kaleid modular neural proposal, measured scalar gain acquisition, bounded joint refinement',
                     gain_rounds=[1.,.7,.3],gain_offsets=[-.2,-.1,0.,.1,.2],
                     joint_solver='L-BFGS-B',max_joint_iterations=30,measurement_budget=80,
                     feedback_merit='nine-field mean squared camera residual to a fixed nominal speckle calibration, in ADU/65535',
                     integration_noise_policy='common camera RNG seed within each simulator fixture, independent across fixtures',
                     hidden_state_used_for_prediction_or_selection=False,
                     simulator_motion_interlock='whole path from last accepted state to trial state; rejected commands never evaluated or applied',
                     integration_cases=2,case_ids=[50000,50001]),
        sampling_source_hashes=sources,
        limitations=['body shoulders outside optical caps modeled as flat annuli at prescribed mechanical radius; no holder CAD',
                     'scalar diffraction and original camera sampling retained; this does not establish high-NA vector accuracy or broad-range recovery performance'])
    atomic_json(path,value)
    return value
