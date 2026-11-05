"""Immutable-runtime checkpoint audit: never import candidate Python modules.

Only fixed-shape numeric NPZ arrays are accepted. Pickle/PT deserialization is
not used at this trust boundary, including torch.load(weights_only=True).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import zipfile
import numpy as np
import torch
from torch import nn


def audit(folder, config, coordinates, wall_seconds):
    folder=Path(folder)
    cfg=json.loads(Path(config).read_text())
    n,p,tr=cfg['network'],cfg['physics'],cfg['training']
    assert n=={'hidden_layers':5,'width':64,'activation':'SiLU'}, 'Unexpected published architecture'
    weights=folder/'model_final.npz'
    assert weights.stat().st_size<10_000_000, 'Checkpoint too large'
    with zipfile.ZipFile(weights) as z:
        entries=z.infolist()
        assert len(entries)<=40 and len({i.filename for i in entries})==len(entries)
        assert sum(i.file_size for i in entries)<10_000_000
        assert all(i.filename.endswith('.npy') and '/' not in i.filename for i in entries)
    layers=[]
    for index in range(5):
        layers += [nn.Linear(2 if index==0 else 64,64),nn.SiLU()]
    layers += [nn.Linear(64,1)]
    network=nn.Sequential(*layers).eval()
    expected=network.state_dict()
    with np.load(weights,allow_pickle=False) as archive:
        keys=set(archive.files)
        assert keys=={domain+'.net.'+key for domain in ('cu','ox') for key in expected}, 'Checkpoint keys differ from public architecture'
        state={}
        for key,template in expected.items():
            cu,ox=archive['cu.net.'+key],archive['ox.net.'+key]
            assert cu.dtype==np.float32 and cu.shape==tuple(template.shape)
            assert np.isfinite(cu).all() and np.array_equal(cu,ox), 'Shared-network weights inconsistent'
            state[key]=torch.from_numpy(cu.copy())
    network.load_state_dict(state,strict=True)
    # A newly trained network must differ from its seeded initialization.
    torch.manual_seed(42)
    initial=[]
    for index in range(5): initial += [nn.Linear(2 if index==0 else 64,64),nn.SiLU()]
    initial += [nn.Linear(64,1)]
    initial=nn.Sequential(*initial)
    for layer in initial:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)
    assert any(not torch.equal(v,initial.state_dict()[k]) for k,v in state.items()), 'Untrained initialization submitted'
    data=np.loadtxt(coordinates,delimiter=',',skiprows=1,ndmin=2)
    prediction=[]
    torch.set_num_threads(2)
    with torch.no_grad():
        for begin in range(0,len(data),8192):
            block=torch.tensor(data[begin:begin+8192],dtype=torch.float32)
            x=torch.stack((block[:,0]/p['R_out'],block[:,1]/p['H']),dim=1)
            prediction.extend((network(x).flatten()*((p['T_top']-p['T_bottom'])/2)+(p['T_top']+p['T_bottom'])/2).tolist())
    with (folder/'prediction.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['r_um','z_um','temperature_K']
        rows=list(reader)
    assert len(rows)==len(data)==len(prediction)
    discrepancy=[]
    for row,point,t in zip(rows,data,prediction):
        r,z,reported=map(float,(row['r_um'],row['z_um'],row['temperature_K']))
        assert all(math.isfinite(v) for v in (r,z,reported,t))
        assert abs(r-point[0])<=1e-10 and abs(z-point[1])<=1e-10
        discrepancy.append(abs(reported-t))
    assert max(discrepancy)<2e-3, f'Checkpoint/CSV mismatch on full coordinate set: {max(discrepancy)} K'
    receipt=json.loads((folder/'run.json').read_text())
    assert type(receipt.get('seed')) is int and receipt['seed']==42, 'run.json seed mismatch'
    assert type(receipt.get('steps')) is int and receipt['steps']==tr['adam_steps']+tr['lbfgs_steps'], 'run.json step count mismatch'
    assert type(receipt.get('loss_evaluations')) is int and receipt['loss_evaluations']>=receipt['steps'], 'Invalid loss evaluation count'
    assert receipt.get('device')=='cpu', 'run.json device mismatch'
    for key in ('training_seconds','total_seconds','final_loss'):
        value=receipt.get(key)
        assert type(value) in (int,float) and math.isfinite(value) and value>=0, 'Invalid run.json '+key
    assert 0<receipt['training_seconds']<=receipt['total_seconds']<=wall_seconds+1, 'run.json timings exceed trusted wall clock'
    history=json.loads((folder/'training_history.json').read_text())
    assert len(history)==receipt['steps'], 'Training history length mismatch'
    for i,row in enumerate(history):
        assert row['iteration']==i+1
        stage='Adam' if i<tr['adam_steps'] else 'L-BFGS'
        assert row['stage']==stage
        assert row['stage_step']==(i+1 if stage=='Adam' else i+1-tr['adam_steps'])
        assert type(row['loss_evaluations']) is int and row['loss_evaluations']>=i+1
        if i: assert row['loss_evaluations']>history[i-1]['loss_evaluations']
        for key,value in row.items():
            if key not in ('stage',):
                assert type(value) in (int,float) and math.isfinite(value), 'Invalid training history value'
        assert row['objective']>=0
    assert history[-1]['loss_evaluations']==receipt['loss_evaluations']
    assert math.isclose(history[-1]['objective'],receipt['final_loss'],rel_tol=1e-7,abs_tol=1e-12), 'run.json final loss mismatch'
    return {'checked_coordinates':len(data),'checkpoint_max_discrepancy_K':max(discrepancy),
            'run_json_validated':True,'training_history_rows':len(history),'numeric_npz_only':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--points',required=True)
    parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.output,args.config,args.points,args.wall_seconds),allow_nan=False))


if __name__=='__main__': main()
