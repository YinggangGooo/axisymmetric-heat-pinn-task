"""Trusted verifier; candidate code always runs as the unprivileged agent.

Only the standard library is imported in the root verifier. Numerical values
are independently recomputed here, never trusted from candidate error reports.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import sys
sys.path.insert(0,str(Path(__file__).parent))
import runtime_guard as guard

LOG = Path('/logs/verifier')
HERE = Path(__file__).parent
WEIGHTS = {'environment': .10, 'cylindrical_operator': .25, 'top_boundary': .25,
           'pointwise_error': .20, 'fresh_training_fem': .20}
# Environment is an enabling skill (10%). The scientific PDE and boundary
# definitions are the main domain knowledge (25% each). Honest error evaluation
# and independently verified end-to-end predictions each carry 20%.
PREAMBLE = '''import sys, json
sys.path.insert(0, '/app')
sys.path.append('/app/.deps')
'''


def candidate(code, timeout=90):
    run = guard.execute(code=PREAMBLE + code,timeout=timeout)
    if run.returncode:
        raise AssertionError(run.stderr[-3000:] or run.stdout[-3000:] or 'candidate exited unsuccessfully')
    lines = [line[len('RESULT:'):] for line in run.stdout.splitlines() if line.startswith('RESULT:')]
    if len(lines) != 1:
        raise AssertionError('Expected exactly one numerical probe result')
    return json.loads(lines[0])


def environment():
    return candidate('''
import numpy, scipy, torch
from scipy.interpolate import CloughTocher2DInterpolator
from error_metrics import interpolate_reference
import numpy as np
points = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
temperature = 300 + 2*points[:,0] + 3*points[:,1]
v = interpolate_reference(points, temperature, np.array([[.23,.31],[.62,.11]]))
assert np.allclose(v, [301.39,301.57], atol=2e-5)
print('RESULT:'+json.dumps({'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__}))
''')


def cylindrical_operator():
    got = candidate('''
import torch
from baseline_models import pde_residual
torch.set_default_dtype(torch.float64)
class Field(torch.nn.Module):
    def __init__(self, a,b,c):
        super().__init__(); self.a,self.b,self.c=a,b,c
    def forward(self,x):
        r,z=x[:,:1],x[:,1:2]
        return 300+self.a*r**4+self.b*r**2*z**2+self.c*z**2+.7*z
outputs=[]
for a,b,c in [(1.2,.3,.8),(-.4,.17,2.1),(.015,-.22,-.7)]:
    r=torch.tensor([[.07],[.41],[1.37],[2.82]],requires_grad=True)
    z=torch.tensor([[.2],[2.7],[11.3],[23.8]],requires_grad=True)
    outputs.append(pde_residual(Field(a,b,c),r,z,1e-4).detach().reshape(-1).tolist())
print('RESULT:'+json.dumps(outputs))
''')
    if len(got) != 3:
        raise AssertionError('Wrong number of operator probe fields')
    for values, (a,b,c) in zip(got, [(1.2,.3,.8),(-.4,.17,2.1),(.015,-.22,-.7)]):
        if len(values) != 4:
            raise AssertionError('PDE residual must preserve one result per point')
        for value,r,z in zip(values,[.07,.41,1.37,2.82],[.2,2.7,11.3,23.8]):
            expected=16*a*r*r+4*b*z*z+2*b*r*r+2*c
            if not math.isfinite(value) or not math.isclose(value,expected,rel_tol=2e-5,abs_tol=2e-6):
                raise AssertionError(f'Cylindrical differential operator mismatch: {value} vs {expected}')
    return {'polynomial_fields':3,'points_per_field':4}


def top_boundary():
    values = candidate('''
import torch
from baseline_models import top_residual
torch.set_default_dtype(torch.float64)
class Field(torch.nn.Module):
    def __init__(self,offset,slope):
        super().__init__(); self.offset,self.slope=offset,slope
    def forward(self,x):
        return 380+self.offset+self.slope*(x[:,1:2]-25)+0*x[:,:1]
out=[]
for offset,slope in [(0,-3.2),(0,7),(0,.23),(-12,-3.2),(3.5,7),(-1,.23)]:
    r=torch.tensor([[0.],[1.1],[2.5],[3.]],requires_grad=True)
    z=torch.full_like(r,25.,requires_grad=True)
    p={'T_top':380.,'T_bottom':300.,'H':25.}
    result=top_residual(Field(offset,slope),r,z,p)
    assert tuple(result.shape)==(4,1)
    out.append(result.detach().reshape(-1).tolist())
print('RESULT:'+json.dumps(out))
''')
    if len(values)!=6 or any(len(row)!=4 for row in values):
        raise AssertionError('Wrong boundary residual shape')
    if not all(math.isfinite(value) for row in values for value in row):
        raise AssertionError('Non-finite boundary residual')
    violating=values[3:]
    scale=max(abs(value) for row in violating for value in row)
    if scale==0 or any(any(value==0 for value in row) for row in violating):
        raise AssertionError('Violated top boundary was reported as satisfied')
    if any(abs(value)>max(scale*1e-7,1e-300) for row in values[:3] for value in row):
        raise AssertionError('A satisfied top boundary has a nonzero residual')
    integration=training_boundary()
    return {'satisfied_fields':3,'violating_fields':3,'radii':4,
            'equivalent_sign_and_scaling_accepted':True,'training_integration':integration}


def training_boundary():
    result=candidate('''
import torch
import baseline_models as bm
torch.set_default_dtype(torch.float64)
class Net(torch.nn.Module):
    def __init__(self,values):
        super().__init__();self.w=torch.nn.Parameter(torch.tensor(values))
    def forward(self,x):
        return self.w[0]*x[:,:1]**2+self.w[1]*x[:,1:2]**2+self.w[2]
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.cu=Net([.013,.021,300.]);self.ox=Net([.017,.028,310.])
        self.spec={'xpinn':False};self.cfg={'physics':{'R_Cu':2.5,'R_out':3.,'H':25.,
            'T_top':380.,'T_bottom':300.,'k_Cu':400.,'k_SiO2':1.4,'pde_eps':1e-4}}
model=Model()
samples={key:__import__('numpy').array([[.3,2.],[1.1,7.],[2.7,19.]])
         for key in ['cu','ox','top_cu','top_ox','bottom_cu','bottom_ox','axis','outer','interface']}
samples['top_cu'][:,1]=25.;samples['top_ox'][:,1]=25.
samples['bottom_cu'][:,1]=0.;samples['bottom_ox'][:,1]=0.
physics=bm.PhysicsLoss(model,samples)
params=list(model.parameters())
first=physics.components()['dir']
g1=torch.autograd.grad(first,params,allow_unused=True)
original=bm.top_residual
penalty=sum(original(net,*physics.points[key],model.cfg['physics']).square().mean()
            for net,key in [(model.cu,'top_cu'),(model.ox,'top_ox')])
gp=torch.autograd.grad(penalty,params,allow_unused=True)
bm.top_residual=lambda net,r,z,p:2*original(net,r,z,p)
second=physics.components()['dir'];g2=torch.autograd.grad(second,params,allow_unused=True)
def vector(items):
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                      for p,g in zip(params,items)]).detach().tolist()
print('RESULT:'+json.dumps({'loss_before':float(first),'loss_after':float(second),
                          'top_penalty':float(penalty),'grad_before':vector(g1),
                          'grad_after':vector(g2),'grad_penalty':vector(gp)}))
''')
    assert math.isfinite(result['top_penalty']) and result['top_penalty']>0
    assert math.isclose(result['loss_after']-result['loss_before'],3*result['top_penalty'],rel_tol=1e-7,abs_tol=1e-6), 'Top residual is absent from the training loss'
    g1,g2,gp=[result[k] for k in ('grad_before','grad_after','grad_penalty')]
    assert len(g1)==len(g2)==len(gp)==6
    assert any(abs(v)>1e-9 for v in gp), 'Top constraint has no parameter gradient'
    assert all(math.isfinite(v) for values in (g1,g2,gp) for v in values)
    assert all(math.isclose(b-a,3*c,rel_tol=1e-6,abs_tol=1e-5) for a,b,c in zip(g1,g2,gp)), 'Top loss is detached from training gradients'
    return {'loss_and_parameter_gradients_checked':True}


def reported_errors(prediction,reference,radii):
    payload=json.dumps([prediction,reference,radii],allow_nan=False)
    return candidate('''
import numpy as np
from error_metrics import relative_errors
pred,ref,r=json.loads('''+repr(payload)+''')
result=np.asarray(relative_errors(np.array(pred),np.array(ref),np.array(r)))
assert result.shape==np.asarray(ref).shape
print('RESULT:'+json.dumps(result.tolist(),allow_nan=False))
''')


def pointwise_error():
    with (HERE/'baseline_error_fixture.csv').open(encoding='utf-8',newline='') as f:
        legacy=list(csv.DictReader(f))
    radii=[float(r['r_um']) for r in legacy]
    reference=[float(r['FEM_K']) for r in legacy]
    prediction=[float(r['PINN_K']) for r in legacy]
    expected=[abs(a-b)/abs(b)*100 for a,b in zip(prediction,reference)]
    span_normalized=[abs(a-b)/80*100 for a,b in zip(prediction,reference)]
    values=reported_errors(prediction,reference,radii)
    if len(values)!=len(expected) or any(not math.isfinite(v) or not math.isclose(v,e,rel_tol=1e-8,abs_tol=1e-8)
                                         for v,e in zip(values,expected)):
        raise AssertionError(f'Real baseline error report mismatch: reported max {max(values):.9f}%, '
                             f'independent pointwise max {max(expected):.9f}% (limit 0.1%)')
    if max(values)>.1:
        raise AssertionError('Real baseline pointwise report exceeds 0.1%')
    cases=[([330.,314.,380.],[300.,320.,380.],[0.,1.,3.]),
           ([310.,330.,315.,390.],[300.,300.,350.,375.],[.01,.9,2.5,3.]),
           ([300.,320.,380.],[300.,320.,380.],[0.,2.,3.])]
    cases.append((cases[0][0],cases[0][1],[3.,.001,0.]))
    for pred,ref,r in cases:
        row=reported_errors(pred,ref,r)
        target=[abs(a-b)/abs(b)*100 for a,b in zip(pred,ref)]
        if len(row)!=len(target) or any(not math.isfinite(v) or not math.isclose(v,e,rel_tol=1e-8,abs_tol=1e-8)
                                       for v,e in zip(row,target)):
            raise AssertionError('Pointwise percentage error mismatch on varying input')
    return {'variation_cases':len(cases),'real_baseline_points':len(legacy),
            'real_baseline_max_pointwise_percent':max(expected),
            'real_baseline_max_delta80_percent':max(span_normalized),
            'max_error_limit_percent':.1}


def load_reference():
    data=[]
    with (HERE/'fem_reference.csv').open(encoding='utf-8-sig') as f:
        for row in csv.reader(line for line in f if not line.startswith('%')):
            values=tuple(map(float,row))
            if len(values)!=3 or not all(math.isfinite(v) for v in values) or values[2]<=0:
                raise RuntimeError('Invalid trusted FEM reference')
            data.append(values)
    if len(data)<100:
        raise RuntimeError('Incomplete trusted FEM reference')
    return data


def fresh_training_fem():
    data=load_reference()
    work=Path(tempfile.mkdtemp(prefix='pinn-verifier-'))
    os.chmod(work,0o755)
    coords=work/'coordinates.csv'
    with coords.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['r_um','z_um'])
        writer.writerows(row[:2] for row in data)
    config=work/'config.json'
    shutil.copyfile('/opt/benchmark/protocol.json',config)
    output=work/'output'; output.mkdir(); os.chown(output,guard.RUNTIME_UID,guard.RUNTIME_UID)
    args=['/app/run_case.py','--points',str(coords),'--config',str(config),
          '--output',str(output),'--seed','42','--device','cpu']
    started=time.monotonic()
    run=guard.execute(args=args,timeout=6600)
    wall_seconds=time.monotonic()-started
    guard.freeze_tree(output)
    (LOG/'training-stdout.txt').write_text(run.stdout,encoding='utf-8')
    (LOG/'training-stderr.txt').write_text(run.stderr,encoding='utf-8')
    if run.returncode:
        raise AssertionError('Fresh training failed: '+run.stderr[-2500:])
    with (output/'prediction.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['r_um','z_um','temperature_K']:
            raise AssertionError('Wrong prediction schema')
        prediction=[tuple(float(row[key]) for key in reader.fieldnames) for row in reader]
    if len(prediction)!=len(data):
        raise AssertionError('Every FEM coordinate requires a prediction')
    errors=[]
    for (r,z,t),(rf,zf,tf) in zip(prediction,data):
        if not all(map(math.isfinite,(r,z,t))) or abs(r-rf)>1e-10 or abs(z-zf)>1e-10:
            raise AssertionError('Non-finite temperature or mismatched coordinates')
        errors.append(abs(t-tf)/abs(tf)*100)
    if max(errors)>0.1:
        raise AssertionError(f'Worst pointwise FEM error {max(errors):.8f}% exceeds 0.1%')
    reported=reported_errors([row[2] for row in prediction],[row[2] for row in data],[row[0] for row in data])
    if len(reported)!=len(errors) or any(not math.isfinite(v) or not math.isclose(v,e,rel_tol=1e-8,abs_tol=1e-8)
                                       for v,e in zip(reported,errors)):
        raise AssertionError(f'Trained-model error report mismatch: reported max {max(reported):.9f}%, '
                             f'independent pointwise max {max(errors):.9f}% (limit 0.1%)')
    if max(reported)>0.1:
        raise AssertionError('Program-reported maximum error exceeds 0.1%')
    for filename in ('model_final.npz','run.json','training_history.json','prediction.csv'):
        path=output/filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size>50_000_000:
            raise AssertionError('Missing or unsafe output artifact: '+filename)
    audit=guard.execute(code=(HERE/'audit_checkpoint.py').read_text(),user='auditor',timeout=120,
                        args=['--output',str(output),'--config',str(config),'--points',str(coords),
                              '--wall-seconds',str(wall_seconds)])
    if audit.returncode:
        raise AssertionError('Independent checkpoint/run audit failed: '+audit.stderr[-2500:])
    check=json.loads(audit.stdout)
    shutil.copyfile(output/'run.json',LOG/'fresh-run.json')
    return {'points':len(errors),'max_pointwise_relative_error_percent':max(errors),
            'program_reported_max_relative_error_percent':max(reported),'max_error_limit_percent':.1,
            'mean_pointwise_relative_error_percent':sum(errors)/len(errors),
            'training_wall_seconds':time.monotonic()-started,**check}


FUNCTIONS={'environment':environment,'cylindrical_operator':cylindrical_operator,
           'top_boundary':top_boundary,'pointwise_error':pointwise_error,
           'fresh_training_fem':fresh_training_fem}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',choices=list(FUNCTIONS))
    args=parser.parse_args()
    LOG.mkdir(parents=True,exist_ok=True)
    os.chmod(LOG,0o700)
    guard.prepare()
    diagnostic=args.diagnostic is not None
    if not diagnostic:
        (LOG/'reward.txt').write_text('0\n')
    groups={}
    for name in ([args.diagnostic] if diagnostic else FUNCTIONS):
        try:
            detail=FUNCTIONS[name]()
            groups[name]={'passed':True,'weight':WEIGHTS[name],'detail':detail}
        except (AssertionError,subprocess.TimeoutExpired,ValueError,KeyError) as exc:
            groups[name]={'passed':False,'weight':WEIGHTS[name],'error':str(exc)}
        print(name,json.dumps(groups[name]),flush=True)
    if diagnostic:
        print('DIAGNOSTIC ONLY: no formal reward file changed')
        return
    reward=round(sum(item['weight'] for item in groups.values() if item['passed']),10)
    (LOG/'breakdown.json').write_text(json.dumps({'groups':groups,'reward':reward},indent=2),encoding='utf-8')
    (LOG/'reward.txt').write_text(str(reward)+'\n')
    print('reward =',reward,flush=True)


if __name__=='__main__':
    main()
