from common import *
T=json.loads((D/'thresholds_L100.json').read_text())['100'];cs=[]
for key,kind,tau in [('SmokeNative',0,0),('SmokeOffMLP',2,0),('SmokeOffLR',3,0),('SmokeStaticMLP',2,T['state'][4]['tau']),('SmokeIdentityMLP',4,T['state'][4]['tau']),('SmokeStaticLR',3,T['lr'][4]['tau']),('SmokeIdentityLR',5,T['lr'][4]['tau'])]:
    cs.append(dict(key=key,family='Smoke',kind=kind,training_L=100 if kind>=2 else 0,inference_L=100,tau=tau,late_tau=tau,rank=72,exp=100,base_scale=1.,multiplier=1.,diagnostic=True))
rs=load('validation',cs,99,True);V={r['key']:r for r in rs}
for a,b in [('SmokeNative','SmokeOffMLP'),('SmokeNative','SmokeOffLR'),('SmokeStaticMLP','SmokeIdentityMLP'),('SmokeStaticLR','SmokeIdentityLR')]:
    for k in ['ids_sha256','reads','expansions','total_hits']:assert V[a][k]==V[b][k],(a,b,k)
assert V['SmokeNative']['ids_sha256']==sha(W/'runs/validation/DiskANN_L100_p0_r0.ids')
assert V['SmokeStaticMLP']['ids_sha256']==sha(W/'runs/validation/MLP_L100_L100_p4_r0.ids')
dump(D/'SMOKE_COMPLETE.json',dict(passed=True,model_off_equals_native=True,identity_equals_static=True,historical_native_and_MLP_equal=True,runs=rs))
print('SMOKE PASSED',flush=True)
