import json,subprocess,sys,time
from pathlib import Path
out=Path('data/experiments/phase4-quality')
start=time.perf_counter()
try:
 p=subprocess.run([sys.executable,'-B','experiments/phase4_quality/run_dense.py'],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=300)
 report={'exit_code':p.returncode,'elapsed_seconds':time.perf_counter()-start,'stdout':p.stdout[-4000:],'stderr':p.stderr[-6000:]}
 if p.returncode:report['status']='stopped_pilot_exception'
except subprocess.TimeoutExpired as exc:
 report={'status':'stopped_pilot_timeout','elapsed_seconds':time.perf_counter()-start,'stdout':str(exc.stdout)[-3000:],'stderr':str(exc.stderr)[-3000:]}
(out/'bge_pilot_execution.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)
