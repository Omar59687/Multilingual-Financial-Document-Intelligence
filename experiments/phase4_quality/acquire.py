"""Bounded pinned official model acquisition (600 seconds total, at most2 attempts)."""
import json,subprocess,sys,time
from pathlib import Path
OUT=Path('data/experiments/phase4-quality')
config=json.loads((OUT/'bge_acquisition.json').read_text())
files=['config.json','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json','pytorch_model.bin']
start=time.perf_counter();attempts=[]
for attempt in range(2):
 remaining=600-(time.perf_counter()-start)
 if remaining<=0:break
 code="from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-m3',revision="+repr(config['revision'])+",allow_patterns="+repr(files)+",max_workers=2),flush=True)"
 try:
  p=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=remaining)
  attempts.append({'attempt':attempt+1,'exit_code':p.returncode,'stdout':p.stdout[-2000:],'stderr':p.stderr[-4000:]})
  if p.returncode==0:break
 except subprocess.TimeoutExpired:
  attempts.append({'attempt':attempt+1,'timeout':True});break
status='downloaded' if attempts and attempts[-1].get('exit_code')==0 else 'stopped_acquisition'
report={'status':status,'download_ms':(time.perf_counter()-start)*1000,'revision':config['revision'],'files':files,'attempts':attempts}
(OUT/'bge_download.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)
