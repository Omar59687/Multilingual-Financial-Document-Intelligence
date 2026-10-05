import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,time,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from retrieval.hybrid import UnifiedRetriever
from evaluation import g2_scoring as G2
from huggingface_hub import snapshot_download
import torch
from sentence_transformers import SentenceTransformer
out=Path('data/experiments/phase4-quality')
manifest=json.loads((out/'preservation_before.json').read_text(encoding='utf-8-sig'))
for name in ['build_retrieval_index.py','build_vectors.py','measure_retrieval.py']:
 p=Path('scripts')/name
 if p.exists():manifest.append({'path':p.as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(out/'preservation_before.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
r=UnifiedRetriever.from_artifacts();t=time.perf_counter()
snapshot=snapshot_download('sentence-transformers/'+r._model_id,revision=r._model_revision,local_files_only=True)
r._model=SentenceTransformer(snapshot,device='cpu',trust_remote_code=False).eval()
init_ms=(time.perf_counter()-t)*1000
gold=json.loads(Path('data/eval/gold_questions.json').read_text(encoding='utf8'));qs=gold['questions']
accepted=json.loads(Path('data/eval/retrieval_g2_dev.json').read_text(encoding='utf8'))
r.retrieve(qs[0]['text'],10,'hybrid')
rows=[]
for repeat in range(3):
 for q in qs:
  t=time.perf_counter();hits=r.retrieve(q['text'],10,'hybrid');ms=(time.perf_counter()-t)*1000
  ids=[h['chunk_id'] for h in hits];prev=next(x for x in accepted['evidence']['questions'] if x['qid']==q['qid'])
  assert ids==prev['per_mode']['hybrid']['retrieved_ids']
  rows.append({'repeat':repeat,'qid':q['qid'],'lang':q['lang'],'category':q['category'],**gold['_header']['verification'][q['qid']],'latency_ms':ms,'hits':hits,**G2.score_question(ids,q['targets'])})
report={'model_init_ms':init_ms,'snapshot':snapshot,'threads':4,'repeats':3,'warmup_queries':1,'observations':rows,'slices':G2.slice_report([x for x in rows if x['repeat']==0]),'latency':G2.latency_summary([x['latency_ms'] for x in rows]),'accepted_top10_matches':True}
(out/'e0_warm.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(report['latency'],report['slices']['overall'],flush=True)
