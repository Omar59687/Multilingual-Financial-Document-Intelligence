"""Isolated DEV-only BM25+BGE-M3 dense experiment; never writes accepted paths."""
import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
import hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'data/experiments/phase4-quality'
REPO='BAAI/bge-m3'
REVISION='5617a9f61b028005a4858fdac845db406aefb181'
FILES=['config.json','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json','pytorch_model.bin']
MAX_LENGTH=512
BATCH_SIZE=1
THREADS=4

def cls_l2(last_hidden):
    import torch
    return torch.nn.functional.normalize(last_hidden[:,0],p=2,dim=1)

def dense_ranking(ids,matrix,query):
    import numpy as np
    scores=matrix @ np.asarray(query,dtype=np.float64)
    order=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))
    return [{'chunk_id':ids[i],'score':float(scores[i]),'rank':rank,'source':'semantic'} for rank,i in enumerate(order,1)]

def check_integrity():
    before=json.loads((OUT/'preservation_before.json').read_text(encoding='utf-8-sig'))
    changed=[x['path'] for x in before if hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()!=x['sha256']]
    assert not changed,changed
    return {'files_checked':len(before),'changed':changed}

def main():
    import numpy as np, torch, psutil
    from huggingface_hub import snapshot_download
    from transformers import AutoModel,AutoTokenizer
    from retrieval.hybrid import UnifiedRetriever,rrf_fuse
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    result={'status':'running','configuration':{'repo':REPO,'revision':REVISION,'max_length':MAX_LENGTH,'batch_size':BATCH_SIZE,'threads':THREADS,'dtype':'float32','pooling':'CLS+L2','rrf_k':60,'pilot_budget_seconds':1800,'source':'https://huggingface.co/BAAI/bge-m3'},'timings':{},'integrity_before':check_integrity()}
    def save(): (OUT/'e2_dev.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    save()
    t=time.perf_counter()
    snapshot=snapshot_download(REPO,revision=REVISION,allow_patterns=FILES,local_files_only=True)
    result['timings']['cache_resolution_ms']=(time.perf_counter()-t)*1000
    result['snapshot']=snapshot; save()
    t=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    model=AutoModel.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32,low_cpu_mem_usage=True,weights_only=True).eval().to('cpu')
    result['timings']['bge_init_ms']=(time.perf_counter()-t)*1000
    result['parameters']=sum(p.numel() for p in model.parameters())
    result['rss_bytes_after_load']=psutil.Process().memory_info().rss
    r=UnifiedRetriever.from_artifacts()
    chunks=list(r._chunks_by_id.values()); ids=[c['chunk_id'] for c in chunks]
    def encode(texts):
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
        with torch.inference_mode(): return cls_l2(model(**tokens).last_hidden_state).cpu().numpy()
    # Fixed corpus-order pilot, independent of gold targets/query IDs.
    pilot_indices=sorted(set(range(0,len(chunks),max(1,len(chunks)//16))) | {max(range(len(chunks)),key=lambda i:len(chunks[i]['text']))})
    pilot=[chunks[i] for i in pilot_indices]
    t=time.perf_counter(); encode([pilot[0]['text']]); result['timings']['bge_warmup_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter()
    for c in pilot: encode([c['text']])
    pilot_sec=time.perf_counter()-t
    result['pilot']={'indices':pilot_indices,'n_chunks':len(pilot),'seconds':pilot_sec,'projected_full_seconds':pilot_sec/len(pilot)*len(chunks),'available_ram':psutil.virtual_memory().available}; save()
    print('PILOT',result['pilot'],flush=True)
    # Bound CPU full encoding to under thirty minutes; no partial-corpus metric claims.
    if result['pilot']['projected_full_seconds']>1800:
        result['status']='stopped_cpu_cost';result['integrity_after']=check_integrity();save();return
    result['status']='pilot_complete_awaiting_gate';save();return
if __name__=='__main__': main()
