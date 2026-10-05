"""Explicitly gated held-out reporting only: one embedding/ranking per question."""
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
OUT = ROOT/'data/experiments/phase4-e3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Validate DEV architecture selection before parsing any held-out content.
    selection = json.loads((OUT/'selected_dev.json').read_text(encoding='utf8'))
    assert selection['selected_system'] == 'E3'
    assert selection['independent_review'] == 'PASS'
    assert selection['dev_results_sha256'] == sha(OUT/'dev_results.json')
    assert selection['implementation_sha256'] == sha(ROOT/'experiments/phase4_e3/identifier_recovery.py')
    assert selection['threads'] == 4 and selection['rrf_k'] == 60
    dev = json.loads((OUT/'dev_results.json').read_text(encoding='utf8'))
    assert selection['model_revision'] == dev['model']['revision']
    destination = OUT/'eval_results.json'
    # Exclusive reservation prevents a second held-out run even after a crash.
    with destination.open('x',encoding='utf8') as handle:
        json.dump({'status':'reserved_once_only','selection_sha256':sha(OUT/'selected_dev.json')},handle)
    result = {'status':'running_once_only','selection':selection,'timings':{},'observations':[]}
    try:
        start = time.perf_counter()
        import torch
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download
        from retrieval.hybrid import UnifiedRetriever,rrf_fuse
        from evaluation import g2_scoring as G2
        from experiments.phase4_e3.identifier_recovery import ExperimentalBM25Index
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        result['timings']['imports_ms'] = (time.perf_counter()-start)*1000
        t = time.perf_counter()
        r = UnifiedRetriever.from_artifacts()
        r._bm25 = ExperimentalBM25Index(list(r._chunks_by_id.values()))
        result['timings']['artifact_index_init_ms'] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        snapshot = snapshot_download('sentence-transformers/'+r._model_id,
                                     revision=r._model_revision,local_files_only=True)
        assert r._model_revision == selection['model_revision']
        r._model = SentenceTransformer(snapshot,device='cpu',trust_remote_code=False).eval()
        result['timings']['model_init_ms'] = (time.perf_counter()-t)*1000
        warmup = json.loads((ROOT/'data/eval/gold_questions.json').read_text(encoding='utf8'))['questions'][0]['text']
        r.retrieve(warmup,10,'hybrid')
        goldpath = ROOT/'data/eval/gold_eval_questions.json'
        gold = json.loads(goldpath.read_text(encoding='utf8'))
        assert len(gold['questions']) == 12
        result['gold_sha256'] = sha(goldpath)
        for q in gold['questions']:
            # Exactly one full semantic query encoding; lexical and semantic
            # top10 reporting reuse the same side rankings used by Hybrid.
            t = time.perf_counter()
            bm = r._full_bm25_ranking(q['text'],None)
            bm_ms = (time.perf_counter()-t)*1000
            t = time.perf_counter()
            sem = r._full_semantic_ranking(q['text'],None)
            sem_ms = (time.perf_counter()-t)*1000
            t = time.perf_counter()
            hy = rrf_fuse(bm,sem,10)
            fuse_ms = (time.perf_counter()-t)*1000
            modes = {}
            for name,hits,ms in [('bm25',bm[:10],bm_ms),('semantic',sem[:10],sem_ms),
                                 ('hybrid',hy,bm_ms+sem_ms+fuse_ms)]:
                ids = [h['chunk_id'] for h in hits]
                primary = {x['chunk_id'] for x in q['targets'] if x['grade']=='primary'}
                modes[name] = {'hits':hits,'retrieved_ids':ids,'latency_ms':ms,
                               'scores_provisional':G2.score_question(ids,q['targets']),
                               'coverage@5':float(bool(primary.intersection(ids[:5]))),
                               'coverage@10':float(bool(primary.intersection(ids[:10])))}
            result['observations'].append({'qid':q['qid'],'lang':q['lang'],'category':q['category'],
                                           **gold['_header']['verification'][q['qid']],
                                           'per_mode':modes})
        evidence = {'modes':['bm25','semantic','hybrid'],'questions':result['observations']}
        result['reports'] = G2.build_mode_reports(evidence)
        for name in evidence['modes']:
            for metric in ['coverage@5','coverage@10']:
                result['reports'][name][metric] = sum(x['per_mode'][name][metric] for x in result['observations'])/12
        baseline = json.loads((ROOT/'data/eval/retrieval_g2_eval.json').read_text(encoding='utf8'))
        result['accepted_e0_reporting'] = baseline['reports']
        result['latency_method'] = 'Single full side ranking each per query; hybrid=sum(BM25+semantic+RRF stage timings). No cold init, no EVAL warmup, no E0 rerun.'
        manifest = json.loads((OUT/'preservation_before.json').read_text())
        changed = [x['path'] for x in manifest if sha(ROOT/x['path']) != x['sha256']]
        assert not changed,changed
        result['integrity_after'] = {'files_checked':len(manifest),'changed':changed}
        result['status'] = 'complete_eval_once_reporting_only'
    except Exception as exc:
        result['status'] = 'failed_once_only_no_retry'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['reports'],indent=2),flush=True)


if __name__ == '__main__':
    main()
