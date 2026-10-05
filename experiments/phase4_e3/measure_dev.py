"""DEV-only paired E0/E3 measurement, isolated outputs and frozen gold."""
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
OUT = ROOT/'data/experiments/phase4-e3'
THREADS = 4
REPEATS = 3


def integrity():
    manifest = json.loads((OUT/'preservation_before.json').read_text())
    changed = [x['path'] for x in manifest
               if hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest() != x['sha256']]
    assert not changed, changed
    return {'files_checked':len(manifest), 'changed':changed}


def main():
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from retrieval.hybrid import UnifiedRetriever
    from evaluation import g2_scoring as G2
    from experiments.phase4_e3.identifier_recovery import ExperimentalBM25Index

    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    result = {'configuration':{'threads':THREADS,'repeats':REPEATS,'rrf_k':60,
                              'warmup_calls_per_system':1,'order':'E0/E3 on even repeats; E3/E0 on odd repeats'},
              'integrity_before':integrity(), 'timings':{}}
    result['timings']['imports_ms'] = (time.perf_counter()-START)*1000
    t = time.perf_counter()
    e0 = UnifiedRetriever.from_artifacts()
    e3 = UnifiedRetriever(list(e0._chunks_by_id.values()), vectors=e0._vectors,
                          model_id=e0._model_id, model_revision=e0._model_revision)
    e3._bm25 = ExperimentalBM25Index(list(e0._chunks_by_id.values()))
    result['timings']['artifact_index_init_ms'] = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    snapshot = snapshot_download('sentence-transformers/'+e0._model_id,
                                 revision=e0._model_revision, local_files_only=True)
    model = SentenceTransformer(snapshot,device='cpu',trust_remote_code=False).eval()
    e0._model = e3._model = model
    result['timings']['model_init_ms'] = (time.perf_counter()-t)*1000
    result['model'] = {'id':e0._model_id,'revision':e0._model_revision,'snapshot':snapshot}
    result['versions'] = {name:importlib.metadata.version(name) for name in
                          ['torch','transformers','sentence-transformers','numpy','huggingface-hub']}
    gold = json.loads((ROOT/'data/eval/gold_questions.json').read_text(encoding='utf8'))
    accepted = json.loads((ROOT/'data/eval/retrieval_g2_dev.json').read_text(encoding='utf8'))
    questions = gold['questions']
    result['gold_sha256'] = hashlib.sha256((ROOT/'data/eval/gold_questions.json').read_bytes()).hexdigest()
    # Cheap tokenizer-only truncation evidence for every DEV target, no new model pass.
    token_evidence = []
    for q in questions:
        for target in q['targets']:
            text = e0.get_chunk(target['chunk_id'])['text']
            tokens = model.tokenizer(text, truncation=False)['input_ids']
            visible = model.tokenizer.decode(tokens[:model.max_seq_length])
            token_evidence.append({'qid':q['qid'],'chunk_id':target['chunk_id'],
                                   'token_count':len(tokens),'max_seq_length':model.max_seq_length,
                                   'decoded_visible_window':visible})
    result['target_tokenizer_evidence'] = token_evidence
    for retriever in [e0,e3]:
        retriever.retrieve(questions[0]['text'],10,'hybrid')
    rows = []
    for repeat in range(REPEATS):
        systems = [('E0',e0),('E3',e3)] if repeat%2 == 0 else [('E3',e3),('E0',e0)]
        for q in questions:
            for name,retriever in systems:
                t = time.perf_counter()
                hits = retriever.retrieve(q['text'],10,'hybrid')
                elapsed = (time.perf_counter()-t)*1000
                ids = [h['chunk_id'] for h in hits]
                if name == 'E0':
                    prior = next(x for x in accepted['evidence']['questions'] if x['qid']==q['qid'])
                    assert ids == prior['per_mode']['hybrid']['retrieved_ids']
                scores = G2.score_question(ids,q['targets'])
                primary = {t['chunk_id'] for t in q['targets'] if t['grade']=='primary'}
                scores['coverage@5'] = float(bool(primary.intersection(ids[:5])))
                scores['coverage@10'] = float(bool(primary.intersection(ids[:10])))
                bm = retriever._full_bm25_ranking(q['text'],None)
                bm_ranks = {h['chunk_id']:h['rank'] for h in bm}
                rows.append({'system':name,'repeat':repeat,'qid':q['qid'],'lang':q['lang'],
                             'category':q['category'],**gold['_header']['verification'][q['qid']],
                             'latency_ms':elapsed,'hits':hits,**scores,
                             'primary_bm25_ranks':{cid:bm_ranks.get(cid) for cid in primary}})
    result['observations'] = rows
    result['reports'] = {}
    for name in ['E0','E3']:
        group = [r for r in rows if r['system']==name]
        first = [r for r in group if r['repeat']==0]
        assert all([h['chunk_id'] for h in row['hits']] ==
                   [h['chunk_id'] for h in next(r for r in first if r['qid']==row['qid'])['hits']]
                   for row in group)
        result['reports'][name] = {'slices':G2.slice_report(first),
                                  'latency':G2.latency_summary([r['latency_ms'] for r in group]),
                                  'coverage@5':sum(r['coverage@5'] for r in first)/len(first),
                                  'coverage@10':sum(r['coverage@10'] for r in first)/len(first)}
    deltas = []
    for q in questions:
        pair = {name:next(r for r in rows if r['qid']==q['qid'] and r['system']==name and r['repeat']==0)
                for name in ['E0','E3']}
        deltas.append({'qid':q['qid'], **{m:pair['E3'][m]-pair['E0'][m]
                                       for m in ['recall@5','recall@10','ndcg@10','coverage@5','coverage@10']},
                       'e0_ids':[h['chunk_id'] for h in pair['E0']['hits']],
                       'e3_ids':[h['chunk_id'] for h in pair['E3']['hits']]})
    result['per_question_deltas'] = deltas
    result['integrity_after'] = integrity()
    result['timings']['total_process_ms'] = (time.perf_counter()-START)*1000
    result['status'] = 'complete_dev_only'
    (OUT/'dev_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['reports'],indent=2),flush=True)


if __name__ == '__main__':
    main()
