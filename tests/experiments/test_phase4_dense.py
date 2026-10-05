import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('phase4_dense', ROOT/'experiments/phase4_quality/run_dense.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_dense_rank_ties_are_deterministic():
    hits = module.dense_ranking(['z','a','b'], np.asarray([[1.,0.],[1.,0.],[0.,1.]]), [1.,0.])
    assert [h['chunk_id'] for h in hits] == ['a','z','b']
    assert [h['rank'] for h in hits] == [1,2,3]

def test_cls_pooling_ignores_non_cls_and_normalizes():
    import torch
    hidden = torch.tensor([[[3.,4.],[100.,0.]], [[0.,2.],[-3.,8.]]])
    pooled = module.cls_l2(hidden)
    assert torch.allclose(pooled, torch.tensor([[0.6,0.8],[0.,1.]]))

def test_dev_only_fixed_configuration():
    source = (ROOT/'experiments/phase4_quality/run_dense.py').read_text(encoding='utf-8-sig')
    assert 'gold_eval' not in source and 'retrieval_g2_eval' not in source
    assert module.MAX_LENGTH == 512 and module.BATCH_SIZE == 1 and module.THREADS == 4
    assert module.REVISION == '5617a9f61b028005a4858fdac845db406aefb181'
    assert 'weights_only=True' in source

def test_frozen_metrics_known_example():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from evaluation.g2_scoring import score_question
    value = score_question(['a','x'], [{'chunk_id':'a','grade':'primary'},{'chunk_id':'b','grade':'primary'}])
    assert value['recall@5'] == 0.5 and value['recall@10'] == 0.5
    assert abs(value['ndcg@10'] - 0.6131471927654584) < 1e-12
