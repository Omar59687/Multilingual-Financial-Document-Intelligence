import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from retrieval.bm25 import BM25Index
from experiments.phase4_e3.identifier_recovery import (
    ExperimentalBM25Index, identifier_token_runs,
)


@pytest.fixture
def chunks():
    return [
        {'chunk_id':'a','text':'record QZX-2091-0042 branch east'},
        {'chunk_id':'b','text':'record QZX-2091-0043 branch west'},
        {'chunk_id':'c','text':'record QZX-2091-0042 ALT-77 branch east'},
        {'chunk_id':'d','text':'ordinary record period R2 ABC123'},
    ]


def test_generic_exact_missing_close_and_multiple_ids(chunks):
    index = ExperimentalBM25Index(chunks)
    assert set(index.score('QZX-2091-0042 branch')) == {'a','c'}
    assert set(index.score('QZX-2091-0043 branch')) == {'b'}
    assert index.score('QZX-2091-9999 branch') == {}
    assert set(index.score('QZX-2091-0042 ALT-77')) == {'c'}
    assert index.score('QZX-2091-0042 ALT-78') == {}


def test_compact_context_is_ordinary_and_not_special_cased(chunks):
    index = ExperimentalBM25Index(chunks)
    assert identifier_token_runs('R2 FY2091 ABC123 Z9') == []
    assert set(index.score('QZX-2091-0042 R2 branch')) == {'a','c'}
    assert set(index.score('ABC123 branch')) == {'a','b','c','d'}
    assert BM25Index(chunks).score('MISSING123 branch') == {}
    assert set(index.score('MISSING123 branch')) == {'a','b','c'}


def test_frozen_normalizer_casing_digits_and_punctuation(chunks):
    index = ExperimentalBM25Index(chunks)
    expected = index.score('QZX-2091-0042')
    assert index.score('(qzx/٢٠٩١/٠٠٤٢),') == expected
    assert identifier_token_runs('QZX_2091_0042') == []
    assert identifier_token_runs('QZX 2091 0042') == []


def test_plain_queries_and_segmented_ids_equal_accepted(chunks):
    accepted, experiment = BM25Index(chunks), ExperimentalBM25Index(chunks)
    for query in ['branch east', 'ordinary record', '', 'zzzz', 'QZX-2091-0042 branch']:
        assert experiment.score(query) == accepted.score(query)
        assert experiment.retrieve(query, 3) == accepted.retrieve(query, 3)


def test_public_errors_filters_ties_and_empty_index(chunks):
    index = ExperimentalBM25Index(chunks)
    with pytest.raises(TypeError): index.score(None)
    for bad in [False, 0, 101, '2']:
        with pytest.raises(TypeError): index.retrieve('record', bad)
    with pytest.raises(TypeError): index.retrieve('record', 2, [])
    assert index.retrieve('record', 10, {'b':chunks[1]})[0]['chunk_id'] == 'b'
    assert ExperimentalBM25Index().score('record') == {}
    tied = ExperimentalBM25Index([{'chunk_id':'z','text':'same'},{'chunk_id':'a','text':'same'}])
    assert [h['chunk_id'] for h in tied.retrieve('same', 10)] == ['a','z']


def test_no_shared_global_mutation(chunks):
    accepted = BM25Index(chunks)
    before = accepted.score('QZX-2091-0042 R2 branch')
    ExperimentalBM25Index(chunks).score('QZX-2091-0042 R2 branch')
    assert accepted.score('QZX-2091-0042 R2 branch') == before == {}
