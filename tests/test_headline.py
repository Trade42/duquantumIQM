import numpy as np
import pytest
from challenge.benchmark_headline import challenge_fit,fit_many,ler_difference
from challenge.decoding import fit_ler


def test_primary_is_exact_challenge_equation_even_when_fit_bad():
    f=challenge_fit([1,2,3],[20,150,260],[1000]*3)
    assert not f['fit_spam'] and f['amplitude']==1
    assert f['stationary_model_rejected_5pct']
    assert f['epsilon_per_round']==fit_ler([1,2,3],[20,150,260],[1000]*3)['epsilon_per_round']


def test_bootstrap_mle_matches_repository_mle_and_zero_boundary():
    counts=np.array([[0,0,0],[30,80,140],[100,190,250],[900,900,900]])
    actual=fit_many(counts,np.full(3,1000))
    expected=[fit_ler([1,2,3],k,[1000]*3)['epsilon_per_round'] for k in counts]
    assert np.allclose(actual,expected,atol=1e-7)


def test_same_shot_identical_decoders_have_zero_difference_interval():
    failures={r:{0:np.arange(1000)%11==0,1:np.arange(1000)%13==0} for r in (1,2,3)}
    result=ler_difference(failures,failures,paired=True,draws=100)
    assert result['absolute_ler_reduction']==0
    assert result['ci']==[0,0]
