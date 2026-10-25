"""Regression for the correlation fit retained in the Z prior."""
import numpy as np
import stim
from darawork.decoders.dem_fit import fit_dem


def test_dem_fit_recovers_true_rates():
    model = stim.DetectorErrorModel("error(0.02) D0 D1 L0\nerror(0.02) D0\nerror(0.02) D1")
    truth = stim.DetectorErrorModel("error(0.12) D0 D1 L0\nerror(0.04) D0\nerror(0.06) D1")
    dets, _, _ = truth.compile_sampler(seed=11).sample(20000)
    fitted = fit_dem(model, dets)
    rates = [instruction.args_copy()[0] for instruction in fitted if instruction.type == "error"]
    assert np.allclose(rates, [.12, .04, .06], atol=.015)
    assert [i.targets_copy() for i in fitted] == [i.targets_copy() for i in model]
