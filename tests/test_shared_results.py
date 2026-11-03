import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from CONAN._classes import load_result

outputs = importlib.import_module('CONAN.outputs')


@pytest.mark.parametrize('sampled_donor', [True, False])
def test_shared_parameter_summaries(tmp_path, monkeypatch, sampled_donor):
    names = np.array(['rho_star', 'T_0', 'RpRs', 'Impact_para', 'Period',
                      'sesin(w)', 'secos(w)', 'K', 'V_q1', 'V_q2',
                      'V_Fn', 'V_DFocc', 'V_ph_off', 'V_cont'])
    bestfit = np.array([1., 0., .1, .2, 3., 0., 0., 0., .4, np.nan,
                        0., 0., 0., 0.])
    jumping = np.array([8 if sampled_donor else 1])
    fixed = np.array([i for i in range(len(names)) if i not in jumping])
    posterior = np.random.default_rng(24).normal(.3, .02, (500, 1))
    original_bestfit = bestfit.copy()
    original_posterior = posterior.copy()

    def derive(*args):
        # Both the posterior PDFs and best-fit values must use the donor.
        np.testing.assert_array_equal(args[12], args[13])  # q1 and q2
        return [], [], 'test'

    monkeypatch.setattr(outputs, 'derive_parameters', derive)
    monkeypatch.setattr(outputs, 'param_hist', lambda *args, **kwargs: None)
    stats = outputs.mcmc_outputs(
        posterior, names[jumping], (jumping,), names[fixed], (fixed,),
        bestfit, [.6], 1., None, np.random.default_rng(25).normal(1., .01, 500), None, 1, ['V'],
        'Rrho', [], 'km/s', np.empty((500, 0)), 1, str(tmp_path),
        shared_params={'V_q1': ['V_q2']})

    assert all(values.shape == (1,) for values in stats)
    np.testing.assert_array_equal(bestfit, original_bestfit)
    np.testing.assert_array_equal(posterior, original_posterior)
    for stat in ['med', 'max', 'bf']:
        rows = {parts[0]: parts[1:] for line in
                (tmp_path / f'results_{stat}.dat').read_text().splitlines()
                if not line.startswith('#') and (parts := line.split())}
        assert rows['V_q2'] == rows['V_q1']
        assert np.isfinite(float(rows['V_q2'][0]))


@pytest.mark.parametrize('return_type', ['float', 'ufloat', 'array'])
def test_load_legacy_shared_parameter(tmp_path, return_type):
    (tmp_path / 'results_bf.dat').write_text(
        'donor 0.4 -0.1 0.2 -0.3 0.4\nrecipient nan\n')
    result = SimpleNamespace(_folder=str(tmp_path),
                             _ind_para={'shared_params': {'donor': ['recipient']}})
    params = load_result.get_all_params_dict(result, stat='bf',
                                             return_type=return_type, verbose=False)
    assert params['recipient'] is params['donor']
