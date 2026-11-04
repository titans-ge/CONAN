from types import SimpleNamespace as SN

import matplotlib.pyplot as plt
import numpy as np
import pytest
from uncertainties import ufloat

from CONAN._classes import load_result


@pytest.mark.parametrize('sort_lcs', [True, False])
@pytest.mark.parametrize('custom_labels', [None, {
    'ttv00-lc1': 'Observatory A',
    'ttv05-lc6': 'Partial transit',
    'ttv05-lc6-T0_pl1': 'Observatory B',
}])
def test_shared_ttv_lightcurves_keep_their_transit_times(sort_lcs, custom_labels):
    # Input order differs from chronology; lc5 shares lc3's T0 and lc6 shares lc1's.
    centers = [20., 0., 40., 10., 40., 20., 30.]
    names = [f'lc{i+1}' for i in range(7)]
    labels = [f'ttv{i:02d}-lc{i+1}-T0_pl1' for i in range(7)]
    fitted = [0, 1, 2, 3, 6]
    params_dict = {labels[i]: ufloat(centers[i], .001) for i in fitted}
    shared = {labels[2]: [labels[4]], labels[0]: [labels[5]]}
    outdata, smooth, conf = {}, {}, []
    for i, (name, center) in enumerate(zip(names, centers)):
        time = np.linspace(center-.15+i*.01, center+.15, 31)
        flux = np.full(time.size, 1.+i*.001)
        outdata[name] = {'time': time, 'det_flux': flux,
                         'error': np.full(time.size, .001)}
        smooth[name] = SN(time=time, model=np.ones(time.size))
        conf.append(SN(plnum=[0], t0s=[center], tr_edges=[(time[0], time[-1])]))
    result = SN(_nplanet=1, _ttvs=conf, params_dict=params_dict,
                _ind_para={'shared_params': shared},
                params=SN(T0=[0.], P=[10.], dur=[.2]),
                lc=SN(names=names, outdata=outdata), _lc_smooth_time_mod=smooth)

    fig = load_result._ttv_lcplot(result, offset=.1, binsize=.02, sort_lcs=sort_lcs,
                                  labels=custom_labels)
    try:
        points = [line for line in fig.axes[0].lines if line.get_alpha() == .3]
        order = [1, 3, 0, 5, 6, 2, 4] if sort_lcs else list(range(7))
        assert len(points) == len(order)
        annotations = fig.axes[0].texts
        expected_labels = [labels[i].rsplit('-T0_pl', 1)[0] for i in order]
        if custom_labels is not None:
            expected_labels = [custom_labels.get(labels[i], custom_labels.get(text, text))
                               for i, text in zip(order, expected_labels)]
        assert [text.get_text() for text in annotations] == expected_labels
        for row, (line, lc_index) in enumerate(zip(points, order)):
            np.testing.assert_allclose(line.get_ydata(), 1.+lc_index*.001+row*.1)
            np.testing.assert_allclose(annotations[row].xy,
                                       [0., 1.+row*.1])
            assert annotations[row].xycoords is fig.axes[0].get_yaxis_transform()
            assert annotations[row].get_position()[0] == 4
        assert list(params_dict) == [labels[i] for i in fitted]
    finally:
        plt.close(fig)
