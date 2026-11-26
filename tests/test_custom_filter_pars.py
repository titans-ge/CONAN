"""tests for filter-dependent parameters of the custom LC function"""
import sys
import numpy as np
import pytest

from CONAN import load_lightcurves, fit_setup, run_fit, create_configfile, load_configfile
from CONAN.misc import compare_objs, _custom_func_for_filter, _custom_par_indices, _custom_arg_names
from CONAN.conf import _split_top_level, _custom_func_prior_value

CALLS = []   # records the arguments the custom function is called with

def offset_model(time, A, B, extra_args=dict(unit="ppm", scale=1)):
    CALLS.append((len(time), float(A), float(B), extra_args.get("filter"), extra_args.get("unit"), extra_args.get("scale")))
    return A + B + 0*time

def add_op(transit_model, custom_model):
    return transit_model + custom_model


NPTS = {"lcV.dat": 200, "lcR.dat": 150}   # different lengths to identify the lc in CALLS

def make_lc_obj(tmp_path, verbose=False):
    rng = np.random.default_rng(42)
    for fname, n in NPTS.items():
        t   = np.linspace(-0.2, 0.2, n)
        arr = np.column_stack([t, 1+rng.normal(0,1e-4,n), 1e-3*np.ones(n)] + [np.zeros(n)]*6)
        np.savetxt(tmp_path/fname, arr)
    lc_obj = load_lightcurves(file_list=list(NPTS), data_filepath=str(tmp_path), filters=["V","R"], wl=[0.5,0.7], verbose=verbose)
    lc_obj.planet_parameters(RpRs=0.1, Impact_para=0.2, rho_star=1.5, T_0=0, Period=3, verbose=verbose)
    lc_obj.limb_darkening(q1=0.3, q2=0.3, verbose=verbose)
    return lc_obj


def test_setup_filter_dependent_args(tmp_path):
    lc_obj = make_lc_obj(tmp_path)
    cf = lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                       func_args=dict(A=(0,0.001,0.01), B=[(-0.01,0.002,0.01), 0.003]))
    assert list(cf.par_dict) == ["A", "V_B", "R_B"]
    assert cf.filt_args == ["B"] and cf.arg_names == ["A","B"]
    assert cf.npars == 3 and cf.nfree == 2
    np.testing.assert_array_equal(_custom_par_indices(cf, "V"), [0, 1])
    np.testing.assert_array_equal(_custom_par_indices(cf, "R"), [0, 2])

    # resolution for a filter
    cR = _custom_func_for_filter(cf, "R")
    assert cR.func_args == {"A": (0,0.001,0.01), "B": 0.003}
    assert list(cR.par_dict) == ["A","B"] and cR.filt_args == []
    assert cf.func_args["B"] == [(-0.01,0.002,0.01), 0.003]   # original untouched

    # length-1 list is replicated for all filters as independent parameters
    cf1 = lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                        func_args=dict(A=0, B=[(0.002,0.001)]))
    assert list(cf1.par_dict) == ["A", "V_B", "R_B"] and cf1.nfree == 2

    with pytest.raises(AssertionError):
        lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                      func_args=dict(A=0, B=[0.1,0.2,0.3]))


def test_legacy_namespace_without_filter_attrs():
    from types import SimpleNamespace as SN
    from CONAN.misc import _param_obj
    legacy = SN(func_args={"A":(0,1)}, par_dict={"A":_param_obj.from_tuple((0,1))}, npars=1)
    assert _custom_func_for_filter(legacy, "V") is legacy
    np.testing.assert_array_equal(_custom_par_indices(legacy, "V"), [0])
    assert _custom_arg_names(legacy) == ["A"]


def test_prior_string_parsing():
    assert _split_top_level("A:U(0,1,2),B:[N(1,0.1),F(2)]") == ["A:U(0,1,2)", "B:[N(1,0.1),F(2)]"]
    assert _custom_func_prior_value("[N(1,0.1),F(2)]") == [(1,0.1), 2]
    assert _custom_func_prior_value(["N(1,0.1)","U(0,1,2)"]) == [(1,0.1),(0,1,2)]
    assert _custom_func_prior_value("TN(0,1,0.5,0.1)") == (0,1,0.5,0.1)


def test_config_roundtrip(tmp_path, monkeypatch):
    lc_obj  = make_lc_obj(tmp_path)
    lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                  func_args=dict(A=(0,0.001,0.01), B=[(-0.01,0.002,0.01), (0.003,0.001)]))
    fit_obj = fit_setup(verbose=False)
    cfg_dir = tmp_path/"cfg"
    cfg_dir.mkdir()
    create_configfile(lc_obj, None, fit_obj, str(cfg_dir/"config.dat"))   # writes .dat, .yaml and custom_LCfunc.py

    dat_txt = (cfg_dir/"config.dat").read_text()
    assert "B:[U(-0.01,0.002,0.01),N(0.003,0.001)]" in dat_txt

    monkeypatch.syspath_prepend(str(cfg_dir))
    saved_module = sys.modules.pop("custom_LCfunc", None)   # load_configfile imports `custom_LCfunc` from sys.path
    try:
        for cfg in ["config.dat", "config.yaml"]:
            sys.modules.pop("custom_LCfunc", None)
            lc2, _, _ = load_configfile(str(cfg_dir/cfg), lc_path=str(tmp_path), verbose=False)
            assert lc2._custom_LCfunc.func_args == lc_obj._custom_LCfunc.func_args, cfg
            assert list(lc2._custom_LCfunc.par_dict) == ["A","V_B","R_B"], cfg
            assert [p.prior_str for p in lc2._custom_LCfunc.par_dict.values()] == \
                   [p.prior_str for p in lc_obj._custom_LCfunc.par_dict.values()], cfg
    finally:   # do not leak this test's custom_LCfunc module to other tests
        sys.modules.pop("custom_LCfunc", None)
        if saved_module is not None: sys.modules["custom_LCfunc"] = saved_module


def test_fit_uses_filter_values(tmp_path):
    lc_obj  = make_lc_obj(tmp_path)
    lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                  func_args=dict(A=(0,0.001,0.01), B=[(-0.01,0.002,0.01), (0.003,0.001)]))
    fit_obj = fit_setup(verbose=False)

    pnames, jumping, _ = run_fit(lc_obj=lc_obj, fit_obj=fit_obj, get_parameter_names=True, verbose=False)
    assert {"A","V_B","R_B"} <= set(pnames[jumping])

    CALLS.clear()
    assert run_fit(lc_obj=lc_obj, fit_obj=fit_obj, out_folder=str(tmp_path/"out"), init_only=True, verbose=False)
    calls_V = {c[1:4] for c in CALLS if c[0]==NPTS["lcV.dat"]}
    calls_R = {c[1:4] for c in CALLS if c[0]==NPTS["lcR.dat"]}
    assert calls_V == {(0.001, 0.002, "V")}
    assert calls_R == {(0.001, 0.003, "R")}


def test_extra_args_passed_with_filter(tmp_path):
    lc_obj  = make_lc_obj(tmp_path)
    # user extra_args update the signature defaults (unit kept from defaults, scale overridden)
    cf = lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                       func_args=dict(A=(0,0.001,0.01), B=0.002), extra_args=dict(scale=5))
    assert cf.default_extra_args == {"unit":"ppm", "scale":1}
    assert cf.extra_args == {"scale":5}    # stored user input is not modified
    assert _custom_func_for_filter(cf, "R").extra_args == {"unit":"ppm", "scale":5, "filter":"R"}

    CALLS.clear()
    assert run_fit(lc_obj=lc_obj, fit_obj=fit_setup(verbose=False), out_folder=str(tmp_path/"out"), init_only=True, verbose=False)
    assert {c[3:] for c in CALLS} == {("V","ppm",5), ("R","ppm",5)}   # includes model evaluations on finer time grids for plots
    assert {(c[0], c[3:]) for c in CALLS if c[0] in NPTS.values()} == {(NPTS["lcV.dat"], ("V","ppm",5)), (NPTS["lcR.dat"], ("R","ppm",5))}

    # get_decorr also passes the filter of each lc
    CALLS.clear()
    lc_obj.get_decorr(T_0=0, Period=3, rho_star=1.5, RpRs=1e-5, q1=0.3, q2=0.3, exclude_cols=[0,3,4,5,6,7,8],
                      plot_model=False, setup_baseline=False, verbose=False)
    assert {c[3] for c in CALLS} == {"V", "R"}
    assert {(c[0], c[3]) for c in CALLS if c[0] in NPTS.values()} == {(NPTS["lcV.dat"], "V"), (NPTS["lcR.dat"], "R")}


def test_extra_args_reserved_and_required(tmp_path):
    lc_obj = make_lc_obj(tmp_path)
    with pytest.raises(AssertionError, match="reserved"):
        lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                      func_args=dict(A=0, B=0), extra_args=dict(filter="V"))
    def no_extra(time, A, B, other={}): return A+B+0*time
    with pytest.raises(AssertionError, match="extra_args"):
        lc_obj.add_custom_LC_function(func=no_extra, x="time", op_func=add_op, verbose=False, func_args=dict(A=0, B=0))


def test_shared_filter_parameter(tmp_path):
    lc_obj  = make_lc_obj(tmp_path)
    lc_obj.add_custom_LC_function(func=offset_model, x="time", op_func=add_op, verbose=False,
                                  func_args=dict(A=0.001, B=[(-0.01,0.002,0.01), (0.003,0.001)]))
    fit_obj = fit_setup(verbose=False)
    CALLS.clear()
    assert run_fit(lc_obj=lc_obj, fit_obj=fit_obj, out_folder=str(tmp_path/"out"), init_only=True,
                   shared_params={"V_B":["R_B"]}, verbose=False)
    assert {c[2] for c in CALLS} == {0.002}   # R lc now uses V_B
