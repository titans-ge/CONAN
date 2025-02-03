from CONAN.geepee import spleaf_cosine, gp_params_convert, celerite_kernels, george_kernels, spleaf_kernels, celerite_cosine, spleaf_cosine
import pytest
import numpy as np
import matplotlib.pyplot as plt

def test_kernels(show_plot=False):
    amp, len_sc = 1000, 4  #ppm, days
    t = np.linspace(0, 100, 1000)

    gp_conv = gp_params_convert()

    equiv = []

    #mat32
    # sp_mat32 = spleaf_kernels["mat32"](*gp_conv.get_values("sp_mat32", data="lc", pars = [amp, len_sc])).eval(t)
    ce_mat32 = celerite_kernels["mat32"](*gp_conv.get_values("ce_mat32", data="lc", pars = [amp, len_sc])).get_value(t)
    ge_mat32 = 1*george_kernels["mat32"](1)
    ge_mat32.set_parameter_vector(gp_conv.get_values("ge_mat52", data="lc", pars = [amp, len_sc]))
    ge_mat32 = ge_mat32.get_value(np.atleast_2d(t).T)[0]
    # plt.plot(t,ce_mat32,"--"); plt.plot(t,ge_mat32,":"); plt.axvline(len_sc); plt.show()

    equiv.append( ce_mat32 == pytest.approx(ge_mat32,abs=0.01*1e-6) )    #same within 0.01ppm

    #mat52
    # sp_mat52 = spleaf_kernels["mat52"](*gp_conv.get_values("sp_mat52", data="lc", pars = [amp, len_sc])).eval(t)
    # ge_mat52 = george_kernels["mat52"](*gp_conv.get_values("ge_mat52", data="lc", pars = [amp, len_sc])).get_value(np.atleast_2d(t).T)[0]

    #cosine
    sp_cos = spleaf_kernels["cos"](*gp_conv.get_values("sp_cos", data="lc", pars = [amp, len_sc])).eval(t)  
    ce_cos = celerite_kernels["cos"](*gp_conv.get_values("ce_cos", data="lc", pars = [amp, len_sc])).get_value(t)
    ge_cos = 1*george_kernels["cos"](1)
    ge_cos.set_parameter_vector(gp_conv.get_values("ge_cos", data="lc", pars = [amp, len_sc]))
    ge_cos = ge_cos.get_value(np.atleast_2d(t).T)[0]
    # plt.plot(t, sp_cos); plt.plot(t,ce_cos,"--"); plt.plot(t,ge_cos,":"); plt.axvline(len_sc); plt.show()

    equiv.append(  sp_cos == pytest.approx(ce_cos, abs=0.01*1e-6) and ce_cos ==  pytest.approx(ge_cos,abs=0.01*1e-6) )

    #exp
    sp_exp = spleaf_kernels["exp"](*gp_conv.get_values("sp_exp", data="lc", pars = [amp, len_sc])).eval(t)
    ce_exp = celerite_kernels["exp"](*gp_conv.get_values("ce_exp", data="lc", pars = [amp, len_sc])).get_value(t)
    ge_exp = 1*george_kernels["exp"](1)
    ge_exp.set_parameter_vector(gp_conv.get_values("ge_exp", data="lc", pars = [amp, len_sc]))
    ge_exp = ge_exp.get_value(np.atleast_2d(t).T)[0]
    # plt.plot(t, sp_exp); plt.plot(t,ce_exp,"--"); plt.plot(t,ge_exp,":"); plt.axvline(len_sc); plt.show()

    equiv.append(  sp_exp == pytest.approx(ce_exp, abs=0.01*1e-6) and ce_exp ==  pytest.approx(ge_exp,abs=0.01*1e-6) )


    # expsq
    sp_expsq = spleaf_kernels["expsq"](*gp_conv.get_values("sp_expsq", data="lc", pars = [amp, len_sc])).eval(t)
    ge_expsq = 1*george_kernels["expsq"](1)
    ge_expsq.set_parameter_vector(gp_conv.get_values("ge_expsq", data="lc", pars = [amp, len_sc]))
    ge_expsq = ge_expsq.get_value(np.atleast_2d(t).T)[0]
    # plt.plot(t, sp_expsq); plt.plot(t,ge_expsq,":"); plt.axvline(len_sc); plt.show()

    equiv.append(sp_expsq == pytest.approx(ge_expsq, abs=0.01*1e-6))

    #sho
    Q = 1/np.sqrt(2)
    sp_sho   = spleaf_kernels["sho"](*gp_conv.get_values("sp_sho", data="lc", pars = [amp, len_sc]), Q=Q).eval(t)
    log_s0, log_w0 =  gp_conv.get_values("ce_sho", data="lc", pars = [amp, len_sc], fixed_arg=Q)
    ce_sho   = celerite_kernels["sho"](log_S0=log_s0, log_omega0=log_w0, log_Q=np.log(Q)).get_value(t)
    # plt.plot(t, sp_sho); plt.plot(t,ce_sho,":"); plt.axvline(len_sc); plt.show()

    equiv.append(sp_sho == pytest.approx(ce_sho, abs=0.01*1e-6))

    assert all(equiv)

