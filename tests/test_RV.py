from CONAN.models import RadialVelocity_Model, Transit_Model
import radvel
import matplotlib.pyplot as plt
import numpy as np
import pytest

def radvel_model(t, pars):
    params = radvel.Parameters(1,basis='per tc secosw sesinw k') # number of planets = 2
    params['per1'] = radvel.Parameter(value=pars[1])
    params['tc1'] = radvel.Parameter(value=pars[0])
    params['secosw1'] = radvel.Parameter(value=pars[4])
    params['sesinw1'] = radvel.Parameter(value=pars[3])
    params['k1'] = radvel.Parameter(value=pars[2])

    mod = radvel.RVModel(params, time_base=t)
    return mod(t)


def test_radvel_model(show_plot=False):
    phases = np.linspace(-1,1,100)

    equiv    = []
    niter    = 4
    ecc_dist = np.append(0,np.random.uniform(0.,0.1, niter))
    w_dist   = np.append(90,np.random.uniform(0,1,niter)*360)

    for e, w in zip(ecc_dist, w_dist):
        sesinw = np.sqrt(e)*np.sin(w*np.pi/180)
        secosw = np.sqrt(e)*np.cos(w*np.pi/180)


        pars       = [0,1,28,sesinw,secosw]
        conan_rv,_ = RadialVelocity_Model(phases,*pars)
        radvel_rv = radvel_model(phases, pars)

        equiv.append( conan_rv == pytest.approx(radvel_rv,rel=1e-2) )
        print(f"{e=:.2f},{w=:5.1f}, agree: {equiv[-1]}")

        if show_plot:
            fig, ax = plt.subplots(2,1, sharex=True)
            ax[0].set_title(f"{e=:.2f},{w=:.1f}, {equiv[-1]}")
            ax[0].plot(phases, conan_rv, label="CONAN")
            ax[0].plot(phases, radvel_rv,":", label="radvel")
            ax[0].set_ylabel("RV")
            ax[0].legend()
            ax[1].plot(phases, radvel_rv-conan_rv,".-", label="radvel")
            ax[1].set_ylabel("diff")
            plt.subplots_adjust(hspace=0.03)

            if e==ecc_dist[-1]: plt.show()

    assert all(equiv)

if __name__ == "__main__":
    test_radvel_model(show_plot=False)
    