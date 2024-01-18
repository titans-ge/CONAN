from CONAN3.RVmodel_v3 import get_RVmod
import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(-5,5,300)
T0 = 0
per = 2
K   = 3 #m/s

RV = get_RVmod(time, [T0],[per],[K],[0],[0], planet_only=True)

plt.plot(time, RV)
plt.axhline(0,ls="--")