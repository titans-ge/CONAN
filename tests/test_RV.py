from CONAN3.models import RadialVelocity_Model
import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(-5,5,300)
T0 = 0
per = 2
K   = 3 #m/s

RV = RadialVelocity_Model(time, [T0],[per],[K],[0],[0])

plt.plot(time, RV)
plt.axhline(0,ls="--")