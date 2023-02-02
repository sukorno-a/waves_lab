import uncertainties
from uncertainties import ufloat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# task 3.5b - calculate Vphase and its uncertainty using values of L = 330 + 20% μH and C = 0.015 + 10% μF/section

def v_phase(L, C):
    return 1 / (L * C)**0.5

L = ufloat(330e-6, 66e-6)
C = ufloat(0.015e-6, 0.0015e-6)
sec_per_sec = v_phase(L, C)
sec_per_micros = (sec_per_sec)*1e-6
print(f"Phase velocity is {v_phase(L, C)} sections per second, or {sec_per_micros} sections per microsecond")

# task 3.8a - plot dispersion relation of the line, w against k
data = np.loadtxt("PART3_Electrical_Waves/task_3.7/amplitude_frequency.csv", delimiter=",", skiprows = 1, usecols = [0, 2, 3, 4, 5, 6])
frequencies = data[:,0]
amp_ratio = data[:,5]

k = []
plt.figure()
for i in range(len(frequencies)):
    k.append(i + 1)
plt.plot(frequencies, k)
plt.xlabel("Frequency/kHz")
plt.ylabel("Wavenumber")
plt.show()



