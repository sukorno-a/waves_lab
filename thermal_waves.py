import numpy as np
import matplotlib.pyplot as plt
import fourier as f
import seaborn as sns
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import * 

sns.set_style("darkgrid")

x_4min_a, y_4min_a = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_a.txt", unpack=True, skiprows=3)
x_4min_b, y_4min_b = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_b.txt", unpack=True, skiprows=3)
x_4min_a, x_4min_b = x_4min_a / 10, x_4min_b / 10

square_wave = f.square(x_4min_a, 2400, 100, 50)
fundamental = f.fourier_square(x_4min_a, 1, 240, 100)

# plt.plot(x_4min_a, y_4min_a, label="Data A")
# plt.plot(x_4min_a, square_wave, label="Square")
# plt.plot(x_4min_a, fundamental, label="Fundamental")
# plt.xlabel("Time/s")
# plt.ylabel("Temperature/K")
# plt.legend()
# plt.show()

# plt.plot(x_4min_b, y_4min_b, label="Data B")
# plt.plot(x_4min_b, square_wave, label="Square")
# plt.plot(x_4min_a, fundamental, label="Fundamental")
# plt.xlabel("Time/s")
# plt.ylabel("Temperature/K")
# plt.legend()
# plt.show()

y_4min_b_stable = y_4min_b[4000:]
y_4min_b_amplitude = max(y_4min_b_stable) - ((max(y_4min_b_stable) - min(y_4min_b_stable)) / 2)
t_factor_a = ufloat((max(y_4min_a) - 50), 0.5) / (max(fundamental) - 50)
t_factor_b = ufloat((y_4min_b_amplitude - 50), 0.5) / (max(fundamental) - 50)

print(f"Transmission factor for set A: {t_factor_a}")
print(f"Transmission factor for set B: {t_factor_b}")

y_a_spliced = y_4min_a[:2400]
x_y_a_max = np.argmax(y_a_spliced) / 10
print(f"x-value of dataset A peak = {x_y_a_max}")

y_b_spliced = y_4min_b[:2400]
x_y_b_max = np.argmax(y_b_spliced) / 10
print(f"x-value of dataset B peak = {x_y_b_max}")

fundamental_spliced = fundamental[:2400]
x_fundamental_max = np.argmax(fundamental_spliced) / 10
print(f"x-value of fundamental peak = {x_fundamental_max}")
phase_lag_a = (x_y_a_max - x_fundamental_max) * np.pi / 120
phase_lag_b = (x_y_b_max - x_fundamental_max) * np.pi / 120
print(f"Phase lag for dataset A: {phase_lag_a}")
print(f"Phase lag for dataset B: {phase_lag_b}")

delta_r = ufloat(7.75, 0.06)
omega = 1 / 240
diffusivity_a = (omega * (delta_r**2)) / (2 * ((log(t_factor_a))**2))
print(f"Thermal diffusivity from data set A = {diffusivity_a}")

diffusivity_b = (omega * (delta_r**2)) / (2 * ((log(t_factor_b))**2))
print(f"Thermal diffusivity from data set B = {diffusivity_b}")