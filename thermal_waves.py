import numpy as np
import matplotlib.pyplot as plt
import fourier as f
import seaborn as sns

sns.set_style("darkgrid")

x_4min_a, y_4min_a = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_a.txt", unpack=True, skiprows=3)
x_4min_b, y_4min_b = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_b.txt", unpack=True, skiprows=3)
x_4min_a, x_4min_b = x_4min_a / 10, x_4min_b / 10

square_wave = f.square(x_4min_a, 2400, 100, 50)
fundamental = f.fourier_square(x_4min_a, 1, 2400, 100)

plt.plot(x_4min_a, y_4min_a, label="Dataset A")
plt.plot(x_4min_a, square_wave, label="Square")
plt.plot(x_4min_a, fundamental, label="Fundamental")
plt.xlabel("Time/s")
plt.ylabel("Temperature/K")
plt.legend()
plt.show()

plt.plot(x_4min_b, y_4min_b, label="Dataset B")
plt.plot(x_4min_b, square_wave, label="Square")
plt.plot(x_4min_a, fundamental, label="Fundamental")
plt.xlabel("Time/s")
plt.ylabel("Temperature/K")
plt.legend()
plt.show()

