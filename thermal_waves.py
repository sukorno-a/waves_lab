import numpy as np
import matplotlib.pyplot as plt
import fourier as f
import seaborn as sns
import math as math
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import * 

sns.set_style("darkgrid")
    
# task 2.3
# (a)
x_4min_a, y_4min_a = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_a.txt", unpack=True, skiprows=3)
x_4min_b, y_4min_b = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_b.txt", unpack=True, skiprows=3)
x_4min_a, x_4min_b = x_4min_a / 10, x_4min_b / 10

square_wave = f.square(x_4min_a, 2400, 100, 50)
fundamental = f.fourier_square(x_4min_a, 1, 240, 100)

# plt.plot(x_4min_a, y_4min_a, label="Data A")
# plt.plot(x_4min_a, square_wave, label="Square")
# plt.plot(x_4min_a, fundamental, label="Fundamental")
# plt.xlabel("Time/s")
# plt.ylabel("Temperature/\xb0C")
# plt.legend()
# plt.show()

# plt.plot(x_4min_b, y_4min_b, label="Data B")
# plt.plot(x_4min_b, square_wave, label="Square")
# plt.plot(x_4min_a, fundamental, label="Fundamental")
# plt.xlabel("Time/s")
# plt.ylabel("Temperature/\xb0C")
# plt.legend()
# plt.show()

# (b)
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
transmission_diffusivity_a = (omega * (delta_r**2)) / (2 * ((log(t_factor_a, math.e))**2))
print(f"Thermal diffusivity from data set A (eq. 2.3) = {transmission_diffusivity_a} mm\s")
transmission_diffusivity_b = (omega * (delta_r**2)) / (2 * ((log(t_factor_b))**2))
print(f"Thermal diffusivity from data set B (eq. 2.3)= {transmission_diffusivity_b} mm\s")

phase_diffusivity_a = (omega * (delta_r**2)) / (2 * (phase_lag_a)**2)
print(f"Thermal diffusivity from data set A (eq. 2.4) = {phase_diffusivity_a} mm\s")
phase_diffusivity_b = (omega * (delta_r**2)) / (2 * (phase_lag_b)**2)
print(f"Thermal diffusivity from data set B (eq. 2.4) = {phase_diffusivity_b} mm\s")

# optimised the above code by implementing OOP techniques:

class thermal:
    """Various functions required for the "Thermal Waves" section
    
    Methods
    -------
    plot_square
        Plots the square wave for the corresponding period of the dataset
    plot_fundamental
        Plots the fundamental mode of the square wave corresponding to the data
    plot
        Plots a dataset
    transmission
        Calculates the transmission coefficient between a dataset and the fundamental mode
    diffusivity_approx
        Estimates a value for thermal diffusivity using the transmission coefficient
    """
    def __init__(self, file_name) -> None:
        self.file = file_name
        self.x, self.y = np.loadtxt(self.file, unpack=True, skiprows=3)
        self.x = self.x / 10
        pass

    def plot_square(self, period, amplitude = 100, offset = 50):
        square_wave = f.square(self.x, period * 10, amplitude, offset)
        plt.plot(self.x, square_wave, label="Square")

    def plot_fundamental(self, period, amplitude):
        """
        n : int
            Odd integer, number of terms to sum when plotting the fourier coefficients
        """
        global fundamental 
        fundamental = f.fourier_square(self.x, 1, period, amplitude)
        plt.plot(self.x, fundamental, label="Fundamental")

    def plot(self):
        plt.plot(self.x, self.y, label="Data") # label = "Period:" + self.file[-10:-6] + "s"
        plt.xlabel("Time/s")
        plt.ylabel("Temperature/\xb0C")

    def transmission(self):
        y_amplitude = (max(self.y) - min(self.y)) / 2
        t_factor = ufloat(y_amplitude, 0.5) / (max(fundamental) - 50)
        return t_factor
    
    def diffusivity_approx(self, period):
        omega = 1 / period
        delta_r = ufloat(7.75, 0.06)
        t_factor = self.transmission()
        diffusivity = (omega * (delta_r**2)) / (2 * ((log(t_factor, math.e))**2))
        return diffusivity

# task 2.4
data = ["1min_a", "1min_b", "2min_a", "2min_b", "4min_a", "4min_b", "6min", "8min", "16min"]
# for i in data:
#     if i[1].isdigit():
#         period = int(i[0:2]) * 60
#     else:
#         period = int(i[0]) * 60
#     dataset = thermal("PART2_Thermal_Waves/data_sets/thermal_" + i + ".txt")
#     plt.title(f"Period: {str(period)}s")
#     dataset.plot()
#     dataset.plot_fundamental(period, 100)
#     dataset.plot_square(period)
#     plt.legend(loc = "upper right")
#     plt.show()

# task 2.5
data = ["1min_a", "2min_a", "4min_a", "6min", "8min", "16min"]
list_of_diffusivity = []
periods = []
for i in data:
    if i[1].isdigit():
        period = int(i[0:2]) * 60
    else:
        period = int(i[0]) * 60
    periods.append(period / 60)
    dataset = thermal("PART2_Thermal_Waves/data_sets/thermal_" + i + ".txt")
    print(f"Thermal diffusivity approximation for period {period}s: {dataset.diffusivity_approx(period)}")
    list_of_diffusivity.append(dataset.diffusivity_approx(period))

plt.plot(periods, list_of_diffusivity.nominal_value)
plt.show()

