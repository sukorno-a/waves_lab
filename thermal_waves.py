import numpy as np
import matplotlib.pyplot as plt
import fourier as f
import num_integration as numint
import seaborn as sns
import math as math
import uncertainties
import re
from uncertainties import ufloat
from uncertainties.umath import * 
import scipy
from scipy import integrate

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
y_a_amp = max(y_4min_a) - 50
print(y_a_amp)
t_factor_a = ufloat(y_a_amp, 0.5) / (max(fundamental) - 50)
t_factor_b = ufloat((y_4min_b_amplitude - 50), 0.5) / (max(fundamental) - 50)
print(f"Transmission factor for set A: {t_factor_a}")
# print(f"Transmission factor for set B: {t_factor_b}")

# y_a_spliced = y_4min_a[:2400]
# x_y_a_max = np.argmax(y_a_spliced) / 10
# print(f"x-value of dataset A peak = {x_y_a_max}")
# y_b_spliced = y_4min_b[:2400]
# x_y_b_max = np.argmax(y_b_spliced) / 10
# print(f"x-value of dataset B peak = {x_y_b_max}")

# fundamental_spliced = fundamental[:2400]
# x_fundamental_max = np.argmax(fundamental_spliced) / 10
# print(f"x-value of fundamental peak = {x_fundamental_max}")
# phase_lag_a = (x_y_a_max - x_fundamental_max) * np.pi / 120
# phase_lag_b = (x_y_b_max - x_fundamental_max) * np.pi / 120
# print(f"Phase lag for dataset A: {phase_lag_a}")
# print(f"Phase lag for dataset B: {phase_lag_b}")

delta_r = ufloat(7.75, 0.06)
omega = (2 * np.pi) / 240
transmission_diffusivity_a = (omega * (delta_r**2)) / (2 * ((log(t_factor_a, math.e))**2))
print(f"Thermal diffusivity from data set A (eq. 2.3) = {transmission_diffusivity_a} mm\s")
# transmission_diffusivity_b = (omega * (delta_r**2)) / (2 * ((log(t_factor_b))**2))
# print(f"Thermal diffusivity from data set B (eq. 2.3)= {transmission_diffusivity_b} mm\s")

# phase_diffusivity_a = (omega * (delta_r**2)) / (2 * (phase_lag_a)**2)
# print(f"Thermal diffusivity from data set A (eq. 2.4) = {phase_diffusivity_a} mm\s")
# phase_diffusivity_b = (omega * (delta_r**2)) / (2 * (phase_lag_b)**2)
# print(f"Thermal diffusivity from data set B (eq. 2.4) = {phase_diffusivity_b} mm\s")

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

    def plot_fundamental(self, period, amplitude, n):
        """
        n : int
            Odd integer, number of terms to sum when plotting the fourier coefficients
        """
        global fundamental 
        fundamental = f.fourier_square(self.x, 1, period, amplitude)
        global square_harmonic_2
        square_harmonic_2 = f.fourier_square(self.x, 3, period, amplitude)
        global square_harmonic_3
        square_harmonic_3 = f.fourier_square(self.x, 5, period, amplitude)
        global square_harmonics
        square_harmonics = [fundamental, square_harmonic_2, square_harmonic_3]
        # plt.plot(self.x, fundamental, label="Fundamental")
        return square_harmonics[n]

    def plot(self):
        plt.plot(self.x, self.y, label="Data") # label = "Period:" + self.file[-10:-6] + "s"
        plt.xlabel("Time/s")
        plt.ylabel("Temperature/\xb0C")

    def transmission(self):
        y_amplitude = (max(self.y) - min(self.y)) / 2
        t_factor = ufloat(y_amplitude, 0.5) / (max(fundamental) - 50)
        return t_factor
    
    def diffusivity_approx(self, period):
        omega = 2 * np.pi / period
        delta_r = ufloat(7.75, 0.06)
        t_factor = self.transmission()
        diffusivity = (omega * (delta_r**2)) / (2 * ((log(t_factor, math.e))**2))
        return diffusivity

    # task 2.6a - numerical integration to find Fourier series of 4-min dataset

    def fourier_coefficients(self, period, n = 3):
        function = self.y
        x_array = self.x
        a_n = []
        b_n = []
        for num in range(1, n + 1):
            a_n_integrand = []
            b_n_integrand = []
            for i in range(len(x_array)):
                cos_term = function[i] * np.cos((2 * np.pi * x_array[i] * num) / period)
                a_n_integrand.append(cos_term)
                sin_term = function[i] * np.sin((2 * np.pi * x_array[i] * num) / period)
                b_n_integrand.append(sin_term)
            a_n.append(numint.rect(x_array, a_n_integrand) * (2 / period))
            b_n.append(numint.rect(x_array, b_n_integrand) * (2 / period))
        print(f"for period = {period}, the ans are {a_n} and the bns are {b_n}")
        return a_n, b_n

    def amplitude_phase(self, a_n, b_n):
        amplitudes = []
        phase_lags = []
        for i in range(len(a_n)):
            amplitudes.append( ( (a_n[i] ** 2) + (b_n[i] ** 2) ) ** 0.5)
            phase_lags.append(-1 * np.arctan2(a_n[i], b_n[i]))
        print(f"the amplitudes are {amplitudes} and the phase lags are {phase_lags}")
        return amplitudes, phase_lags

    # task 2.6b - use the amplitude-phase form per harmonic to find values of thermal diffusivity

    def d_transmission(self, amplitude, period, n):
        w_n = (2 * np.pi * n) / period
        square_harmonic = f.fourier_square_term(self.x, (2 * n) - 1, period, 100)
        square_harmonic_amp = (max(square_harmonic) - 50)
        print(f"square harmonic amplitude: {square_harmonic_amp}")
        # plt.figure()
        # plt.plot(self.x, square_harmonic)
        # plt.show()
        transmission_factor = amplitude / square_harmonic_amp
        print(f"Transmission factor for period {period}, amplitude {amplitude} and n = {n}: {transmission_factor}")
        delta_r = ufloat(7.75, 0.06)
        diffusivity = (w_n * (delta_r**2)) / (2 * ((log(transmission_factor, math.e))**2))
        return diffusivity

    def d_phase(self, phase, period, n):
        w_n = (2 * np.pi * n) / period
        delta_r = ufloat(7.75, 0.06)
        diffusivity = (w_n * (delta_r**2)) / (2 * (phase**2))
        return diffusivity

class test(thermal):
    def __init__(self, x, y):
        self.x = x
        self.y = y
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
# data = ["1min_a", "2min_a", "4min_a", "6min", "8min", "16min"]
# list_of_diffusivity = []
# list_of_diffusivity_std = []
# periods = []
# expected = []
# for i in data:
#     m = re.match(r"\d+", i)
#     period = int(m[0]) * 60
#     periods.append(period / 60)
#     dataset = thermal("PART2_Thermal_Waves/data_sets/thermal_" + i + ".txt")
#     print(f"Thermal diffusivity approximation for period {period}s: {dataset.diffusivity_approx(period)}")
#     list_of_diffusivity.append(dataset.diffusivity_approx(period).n)
#     list_of_diffusivity_std.append(dataset.diffusivity_approx(period).s)
#     expected.append(0.124)


# plt.plot(periods, list_of_diffusivity, label="D/period")
# plt.plot(periods, expected, label="Expected D")
# plt.xlabel("Period/minutes")
# plt.ylabel("Thermal diffusivity/mms^-1")
# plt.legend()
# plt.show()



# task 2.6a

# test using a perfect sin wave with amplitude 50 to check Fourier coefficients are correct:

test_x = np.linspace(0, 2 * np.pi, 1000)
test_y = []
for i in test_x:
    test_y.append(50 * np.sin(i))
plt.plot(test_x, test_y, label = "real")
trial_data = test(test_x, test_y)
test_an, test_bn = trial_data.fourier_coefficients(2 * np.pi)
amplitudes, phase_lags = trial_data.amplitude_phase(test_an, test_bn)
plt.plot(test_x, amplitudes[0] * np.sin(test_x - phase_lags[0]), label = "thing 2")
plt.legend()
plt.show()

# test_an = []
# for i in range(len(test_y)):
#     test_an.append(test_y[i] * np.sin((2 * np.pi * test_x[i]) / (2 * np.pi)))
# print(integrate.simps(test_an, test_x))
# now apply this to our 4-min dataset:

# x_4min_a, y_4min_a = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_a.txt", unpack=True, skiprows=3)
# x_4min_a = x_4min_a / 10
# an, bn = fourier_coefficients(x_4min_a, y_4min_a, 240)
# print(f"The an coefficients are {an} and the bn coefficiets are {bn}")
# amplitudes, phase_lags = amplitude_phase(an, bn)
# print(f"The amplitudes are {amplitudes} and the phase lags are {phase_lags}")

# task 2.6b

# for i in range(len(amplitudes)):
#     print(f"The thermal diffusivity for n = {i + 1} using transmission factor: {d_transmission(amplitudes[i], 240, i + 1)}")
#     print(f"The thermal diffusivity for n = {i + 1} using phase lag: {d_phase(phase_lags[i], 240, i + 1)}")

# task 2.7a - find Dtf and Dpl for each of the three harmonics for each dataset

data = ["1min_a", "2min_a", "4min_a", "6min", "8min", "16min"]
periods = []
dtf_n_1 = []
dtf_n_2 = []
dtf_n_3 = []
dpl_n_1 = []
dpl_n_2 = []
dpl_n_3 = []
dtf_n_1_std = []
dtf_n_2_std = []
dtf_n_3_std = []
dpl_n_1_std = []
dpl_n_2_std = []
dpl_n_3_std = []

from uncertainties import unumpy, ufloat
import numpy as np
arr = []

# 3.0+/-1.2
for i in data:
    m = re.match(r"\d+", i)
    period = int(m[0]) * 60
    periods.append(period / 60)
    
    # initialise instance of class object
    dataset = thermal("PART2_Thermal_Waves/data_sets/thermal_" + i + ".txt")

    # plot the original dataset
    # plt.figure()
    # dataset.plot()

    print(f"\nPeriod: {period}\n")

    # find amplitude-phase form of each harmonic
    amplitudes, phase_lags = dataset.amplitude_phase(*dataset.fourier_coefficients(period))

    # assign dtf and dpl arrays with their errors
    list_of_dtf = []
    list_of_dtf_std = []
    list_of_dpl = []
    list_of_dpl_std = []
    

    # for each harmonic, give the Dtf and Dpl
    for i in range(len(amplitudes)):
        d_tf = dataset.d_transmission(amplitudes[i], period, i + 1)
        d_pl = dataset.d_phase(phase_lags[i], period, i + 1)
        arr.append(d_tf)
        print(f"D_tf for n = {i + 1}: {d_tf}")
        print(f"D_pl n = {i + 1}: {d_pl}")
        
        list_of_dtf.append(d_tf.n)
        list_of_dtf_std.append(d_tf.s)
        list_of_dpl.append(d_pl.n)
        list_of_dpl_std.append(d_pl.s)

    dtf_n_1.append(list_of_dtf[0])
    dtf_n_2.append(list_of_dtf[1])
    dtf_n_3.append(list_of_dtf[2])
    dpl_n_1.append(list_of_dpl[0])
    dpl_n_2.append(list_of_dpl[1])
    dpl_n_3.append(list_of_dpl[2])
    dtf_n_1_std.append(list_of_dtf_std[0])
    dtf_n_2_std.append(list_of_dtf_std[1])
    dtf_n_3_std.append(list_of_dtf_std[2])
    dpl_n_1_std.append(list_of_dpl_std[0])
    dpl_n_2_std.append(list_of_dpl_std[1])
    dpl_n_3_std.append(list_of_dpl_std[2])

    
plt.figure()
plt.errorbar(periods, dtf_n_1, yerr = dtf_n_1_std, capsize = 3, label = "Num. Integration",color = "cadetblue")
plt.errorbar(periods, dtf_n_2, yerr = dtf_n_2_std, capsize = 3,color = "cadetblue")
plt.errorbar(periods, dtf_n_3, yerr = dtf_n_3_std, capsize = 3,color = "cadetblue")
plt.errorbar(periods, dpl_n_1, yerr = dpl_n_1_std, capsize = 3,color = "cadetblue")
plt.errorbar(periods, dpl_n_2, yerr = dpl_n_2_std, capsize = 3,color = "cadetblue")
plt.errorbar(periods, dpl_n_3, yerr = dpl_n_3_std, capsize = 3,color = "cadetblue")
plt.plot(periods, [0.124,0.124,0.124,0.124,0.124,0.124], label = "Expected 0.124",linestyle='dashed',color="orange")
plt.plot([1,2,4,6,8,16],[0.13292848703783777, 0.0704956598377271, 0.07448121366891888, 0.10821944900160108, 0.28231027205849724, 0.06755006339216664],label="BESSEL",linestyle="dashdot",color="red")
# plt.plot([1,4,6,8], [0.19,0.2,0.3,0.59],label="FFT",linestyle="dotted",color = "purple")
# plt.plot([1,4,6,8], [0.046,0.04,0.05,0.08],linestyle="dotted",color = "purple")
# plt.plot([1,4,6,8], [0.062,0.08,0.02,0.03],linestyle="dotted",color = "purple")
# plt.plot([1,4,6,8], [3.67,0.08,0.04,0.04],linestyle="dotted",color = "purple")
# plt.plot([1,4,6,8], [2.09,2.09,0.2,0.05],linestyle="dotted",color = "purple")
# plt.plot([1,4,6,8], [1.29,1.29,2.24,0.18],linestyle="dotted",color = "purple")
plt.xlabel("Period (min)")
plt.ylabel("Thermal Diffusivity (mm/s)")
plt.legend(loc="upper right")
plt.show()
arr = np.array(arr)
print("mean:",sum(arr)/len(arr))

y = []
for i in x_4min_a:
    y.append((11 * np.sin(2 * np.pi * i / 240)) + 50)
plt.plot(x_4min_a, y)
plt.plot(x_4min_a, y_4min_a)
plt.show()


# x = np.linspace(0, 1000, 1000)
# square_test = test(x, f.square(x, 240, 100, 50))
# plt.plot(x, f.square(x, 240, 100, 50))
# an, bn = square_test.fourier_coefficients(240, 1)
# array = []
# for i in x:
#     array.append(50 + (an[0] * np.cos(2 * np.pi * i / 240)) + (bn[0] * np.sin(2 * np.pi * i / 240)))
# plt.plot(x, array)
# plt.show()