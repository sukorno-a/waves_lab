import numpy as np
import scipy as sp
import seaborn as sns
import math
import uncertainties
from uncertainties import ufloat
import matplotlib.pyplot as plt
import fourier as f
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

x_4min, y_4min = np.loadtxt("PART2_Thermal_Waves/data_sets/thermal_4min_a.txt", unpack=True, skiprows=3)
x_4min = x_4min / 10

# plt.figure()
# plt.plot(x_4min, y_4min)
# plt.xlabel("Time/s")
# plt.ylabel("Temperature/\xb0C")
# plt.show()

square_wave = f.square(x_4min, 240, 100, 50)

def phase_dif(sig1,sig2,period):
    complex_sig1=np.fft.fft(sig1)
    complex_sig2=np.fft.fft(sig2)
    freq_1 = np.fft.fftfreq(sig1.size)
    print(len(freq_1))
    final_func = np.zeros(len(x_4min))
    print(len(complex_sig1))
    short = int(len(complex_sig1) / 100)
    print(short)

    for n in range(1, short + 1): # for each harmonic, n = 1, 2, 3 etc
        nth_amp = np.abs(complex_sig1[n])
        print(f"amplitude for n = {n} is {nth_amp}")
        nth_phase = np.angle(complex_sig1[n])
        num = n
        nth_array = []
        for x in x_4min:
            nth_array.append(nth_amp * np.sin((num * x_4min[int(x)] / period) - nth_phase))
        for i in range(len(nth_array)):
            final_func[i] += nth_array[i]
    plt.plot(x_4min, final_func)

plt.plot(x_4min, y_4min)
phi = phase_dif(y_4min, square_wave, 240)
plt.show()




# Number of sample points
N = len(x_4min)

# Sample spacing
T = 1.0     # f = 1 Hz

# Create a signal
x = np.linspace(0.0, N*T, N)
y = y_4min
yf = np.fft.fft(y) # to normalize use norm='ortho' as an additional argument

# Where is a 200 Hz frequency in the results?
freq = np.fft.fftfreq(x.size, d=T)

# Get magnitude and phase
magnitude = np.abs(yf)
phase = np.angle(yf)
print("Magnitude:", magnitude, ", phase:", phase)

# Plot a spectrum 
plt.plot(freq[0:N//2], 2/N*np.abs(yf[0:N//2]), label='amplitude spectrum')   # in a conventional form
plt.plot(freq[0:N//2], np.angle(yf[0:N//2]), label='phase spectrum')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot([1,4,6,8], [0.19,0.2,0.3,0.59],label="Dtf, n=1")
plt.plot([1,4,6,8], [0.046,0.04,0.05,0.08],label="Dtf, n=2")
plt.plot([1,4,6,8], [0.062,0.08,0.02,0.03],label="Dtf, n=3")
plt.plot([1,4,6,8], [3.67,0.08,0.04,0.04],label="Dpl, n=1")
plt.plot([1,4,6,8], [2.09,2.09,0.2,0.05],label="Dpl, n=2")
plt.plot([1,4,6,8], [1.29,1.29,2.24,0.18],label="Dpl, n=3")
values = [0.19,0.2,0.3,0.59,0.046,0.04,0.05,0.08,0.062,0.08,0.02,0.03]
print("mean:",np.mean(values),np.std(values))
plt.xlabel("Period (min)")
plt.ylabel("Thermal Diffusivity (mm/s)")
plt.legend(loc="upper right")
plt.show()







# Create 'x', the vector of measured values.
x = y_4min

# Compute the Fourier transform of x.
f = fft(x)
n = 9600
num_samples = 9600

# Suppose the std. dev. of the 'x' measurements increases linearly
# from 0.01 to 0.5:
sigma = np.linspace(1, 1, n)

# Generate 'num_samples' arrays of the form 'x + noise', where the standard
# deviation of the noise for each coefficient in 'x' is given by 'sigma'.
xn = x + sigma*np.random.randn(num_samples, n)

fn = fft(xn, axis=-1)

print("Sum of input variances: %8.5f" % (sigma**2).sum())
print()
print("Variances of Fourier coefficients:")
np.set_printoptions(precision=5)
print(fn.var(axis=0))

# Plot the Fourier coefficient of the first 800 arrays.
num_plot = min(num_samples, 800)
fnf = fn[:num_plot].ravel()
clr = "#4080FF"
plt.plot(fnf.real, fnf.imag, 'o', color=clr, mec=clr, ms=1, alpha=0.3)
plt.plot(f.real, f.imag, 'kD', ms=4)
plt.grid(True)
plt.axis('equal')
plt.title("Fourier Coefficients")
plt.xlabel("$\Re(X_k)$")
plt.ylabel("$\Im(X_k)$")
plt.show()