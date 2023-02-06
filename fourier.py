"""Script to plot the Fourier series of a square wave using input parameters period, amplitude and number of terms"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.signal as signal

sn.set_style("darkgrid")

def square(x, period, amplitude, offset = 50):
    """Function to plot square wave
        
    Parameters
    ----------
    x : array
        For each x, plot its square wave output
    period : float
        Period of time over which the wave oscillates
    amp : float
        Amplitude of square wave
    offset : float
        Give an offset on the y-axis
        
    Returns
    -------
    y : array
        Output of square function"""

    counter = 0
    y = []
    for i in x:
        if counter < (period / 2):
            y.append(offset + (amplitude / 2))
            counter += 1
        elif counter >= (period / 2) and counter < period:
            y.append(offset - (amplitude / 2))
            counter += 1
        elif counter == period:
            counter = 0
            y.append(offset - (amplitude / 2))

    return y

# x = np.linspace(0, 1000, 1000)
# y = square(x, 240, 100)
# plt.plot(x, y)
# plt.show()

def fourier_square(x, n, period, amp):
    """Function to plot fourier series of a square wave with N terms
    
    Parameters
    ----------
    x : array
        For each x, plot its square wave fourier series output
    n : int
        Upper limit for n; must be odd
    period : float
        Period of time over which the wave oscillates
    amp : float
        Amplitude of square wave
        
    Returns
    -------
    T : float
        Output of function"""
    
    # create array of n terms to sum together up to the defined upper limit N
    n_array = np.linspace(1, n, int((n + 1) / 2))
    L = period / 2
    a_0 = amp / 2

    T = 0
    for n in n_array:
        b_n = (2 * amp) / (np.pi * n)
        w_n = (np.pi * n) / L
        term = b_n * np.sin(w_n * x)
        T += term
    return T + a_0

def fourier_square_term(x, n, period, amp):
    """Function to plot fourier series of a square wave with N terms
    
    Parameters
    ----------
    x : array
        For each x, plot its square wave fourier series output
    n : int
        Number of term, must be odd (1, 3, 5 etc)
    period : float
        Period of time over which the wave oscillates
    amp : float
        Amplitude of square wave
        
    Returns
    -------
    T : float
        Output of function"""

    L = period / 2
    a_0 = amp / 2

    b_n = (2 * amp) / (np.pi * n)
    w_n = (np.pi * n) / L
    term = b_n * np.sin(w_n * x)
    T = term

    return T + a_0



# x = np.linspace(0, 1000, 1000)
# period = 240
# amplitude = 100
# N = 33 # number of terms, must be odd; change based on desired precision of Fourier series
# N_array = np.linspace(1, N, int((N + 1) / 2)) # create array of odd numbers from 1 to N

# plt.figure()
# for N in N_array:
#     plt.plot(x, fourier_square(x, N, period, amplitude), label = int(N))
#     plt.legend(title = "Terms:")
# plt.xlabel("t")
# plt.ylabel("T(t)")
# plt.show()

# # with enough terms (N ~ 101), the plot closely resembles a square wave as we would expect from Fourier theory
# # this shows that the Fourier series for this square wave is correct
# # the Gibb's phenomena at the boundaries occur due to x being undefined at the edge of the waves

# x = np.linspace(0, 500, 500)
# N = 9 
# N_array = np.linspace(1, N, int((N + 1) / 2))

# plt.figure()
# for N in N_array:
#     plt.plot(x, fourier_square(x, N, period, amplitude), label = int(N))
#     plt.legend(title = "Terms:")
# plt.plot(x, square(x, 240, 100, 50), label = "Square")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel("T(t)")
# plt.show()