import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

sn.set_style("darkgrid")

def square(x, n, period, amp):
    """Function to plot square wave
    
    Parameters
    ----------
    x : array
        For each x, plot its square wave output
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

x = np.linspace(0, 240, 1000)
period = 240
amplitude = 100
N = 5 # number of terms, must be odd; change based on desired precision of Fourier series
N_array = np.linspace(1, N, int((N + 1) / 2)) # create array of odd numbers from 1 to N

plt.figure()
for N in N_array:
    plt.plot(x, square(x, N, period, amplitude), label = int(N))
    plt.legend(title = "Terms:")
plt.xlabel("t")
plt.ylabel("T(t)")
plt.show()

# with enough terms (N ~ 101), the plot closely resembles a square wave as we would expect from Fourier theory
# this shows that the Fourier series for this square wave is correct
# the Gibb's phenomena at the boundaries occur due to x being undefined at the edge of the waves