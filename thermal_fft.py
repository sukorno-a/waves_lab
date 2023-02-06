import numpy as np
import scipy as sp
import seaborn as sns
import math
import uncertainties
from uncertainties import ufloat
import matplotlib.pyplot as plt

t=np.linspace(0,4*np.pi,1000)



def phase_dif(sig1,sig2):
    complex_sig1=np.fft.fft(sig1)
    complex_sig2=np.fft.fft(sig2)
    print(complex_sig1[0])
    plt.plot
    phase_dif= np.angle(complex_sig1)-np.angle(sig_2)
    return phase_dif

sig_1=np.sin(t-np.pi)
sig_2= np.sin(t)

phi = phase_dif(sig_1,sig_2)
# print(len(phi))

