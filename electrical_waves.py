import uncertainties
from uncertainties import ufloat

# task 3.5b - calculate Vphase and its uncertainty using values of L = 330 + 20% μH and C = 0.015 + 10% μF/section

def v_phase(L, C):
    return 1 / (L * C)**0.5

L = ufloat(330e-6, 66e-6)
C = ufloat(0.015e-6, 0.0015e-6)
sec_per_sec = v_phase(L, C)
sec_per_micros = (sec_per_sec)*1e-6
print(f"Phase velocity is {v_phase(L, C)} sections per second, or {sec_per_micros} sections per microsecond")



