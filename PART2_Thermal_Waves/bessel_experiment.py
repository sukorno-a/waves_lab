from BESSEL import fit_bessel
#
# Then you can use it like this:
    
trans_data = [0.04, 0.04]
trans_periods = [70, 100]
phase_data = [5]
phase_periods = [70]

D_trans, D_phase = fit_bessel(trans_data, trans_periods, phase_data, phase_periods)

print(f"D_trans={D_trans}, D_phase={D_phase}")