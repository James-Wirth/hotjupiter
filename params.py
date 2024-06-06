import numpy as np
import os

# mass (M_solar) of host, secondary and perturber
# (m3 is initialised to 1, but this is overridden in the Monte Carlo
# experiment where we have randomised perturbing masses)

m1 = 1
m2 = 1e-9
m3 = 1

# initial eccentricity and semi-major axis (au)
e_init = 0.9
a_init = 5

# xi := ratio of initial force to force at pericenter
xi = 0.01

# number of initial phases averaged over
init_phases = 20

# (don't change)
m12 = m1 + m2

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Monte Carlo

# number of realisations of system in Monte Carlo 
num_systems = 1000

# maximum impact parameter (au)
b_max = 1000

# stellar density (in units of 10^6 per cubic parsec)
n_tot = 1

# velocity dispersion (in units of 10 km/s)
sigma_v = 1

# minimum, maximum and break mass for IMF (M_solar)
m_min = 0.08
m_max = 50
m_br = 0.8

perts_per_Myr = 1204*n_tot*((a_init/5)**2)*sigma_v

# ---------------------------------------------------------------
# ---------------------------------------------------------------

# File paths

rand_diff_params_path = os.path.expanduser("~/Documents/data/rand_diff_params.h5")
diff_data_path = os.path.expanduser("~/Documents/data/diff_data.csv")
diff_plot_path = os.path.expanduser("~/Documents/data/diff_plot.eps")
