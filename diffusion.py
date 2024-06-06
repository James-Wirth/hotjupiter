import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
import math
import random
import h5py

import params as params
import model as model

def total_perts(time_in_Myr):
    return int(params.perts_per_Myr*time_in_Myr)

#--------------------------------------------
# generate random encounter parameters for Monte Carlo runs

def rand_m3():
    y = random.random()
    a = 1.8/(4*(params.m_br**0.6) - 3*(params.m_min**0.6) - (params.m_br**2.4)*(params.m_max)**(-1.8))
    b = a*(params.m_br**2.4)

    y_crit = (a/0.6)*((params.m_br**0.6)-(params.m_min**0.6))

    if(y <= y_crit):
        return ((0.6*y)/a + (params.m_min**0.6))**(1/0.6)
    else:
        return ((params.m_br**-1.8)+(1.8/b)*(y_crit-y))**(-1/1.8)

def rand_b():
    return np.sqrt(random.random()*(params.b_max)**2)
 
def rand_v_infty():
    y = random.random()
    cdf = lambda x: math.erf(x/(np.sqrt(2)*params.sigma_v)) - ((np.sqrt(2)*x)/(np.sqrt(np.pi)*params.sigma_v))*np.exp(-x**2/(2*params.sigma_v**2))
    return fsolve(lambda x: cdf(x)-y, np.sqrt(2))[0]
    
def rand_i():
    return np.arccos(1-2*random.random())

def rand_2pi():
    return random.random()*2*np.pi

def generate_rand_params(num_perts):

    rand_params = []
    for system in range(0,params.num_systems):

        print("Generating random encounter parameters: " + str(math.ceil(100*(system+1)/params.num_systems)) + "%", end='\r')
        sys.stdout.flush()
        
        for pert in range(0, num_perts):
            rand_params.append([rand_m3(), rand_v_infty(), rand_b(), rand_2pi(), rand_i(), rand_2pi()])

    df = pd.DataFrame(rand_params)
    df.to_hdf(params.rand_diff_params_path, key='rand', mode='w')
    print("\nSUCCESS: random encounter parameters saved to " + params.rand_diff_params_path)

#--------------------------------------------
# diffuse the eccentricity for num_perts total encounters

def diffuse(num_perts):

    rand_params = pd.read_hdf(params.rand_diff_params_path, 'rand')

    # check that we have enough randonm encounter parameters 
    if(num_perts*params.num_systems <= rand_params.shape[0]):

        # store the current eccentricity, semi-major axis for each num_perts and each system realisation
        e_values = np.empty([num_perts, params.num_systems])
        a_values = np.empty([num_perts, params.num_systems])

        # for each system realisation, 0 = bound, 1 = unbound
        unbound = np.zeros(params.num_systems)

        for system in range(0,params.num_systems):

            print("Calculating eccentricity diffusion: " + str(math.ceil(100*(system+1)/params.num_systems)) + "%", end='\r')
            sys.stdout.flush()

            e = params.e_init
            a = params.a_init
    
            for pert in range(0, num_perts):

                e_values[pert, system] = e
                a_values[pert, system] = a
                rand_index = num_perts * system + pert

                if(unbound[system] == 0):
                    
                    # read in the next row of random encounter parameters

                    keys = {"m3":0,"v_infty":1,"b":2,"Omega":3,"inc":4,"omega":5}
                    prms = [rand_params.iat[rand_index,i] for i in range(6)]

                    # update eccentricity and semi-major axis
                    if(model.tidal_param(prms[keys['v_infty']], prms[keys['b']], a) > 5 and model.slow_param(prms[keys['v_infty']], prms[keys['b']], a) > 5):
                        de = model.de_HR(prms[keys['v_infty']], prms[keys['b']], prms[keys['Omega']], prms[keys['inc']], prms[keys['omega']], e, a, prms[keys['m3']])
                        e = e + de
                    else:
                        de = model.de_SIM(prms[keys['v_infty']], prms[keys['b']], prms[keys['Omega']], prms[keys['inc']], prms[keys['omega']], e, a, prms[keys['m3']])
                        e = e + de
                    
                    if(e > 1):
                        unbound[system] = 1

        df = pd.DataFrame(e_values)
        df.to_csv(params.diff_data_path)
        print("\nSUCCESS: diffusion data saved to " + params.diff_data_path)
        
    else:
        print("ERROR: " + params.rand_diff_params_path + " does not contain enough rows to generate " + str(num_perts) + " perturbations")

#--------------------------------------------
# plot the eccentricity diffusion at snapshot values of num_perts held in the array pert_sample

def plot_diffusion(pert_sample):
    df = pd.read_csv(params.diff_data_path)
    df_numpy = df.drop(df.columns[0], axis=1).to_numpy()

    fig, axs = plt.subplots(1, len(pert_sample), sharex = 'all', sharey='all')
    for j in range(0,3):
        axs[j].hist(df_numpy[pert_sample[j]], 50, range = (0,1), log=True, color='silver', label = "Hybrid MC", density=True)
        axs[j].set_title(f'${pert_sample[j]}$ perts')
        axs[j].title.set_size(18)
        axs[j].legend()
    
    fig.tight_layout(pad=2)
    plt.xlim(0,1)
    plt.ylim(10**-2,1.3e2)
    fig.supxlabel('$e$')
    fig.supylabel("$p(e,t)$")
    fig.set_size_inches(12, 6)
    fig.savefig(params.diff_plot_path, format='eps')
    plt.show()

#--------------------------------------------

# generate randomised encounter data for num_perts total encounters
# N.B. only need to run this once, unless you want to increase num_perts again!

# generate_rand_params(total_perts(10))

# diffuse for num_perts total encounters
diffuse(total_perts(10))
plot_diffusion([0,int(total_perts(10)/2),total_perts(10)-1])

