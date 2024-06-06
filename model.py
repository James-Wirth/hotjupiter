import params
import numpy as np
import rebound
from scipy.optimize import fsolve

def kepler(E, e0):
    return E - e0*np.sin(E)

# analytic excitation (Heggie & Rasio)
# if not specified, e0, a0 and m3 take default values in params.py

def de_HR(v_infty, b, Omega, inc, omega, e0 = params.e_init, a0 = params.a_init, m3 = params.m3):

    m123 = params.m12 + m3
    a_pert = -params.m12/(v_infty**2)
    e_pert = np.sqrt(1 + (b/a_pert)**2)
    rp = -a_pert * (e_pert - 1)
    max_anomaly = np.arccos(-1/e_pert)
    y = e0 * np.sqrt(1-e0**2) * (params.m3/(np.sqrt(params.m12 * m123)))
    alpha = -1 * (15/4) * ((1+e_pert)**(-3/2))
    chi = np.arccos(-1/e_pert) + np.sqrt(e_pert**2-1)
    psi = (1/3) * (((e_pert**2-1)**(3/2))/(e_pert**2))
    Theta1 = (np.sin(inc)**2) * np.sin(2 * Omega)
    Theta2 = (1 + (np.cos(inc))**2) * np.cos(2 * omega) * np.sin(2 * Omega)
    Theta3 = 2 * np.cos(inc) * np.sin(2 * omega) * np.cos(2 * Omega)

    return alpha*y*((a0/rp)**(3/2))*(Theta1*chi + (Theta2 + Theta3)*psi)

# simulated excitation (REBOUND)
# if not specified, e0, a0 and m3 take default values in params.py

def de_SIM(v_infty, b, Omega, inc, omega, e0 = params.e_init, a0 = params.a_init, m3 = params.m3):

    m123 = params.m12 + m3
    a_pert = -params.m12/(v_infty**2)
    e_pert = np.sqrt(1+(b/a_pert)**2)
    rp = -a_pert*(e_pert-1)
    max_anomaly = np.arccos(-1/e_pert)

    #calculate start point of simulation as a fraction (alpha) of the maximum true anomaly
    r_crit = rp/(params.xi**(1/3))
    theta_crit = np.arccos((1/e_pert)*(((a_pert*(1-e_pert**2))/r_crit) - 1))
    alpha = theta_crit/max_anomaly

    # generate true anomalies of initial phases (to be averaged over)
    phase_range = np.linspace(-np.pi,np.pi, num=params.init_phases, endpoint = False)
    true_anoms = np.zeros(params.init_phases)
    for i, mean_anom in enumerate(phase_range):
        ecc_anom = fsolve(lambda E: kepler(E, e0)-mean_anom, 0)[0]
        beta = e0/(1+np.sqrt(1-e0**2))
        true_anom = ecc_anom + 2*np.arctan((beta*np.sin(ecc_anom))/(1-beta*np.cos(ecc_anom)))
        true_anoms[i] = true_anom

    avg_de = 0
    for phase in true_anoms:

        sim = rebound.Simulation()
        sim.add(m=params.m1)
        sim.add(m=params.m2, a=a0, e=e0, f=phase)
        sim.add(m=m3, a=a_pert, e=e_pert, f=-alpha*max_anomaly, Omega=Omega, inc=inc, omega=omega)
        sim.move_to_com()

        F = np.arccosh((e_pert+np.cos(alpha*max_anomaly))/(1+e_pert*np.cos(alpha*max_anomaly)))
        t = (e_pert*np.sinh(F)-F)*(-1*a_pert)**(3/2)

        sim.integrate(2*t)

        o_binary = sim.particles[1].orbit()
        e_final = o_binary.e
        delta_e_sim = e_final - e0
        avg_de += delta_e_sim/phase_range.size

    return avg_de

def tidal_param(v_infty, b, a0):
    a_pert = -params.m12/(v_infty**2)
    e_pert = np.sqrt(1+(b/a_pert)**2)
    rp = -a_pert*(e_pert-1)
    return rp/a0

def slow_param(v_infty, b, a0):
    a_pert = -params.m12/(v_infty**2)
    e_pert = np.sqrt(1+(b/a_pert)**2)
    rp = -a_pert*(e_pert-1)
    return ((rp/a0)**(3/2))*(1+e_pert)**(-0.5)
