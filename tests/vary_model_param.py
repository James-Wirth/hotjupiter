import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tqdm
from matplotlib import pyplot as plt

from hjmodel.config import *
from hjmodel.hjmodel import eval_system
from hjmodel import model_utils, rand_utils
from test_data.test_model_param_data.analytic_results import Analytic

# canonical values (unless the parameter in question is being varied)
CANON = {
    'n_tot': 2.5E-3,
    'sigma_v': 1.266,
    'e_init': 0.3,
    'a_init': 1,
    'm1': 1,
    'm2': 1E-3,
    'total_time': 10000
}

def get_outcome_dict(num_systems, *args):
    outcomes = np.array(Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_system)(*args) for _ in tqdm.tqdm(range(num_systems))
    )).T
    unique, counts = np.unique(outcomes[2], return_counts=True)
    return {x:0 for x in list(SC_DICT.values())} | dict(zip(unique.astype(int), counts/num_systems))

def save_res_for_param(param_name, param_values, num_systems):
    d = []
    copy = CANON.copy()
    for i in range(len(param_values)):
        copy[param_name] = param_values[i]
        outcome_dict = get_outcome_dict(num_systems[i], *copy.values())
        error = 1/np.sqrt(num_systems[i])
        d.append(
            {
                param_name: param_values[i],
                'error': error,
            } | outcome_dict
        )
    df = pd.DataFrame(data=d)
    df.to_parquet(f'test_data/test_model_param_data/test_{param_name}.pq', engine='pyarrow')

def vary_density():
    param_values = np.geomspace(1E-4, 1, 20)
    num_systems = [10000] * 20
    save_res_for_param(param_name='n_tot', param_values=param_values, num_systems=num_systems)
    df = pd.read_parquet(f'test_data/test_model_param_data/test_n_tot.pq', engine='pyarrow')
    print(df.head())

def vary_sigma():
    param_values = np.geomspace(0.211, 6.33, 20)
    num_systems = [10000] * 20
    save_res_for_param(param_name='sigma_v', param_values=param_values, num_systems=num_systems)
    df = pd.read_parquet(f'test_data/test_model_param_data/test_sigma_v.pq', engine='pyarrow')
    print(df.head())

def vary_a_init():
    param_values = np.geomspace(1, 1E1, 20)
    num_systems = [10000] * 20
    save_res_for_param(param_name='a_init', param_values=param_values, num_systems=num_systems)
    df = pd.read_parquet(f'test_data/test_model_param_data/test_a_init.pq', engine='pyarrow')
    print(df.head())

def vary_m1():
    param_values = np.geomspace(0.08, 0.8, 20)
    num_systems = [10000] * 20
    save_res_for_param(param_name='m1', param_values=param_values, num_systems=num_systems)
    df = pd.read_parquet(f'test_data/test_model_param_data/test_m1.pq', engine='pyarrow')
    print(df.head())

def add_fig_to_ax(ax, param_name: str,
                  xrange: list[float], yrange: list[float],
                  xlabel: str,
                  clip: dict[str: int]):
    df = pd.read_parquet(f'test_data/test_model_param_data/test_{param_name}.pq', engine='pyarrow').sort_values(by=param_name)
    for outcome in SC_DICT.values():
        if clip[outcome] == 0:
            ax.plot(df[param_name], df[f'{outcome}'],
                    color=COLOR_DICT[outcome][0],
                    label=outcome,
                    zorder=10 - outcome)
        else:
            ax.plot(df[param_name][0:-clip[outcome]], df[f'{outcome}'][0:-clip[outcome]],
                    color=COLOR_DICT[outcome][0],
                    label=outcome,
                    zorder=10-outcome)

    # get analytic predictions
    param_values = np.geomspace(xrange[0], xrange[1], 1000)
    copy = CANON.copy()
    analytic_res = []
    for i in range(len(param_values)):
        copy[param_name] = param_values[i]
        analytic_res.append(Analytic(*copy.values()).get_analytic_probabilities())
    analytic_res_T = np.array(analytic_res).T
    for i in range(analytic_res_T.shape[0]):
        ax.plot(param_values, analytic_res_T[i], linestyle='dotted', color=COLOR_DICT[i][0])

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel(xlabel)

def save_fig():
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 4), gridspec_kw={'height_ratios':[1.5,1],
                                                                           'hspace':0.05,
                                                                           'wspace':0.15})

    clip = {SC_DICT['NM']: 3,SC_DICT['I']: 0,SC_DICT['TD']: 0,SC_DICT['HJ']: 0,SC_DICT['WJ']: 3}
    add_fig_to_ax(axs[0,0], param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[0, 1],
                  xlabel='$n$ / $\\mathrm{pc}^{-3}$', clip=clip)
    axs[0,0].set_xscale('log')
    axs[0,0].set_xticks([])
    add_fig_to_ax(axs[1, 0], param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[0, 5E-2],
                  xlabel='$n$ / $\\mathrm{pc}^{-3}$', clip=clip)
    axs[1, 0].set_xscale('log')

    clip = {SC_DICT['NM']: 0, SC_DICT['I']: 0, SC_DICT['TD']: 0, SC_DICT['HJ']: 0, SC_DICT['WJ']: 0}
    add_fig_to_ax(axs[0,1], param_name='sigma_v',
                  xrange=[0.211, 6.33], yrange=[0, 1],
                  xlabel='$\\sigma$ / $\\mathrm{km} \\mathrm{s}^{-1}$', clip=clip)
    axs[0, 1].set_xticks([])
    add_fig_to_ax(axs[1, 1], param_name='sigma_v',
                  xrange=[0.211, 6.33], yrange=[0, 5E-2],
                  xlabel='$\\sigma$ / $\\mathrm{km} \\mathrm{s}^{-1}$', clip=clip)
    axs[0, 1].set_yticks([])
    axs[1, 1].set_yticks([])

    clip = {SC_DICT['NM']: 0, SC_DICT['I']: 0, SC_DICT['TD']: 0, SC_DICT['HJ']: 0, SC_DICT['WJ']: 0}
    add_fig_to_ax(axs[0,2], param_name='a_init',
                  xrange=[1E0, 1E1], yrange=[0, 1],
                  xlabel='$a_0$ / au', clip=clip)
    axs[0,2].set_xticks([])
    add_fig_to_ax(axs[1, 2], param_name='a_init',
                  xrange=[1E0, 1E1], yrange=[0, 5E-2],
                  xlabel='$a_0$ / au', clip=clip)
    axs[0, 2].set_yticks([])
    axs[1, 2].set_yticks([])

    # style figure
    axs[0,0].plot(np.nan, np.nan, linestyle='dotted', color='black', label='Analytic')
    axs[0,0].set_ylabel('Outcome probability')
    h, l = axs[0,0].get_legend_handles_labels()
    fig.legend(h, l, ncols=6, bbox_to_anchor=(0.82, 1.03), frameon=True)
    fig.tight_layout()
    plt.savefig('test_data/test_model_param_data/test_model_param_plot.pdf', format='pdf', bbox_inches="tight")

if __name__ == '__main__':
    # vary_m1()
    # vary_a_init()
    save_fig()