import random
import numpy as np
from multiprocessing import Pool
from math import isnan
import pandas as pd

# to use the results from R code
import os
os.environ["R_HOME"] = os.path.expanduser(r"~\AppData\Local\Programs\R\R-4.3.0") # adapt this path if necessary !
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.vectors import FloatVector

# suppress 'R[write to console]:' output (startup messages, package masking, etc.) and keep actual errors/warnings
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)


from src.functions.create_data import create_data
from src.functions.create_variables import create_variables
from src.functions.estimators import MC_MC_estimator, CV_CV_estimator
from src.functions.dependence_measures import compute_dependence_measures


def process_iteration(i, k, n, m, marginal_T, EVI_T, marginal_S, EVI_S, param_copula, copula, MLE_SSE=False):
    """
    Compute the different EVI (Extreme Value Index) estimators.

    Parameters
    ----------
    i : integer
        Index of the iteration.
    k : integer
        Number of extremes considered for EVI estimation.
    n : integer
        Number of coupled target and source samples.
    m : integer
        Number of additional source samples.
    marginal_T, marginal_S : {"Pareto", "GPD", "Student", "Beta", "Normal"}
        Marginal distributions for target and source samples.
    EVI_T, EVI_S : floats
        EVI values for target and source samples.
    param_copula : float
        Parameter determining the dependence between target and source samples.
    copula : {"Gumbel", "Gaussian"}
        Copula distribution.
        The Gumbel copula models dependence in the upper tail of the distribution.
        The Gaussian copula models dependence mainly around the center of the distribution.
    MLE_SSE : boolean
        If the comparison with the MLE (Maximum Likelihood Estimator) and the SSE (Semi-Supervised Estimator) from [Ahmed et al. - 2025] is included.
        (calling the R code adds computation time)
    """
    random.seed(i)
    sample_T_n, sample_S_n, sample_S_nm, _, _ = create_data(n, m, marginal_T, EVI_T, marginal_S, EVI_S, param_copula, copula)

    A, B, C, D, G, H, E_B, E_D, E_H = create_variables(sample_T_n, sample_S_n, sample_S_nm, k)

    # Hill estimator
    Hill = MC_MC_estimator(A,C)
    Transferred_Hill = CV_CV_estimator(A, C, B, E_B, D, E_D, coef="optimal")

    # Moment estimator
    M1 = Hill
    M2 = MC_MC_estimator(G,C)
    Moment_est = M1 + 1 - 1/2 * 1/(1 - M1**2/M2)
    M1_CV = Transferred_Hill
    M2_CV = CV_CV_estimator(G, C, H, E_H, D, E_D, coef="optimal")
    Tranferred_Moment_est = M1_CV + 1 - 1/2 * 1/(1 - M1_CV**2/M2_CV)

    # MLE and SSE estimator from [Ahmed et al. - 2025]
    if MLE_SSE :
        r_g_values = FloatVector([0,EVI_T])
        numpy2ri.activate()
        ro.r['source']('src/functions/code_article_Ahmed_Einmahl_Zhou.R')
        r_est = ro.globalenv['est']
        result = r_est(i, sample_T_n, sample_S_nm, sample_S_n, n, m, k, r_g_values)
        MLE = result.rx2("est")[0,0]
        SSE = result.rx2("est")[1,1]
    else : 
        MLE = None
        SSE = None

    corr_A_B, corr_C_D, tail_dep = compute_dependence_measures(A, B, C, D, E_B, E_D)

    return {
        "Hill": Hill,
        "Transferred Hill": Transferred_Hill,
        "Moment": Moment_est,
        "Transferred moment": Tranferred_Moment_est,
        "MLE" : MLE,
        "SSE" : SSE,
        "corr_A_B" : corr_A_B,
        "corr_C_D" : corr_C_D,
        "tail_dep" : tail_dep,
    }


def is_valid_number(value):
    """
    Check that value is a float or integer, and not a nan.
    """
    return isinstance(value, (int, float)) and not isnan(value)


def expe(n_exp, k, n, m, marginal_T, EVI_T, marginal_S, EVI_S, param_copula, copula, folder, MLE_SSE=False):
    """
    Compute the average results of different estimators MC/MC, CV/MC and CV/CV with different coefficients, with parallelization, and save the results to csv files.

    Parameters
    ----------
    n_exp : integer
        Number of repetitions to average the results.
    k : integer
        Number of extremes considered for EVI estimation.
    n : integer
        Number of coupled target and source samples.
    m : integer
        Number of additional source samples.
    marginal_T, marginal_S : {"Pareto", "GPD", "Student", "Beta", "Normal"}
        Marginal distributions for target and source samples.
    EVI_T, EVI_S : floats
        EVI values for target and source samples.
    param_copula : float
        Parameter determining the dependence between target and source samples.
    copula : {"Gumbel", "Gaussian"}
        Copula distribution.
        The Gumbel copula models dependence in the upper tail of the distribution.
        The Gaussian copula models dependence mainly around the center of the distribution.
    folder : path
        Folder in which to save the csv files with the results.
    MLE_SSE : boolean
        If the comparison with the MLE (Maximum Likelihood Estimator) and the SSE (Semi-Supervised Estimator) from [Ahmed et al. - 2025] is included.
        (calling the R code adds computation time)
    """
    # Parallelized loop to average the results on n_exp simulations
    args = [(i, k, n, m, marginal_T, EVI_T, marginal_S, EVI_S, param_copula, copula, MLE_SSE) for i in range(n_exp)]
    with Pool() as pool:
        results = pool.starmap(process_iteration, args)

    # Collect all raw results per method
    keys = ["Hill", "Transferred Hill", "Moment", "Transferred moment", "MLE", "SSE"]
    meas_keys = ["corr_A_B", "corr_C_D", "tail_dep"]
    res_dict  = {key: {"res": [r[key] for r in results if is_valid_number(r[key])]} for key in (keys + meas_keys)}

    # Compute statistics per method
    stats_list = []
    for method in keys:
        res = np.array(res_dict[method]["res"])
        stats = {
            "method": method,
            "mean": np.mean(res),
            "bias": np.mean(res) - EVI_T,
            "MSE": (np.mean(res) - EVI_T) ** 2 + np.var(res, ddof=1),
            "variance": np.var(res, ddof=1),
            "std": np.std(res),
            "variation coef": np.std(res) / EVI_T,
            "mean AE": np.mean(np.abs(res - EVI_T)),
            "variance AE": np.var(np.abs(res - EVI_T), ddof=1),
            "std AE": np.sqrt(np.var(np.abs(res - EVI_T), ddof=1))
        }
        stats_list.append(stats)
    results_stats = pd.DataFrame(stats_list)

    # Add Relative Variance Reduction (RVR)
    method_pairs = [("Hill", "Transferred Hill"), ("Moment", "Transferred moment")] + ([("MLE", "SSE")] if MLE_SSE else [])
    for base, transferred in method_pairs:
        var_base = results_stats.loc[results_stats['method'] == base, "variance"].values[0]
        var_trans = results_stats.loc[results_stats['method'] == transferred, "variance"].values[0]
        results_stats.loc[results_stats['method'] == transferred, "RVR"] = (var_base - var_trans) / var_base
    
    # Save statistics on the results
    results_stats.to_csv(f"{folder}/stats_est_n={n}_k={k}_m={m}_marginal_T={marginal_T}_EVI_T={EVI_T}_marginal_S={marginal_S}_EVI_S={EVI_S}_copula={copula}_param-copula={param_copula}_n-exp={n_exp}.csv", index=False)
       
    # Save all raw results (for boxplots)
    results = []
    for key in res_dict.keys():
        res = np.array(res_dict[key]["res"])
        results.append([key] + res.tolist())
    columns = ['Model'] + [f'Result_{i+1}' for i in range(len(results[0]) - 1)]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"{folder}/results_est_n={n}_k={k}_m={m}_marginal_T={marginal_T}_EVI_T={EVI_T}_marginal_S={marginal_S}_EVI_S={EVI_S}_copula={copula}_param-copula={param_copula}_n-exp={n_exp}.csv", index=False)


if __name__ == "__main__":
    expe(n_exp=10000, k=100, n=1000, m=5000, marginal_T="Pareto", EVI_T=0.25, marginal_S="Pareto", EVI_S=5, param_copula=5, copula="Gumbel", folder="results", MLE_SSE=False)