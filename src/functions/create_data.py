import openturns as ot
import numpy as np


def create_data(n, m, marginal_T, EVI_T, marginal_S, EVI_S, param_copula, copula):
    """
    Create data with chosen target and source EVI.
    
    Parameters
    ----------
    n : integer
        Number of coupled samples.
    m : integer
        Number of additional source samples. (Both target and source additional samples are provided for further analysis.)
    marginal_T, marginal_S : {"Pareto", "GPD", "Student", "Beta", "Normal"}
        Marginal distributions for target and source samples.
    EVI_T, EVI_S : floats
        EVI (Extreme Value Index) values for target and source samples.
    param_copula : float
        Parameter determining the dependence between target and source samples.
    copula : {"Gumbel", "Gaussian"}
        Copula distribution.
        The Gumbel copula models dependence in the upper tail of the distribution.
        The Gaussian copula models dependence mainly around the center of the distribution.
    """     
    if marginal_T == "Pareto": # Pareto marginal, with minimal value=1e-3 (scale parameter beta) (so samples are >0)
        marginal_T = ot.Pareto(1e-3, 1/EVI_T)
    elif marginal_T == "GPD": # Generalized Pareto marginal, with minimal value=1e-3 (scale parameter beta) (so samples are >0)
        marginal_T = ot.GeneralizedPareto(1e-3, EVI_T)
    elif marginal_T == "Student": # Student marginal
        marginal_T = ot.Student(1/EVI_T)
    elif marginal_T == "Beta":
        marginal_T = ot.Beta(2,-EVI_T, 0, 1)
    elif marginal_T == "Normal":
        marginal_T = ot.Normal(0, 1) # EVI_T = 0

    if marginal_S == "Pareto":
        marginal_S = ot.Pareto(1e-3, 1/EVI_S)
    elif marginal_S == "GPD":
        marginal_S = ot.GeneralizedPareto(1e-3, EVI_S)
    elif marginal_S == "Student":
        marginal_S = ot.Student(1/EVI_S)
    elif marginal_S == "Beta":
        marginal_S = ot.Beta(2,-EVI_S, 0, 1)
    elif marginal_S == "Normal":
        marginal_S = ot.Normal(0, 1) # EVI_S = 0

    if copula=="Gaussian":
        correlation_matrix = ot.CorrelationMatrix(2)
        correlation_matrix[0, 1] = param_copula
        copula = ot.NormalCopula(correlation_matrix)
    elif copula=="Gumbel":
        copula= ot.GumbelCopula(param_copula)

    distribution = ot.ComposedDistribution([marginal_T, marginal_S], copula)
    sample_T_S_all = ot.RandomVector(distribution).getSample(n+m)
    sample_T_nm = np.array(sample_T_S_all.getMarginal(0)).ravel()
    sample_S_nm = np.array(sample_T_S_all.getMarginal(1)).ravel()
    sample_T_n = sample_T_nm[:n]
    sample_S_n = sample_S_nm[:n]
    sample_T_m = sample_T_nm[n:]
    sample_S_m = sample_S_nm[n:]
    return sample_T_n, sample_S_n, sample_S_nm, sample_T_m, sample_S_m
