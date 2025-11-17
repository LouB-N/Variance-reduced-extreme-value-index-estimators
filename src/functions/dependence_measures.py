import numpy as np

def compute_dependence_measures(A, B, C, D, E_B, E_D):
    """
    Estimate different dependence measures corr(A,B), corr(C,D) and the tail dependence

    Parameters
    ----------
    A, B, C, D : numpy arrays
        Variables and control variates for numerator and denominator.
    E_B, E_D : floats
        Means of the control variates B and D (known for exact control variates, or estimated on n+m points for approximate control variates)
    """
    corr_A_B = np.mean((A - np.mean(A)) * (B - E_B)) / np.sqrt(np.mean((A - np.mean(A))**2) * np.mean((B - E_B)**2))
    corr_C_D = np.mean((C - np.mean(C)) * (D - E_D)) / np.sqrt(np.mean((C - np.mean(C))**2) * np.mean((D - E_D)**2))
    tail_dep = np.sum((C == 1) & (D == 1)) / np.sum(C == 1)
    return corr_A_B, corr_C_D, tail_dep