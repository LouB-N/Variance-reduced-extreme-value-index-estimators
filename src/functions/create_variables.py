import numpy as np

def create_variables(sample_T, sample_S, sample_S_nm, k):
    """
    Compute the different variables to compute the Hill and moment estimators of the EVI with ratio of means.
    Parameters
    ----------
    sample_T : numpy array
        Target sample of size n (coupled observations).
    sample_S : numpy array
        Source sample of size n (coupled observations).
    sample_S_nm : numpy array
        All source observations, samples of size n+m.
    """
    n = len(sample_T)
    threshold_T = np.sort(sample_T)[n-k]
    threshold_S = np.sort(sample_S)[n-k]

    A = np.where(sample_T >= threshold_T, np.log(sample_T) - np.log(threshold_T), 0)
    C = np.where(sample_T >= threshold_T, 1, 0)
    G = np.where(sample_T >= threshold_T, (np.log(sample_T) - np.log(threshold_T))**2, 0)
    B = np.where(sample_S >= threshold_S, np.log(sample_S) - np.log(threshold_S), 0)
    E_B = np.mean(np.where(sample_S_nm >= threshold_S, np.log(sample_S_nm) - np.log(threshold_S), 0))
    D = np.where(sample_S >= threshold_S, 1, 0)
    E_D = np.mean(np.where(sample_S_nm >= threshold_S, 1, 0))
    H = np.where(sample_S >= threshold_S, (np.log(sample_S) - np.log(threshold_S))**2, 0)
    E_H = np.mean(np.where(sample_S_nm >= threshold_S, (np.log(sample_S_nm) - np.log(threshold_S))**2, 0))
    return A, B, C, D, G, H, E_B, E_D, E_H