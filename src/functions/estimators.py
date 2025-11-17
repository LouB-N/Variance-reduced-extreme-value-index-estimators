import numpy as np

# code from https://github.com/LouB-N/Control-variates-for-variance-reduced-ratio-of-means-estimators.git
# estimators from the article Control variates for variance-reduced ratio of means estimators - Louison Bocquet--Nouaille, Jérôme Morio, Benjamin Bobbia - 2025

def MC_MC_estimator(A,C):
    """
    Monte-Carlo estimator of the ratio E[A]/E[C].
    
    Parameters
    ----------
    A, C : numpy arrays
        Main dataset samples.
    """
    return np.mean(A)/np.mean(C)



def CV_CV_estimator(A, C, B, E_B, D, E_D, coef="optimal", A_coef=None, C_coef=None, B_coef=None, E_B_coef=None, D_coef=None, E_D_coef=None):
    """
    Estimator of the ratio E[A]/E[C] with control variates B for the numerator and D for the denominator.
    
    Parameters
    ----------
    A, C, B, D : numpy arrays
        Main dataset samples.
    E_B, E_D : floats
        Means of the control variates B and D (known for exact control variates, or estimated on n+m points for approximate control variates)
    coef : {"optimal", "Gordon et al.", "classical"}
        Method for computing the control variate coefficient.
    A_coef, C_coef, B_coef, E_B_coef, D_coef, E_D_coef : numpy arrays, optional
        Separate samples for computing the coefficient alpha (to avoid bias).
        If None, the main dataset is used.
    """
    # use main sample if no coefficient sample provided
    A_c = A if A_coef is None else A_coef
    C_c = C if C_coef is None else C_coef
    B_c = B if B_coef is None else B_coef
    E_B_c = E_B if E_B_coef is None else E_B_coef
    D_c = D if D_coef is None else D_coef
    E_D_c = E_D if E_D_coef is None else E_D_coef

    # compute variables in coefficient expression
    var_B = np.mean((B_c-E_B_c)**2)
    var_D = np.mean((D_c-E_D_c)**2)
    cov_A_B = np.mean((A_c-np.mean(A_c))*(B_c-E_B_c))
    cov_B_C = np.mean((B_c-E_B_c)*(C_c-np.mean(C_c)))
    cov_C_D = np.mean((C_c-np.mean(C_c))*(D_c-E_D_c))
    cov_A_D = np.mean((A_c-np.mean(A_c))*(D_c-E_D_c))
    cov_B_D = np.mean((B_c-E_B_c)*(D_c-E_D_c))

    # compute coefficients
    if coef == "classical":
        alpha = cov_A_B / var_B
        beta = cov_C_D / var_D
    elif coef ==  "Gordon et al.":
        MC_MC = MC_MC_estimator(A,C)
        alpha = cov_A_B / var_B
        beta = (cov_C_D - (1/MC_MC)*cov_A_D + (alpha/MC_MC)*cov_B_D) / var_D
    elif coef == "optimal":
        # alert the user if |Corr(B,D)| is close to 1, as the optimal coefficients do not guarantee variance reduction with linearly correlated control variates
        corr_B_D = cov_B_D / np.sqrt(var_B * var_D)
        if abs(corr_B_D) > 0.99:
            print(f"Warning: |corr(B, D)| is very high ({corr_B_D:.6f})")

        MC_MC = MC_MC_estimator(A,C)
        alpha = (var_D*cov_A_B - MC_MC*var_D*cov_B_C + MC_MC*cov_B_D*cov_C_D - cov_B_D*cov_A_D) / (var_B*var_D - cov_B_D**2)
        beta = ((1/MC_MC)*cov_B_D*cov_A_B - cov_B_D*cov_B_C + var_B*cov_C_D - (1/MC_MC)*var_B*cov_A_D) / (var_B*var_D - cov_B_D**2)

    # estimator computed with main sample
    return (np.mean(A) + alpha * (E_B - np.mean(B))) / (np.mean(C) + beta * (E_D - np.mean(D)))