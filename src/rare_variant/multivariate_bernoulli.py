import numpy as np
import numpy.linalg as la
import scipy.stats as st

def multivariate_bernoulli_rvs(mean, covariance, seed, n_samples=1):
    """
    I tried this but couldn't get things to work in the [0,1] case and some
    values of covariance matrices were giving me unfeasible values.

    https://mathoverflow.net/questions/210483/generate-bernoulli-
                                    vector-with-given-covariance-matrix

    Instad I just generate from a multivariate normal making sure marginal 
    expectations are correct via thresh and positive/negative correlations
    are represented in covariance (but not exact values). Trial and error
    can be used to make a reasonable correlation I suppose but need to get 
    moving.

    My suspicion is that with rare diseases its really hard to make negative
    correlations because most of the time if one is observed the other is 
    not due to random chance. Which is fine I suppose since we mainly care
    about groups of diseases cooccuring.
    """
    thresh = st.norm.ppf(mean)
    multi_draw = st.multivariate_normal.rvs(mean=thresh,
        cov=covariance, size=n_samples, random_state=seed
    )
    draws = 1*(multi_draw>0)
    return draws

