from typing import Any, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import scipy
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from pandas.testing import assert_frame_equal
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.linalg import det, inv
import pandas as pd
import scipy
from sklearn.decomposition import PCA


@dataclass
class InputNormal:
    lamb: pd.Series
    Upsilon: pd.DataFrame

    def __post_init__(self) -> None:
        assert det(self.Upsilon) >= 0

    def __iter__(self) -> Iterator:
        return iter([self.lamb, self.Upsilon])


@dataclass
class SparseCovMatrix:
    """Represent input-space covariance matrix Upsilon in sparse form"""

    Sigma: pd.DataFrame
    T: pd.DataFrame
    epsilon: pd.Series

    @property
    def shape(self):
        return T.shape[0], T.shape[0]

    def to_dense(self):
        return self.T @ self.Sigma @ self.T.T + np.diag(self.epsilon)

    def __matmul__(self, X):
        T = self.T
        Sigma = self.Sigma
        epsilon = self.epsilon
        return T @ Sigma @ (T.T @ X) + diag_mult(epsilon, X)

    def inv_quad_form(self, v):
        """Compute v Upsilon v.T using woodbury identity."""
        raise NotImplementedError


@dataclass
class SparseDiagMatrix:
    epsilon: pd.Series

    def __inv__(self):
        return SparseDiagMatrix(self.epsilon**-1.0)

    def __matmul__(self, X):
        return diag_mult(self.epsilon.values, X)


def test_SparseDiagMatrix():
    epsilon = pd.Series([1, 2, 3])
    E = SparseDiagMatrix(epsilon)
    X = np.random.normal(size=(3, 3))
    expected = np.diag(epsilon) @ X
    actual = E @ X
    assert np.allclose(expected, actual)


def test_SparseCovMatrix_to_dense():
    Sigma = pd.DataFrame([[1, 2], [3, 4]])
    T = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    epsilon = pd.Series([1, 2, 3])
    A = SparseCovMatrix(Sigma, T, epsilon)
    expected_A = pd.DataFrame(
        {
            0: {0: 28, 1: 59, 2: 91},
            1: {0: 61, 1: 135, 2: 205},
            2: {0: 95, 1: 207, 2: 322},
        }
    )
    assert_frame_equal(expected_A, A.to_dense())


def test_SparseCovMatrix_mat_mul():
    """Ensure sparse multiplication equals dense multiplication."""
    Sigma = pd.DataFrame([[1, 2], [3, 4]])
    T = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    epsilon = pd.Series([1, 2, 3])
    A = SparseCovMatrix(Sigma, T, epsilon)
    X = pd.DataFrame(np.random.normal(size=(3, 3)))
    assert np.allclose(A.to_dense() @ X, A @ X)


@dataclass
class DegenerateInputNormal:
    """Represent a multivariate normal, possibly with some observed components."""

    lamb: pd.Series
    Upsilon: pd.DataFrame

    def __post_init__(self) -> None:
        assert np.linalg.det(self.Upsilon) >= 0

    def __iter__(self) -> Iterator:
        return iter([self.lamb, self.Upsilon])


@dataclass
class LatentNormal:
    mu: pd.Series
    Sigma: pd.DataFrame

    def __post_init__(self) -> None:
        assert np.linalg.det(self.Sigma) >= 0

    def __iter__(self) -> Iterator:
        return iter([self.mu, self.Sigma])

    def __eq__(self, other):
        return (self.mu == other.mu).all() and (self.Sigma == other.Sigma).all().all()


def project_gaussian(
    T: pd.DataFrame, input_normal: InputNormal | DegenerateInputNormal
) -> LatentNormal:
    lamb, Upsilon = input_normal
    mu = lamb @ T
    Sigma = T.T @ Upsilon @ T
    return LatentNormal(mu, Sigma)


def calculate_inv_project_noise(
    X: pd.DataFrame, T: pd.DataFrame, latent_normal: LatentNormal
) -> float:
    epsilons = np.logspace(-5, 5, 100)

    def f(epsilon: float) -> float:
        normal_dist = scipy.stats.multivariate_normal(
            *inv_project_gaussian(T, latent_normal, epsilon)
        )
        log_likelihoods = normal_dist.logpdf(X)
        return sum(log_likelihoods)

    return max(epsilons, key=f)


def inv_project_gaussian(
    T: pd.DataFrame, latent_normal: LatentNormal, epsilon: float
) -> InputNormal:
    mu, Sigma = latent_normal
    lamb = T @ mu
    Upsilon = T @ Sigma @ T.T
    Upsilon += epsilon * np.eye(len(Upsilon))
    assert np.linalg.det(Upsilon) > 0
    return InputNormal(lamb, Upsilon)


def sample_T(num_features: int, num_latents: int) -> np.ndarray:
    A = np.random.normal(size=(num_latents, num_features))
    pca = PCA()
    T = pca.fit(A).components_.T
    assert T.shape == (num_features, num_latents)
    return T


def test_project_gaussian() -> None:
    variants = ["variant1", "variant2", "variant3"]
    pcs = ["pc1", "pc2", "pc3"]
    T = pd.DataFrame(sample_T(len(variants), len(pcs)), index=variants, columns=pcs)
    lamb = pd.Series(np.random.random(3), index=variants)
    Upsilon = pd.DataFrame(
        random_positive_semidefinite_matrix(len(variants)),
        index=variants,
        columns=variants,
    )
    mu, Sigma = project_gaussian(T, InputNormal(lamb, Upsilon))
    assert all(mu.index == pcs)
    assert all(Sigma.index == pcs)
    assert all(Sigma.columns == pcs)


def test_inv_project_gaussian() -> None:
    variants = ["variant1", "variant2", "variant3"]
    pcs = ["pc1", "pc2", "pc3"]
    T = pd.DataFrame(sample_T(len(variants), len(pcs)), index=variants, columns=pcs)
    mu = pd.Series(np.random.random(3), index=pcs)
    Sigma = pd.DataFrame(
        random_positive_semidefinite_matrix(len(pcs)), index=pcs, columns=pcs
    )
    lamb, Upsilon = inv_project_gaussian(T, LatentNormal(mu, Sigma), 0)
    assert all(lamb.index == variants)
    assert all(Upsilon.index == variants)
    assert all(Upsilon.columns == variants)


def test_project_gaussian_composition() -> None:
    samples = ["sample1", "sample2", "sample3"]
    variants = ["variant1", "variant2", "variant3"]
    pcs = ["pc1", "pc2", "pc3"]

    T = pd.DataFrame(sample_T(len(variants), len(pcs)), index=variants, columns=pcs)
    lamb = pd.Series(np.random.random(3), index=variants)
    Upsilon = pd.DataFrame(
        random_positive_semidefinite_matrix(len(variants)),
        index=variants,
        columns=variants,
    )
    input_normal = InputNormal(lamb, Upsilon)
    latent_normal = project_gaussian(T, input_normal)
    input_normal_p = inv_project_gaussian(T, latent_normal, 0)
    assert np.allclose(input_normal.lamb, input_normal_p.lamb)
    assert np.allclose(input_normal.Upsilon, input_normal_p.Upsilon)


def make_positive_semi_definite(A: np.ndarray) -> np.ndarray:
    B = A.T @ A
    assert np.linalg.det(B) > 0
    return B


def random_positive_semidefinite_matrix(n: int) -> np.ndarray:
    return make_positive_semi_definite(np.random.normal(size=(n, n)))


def cond_input_dist_given_obs_and_cluster(
    x: pd.Series, T: pd.DataFrame, latent_normal: LatentNormal, epsilon: float
) -> DegenerateInputNormal:
    obs = list(x.index[x.notna()])
    mis = list(x.index[x.isna()])
    x_obs = x[obs]
    x_mis = x[mis]
    input_normal = inv_project_gaussian(T, latent_normal, epsilon)
    lamb, Upsilon = input_normal.lamb, input_normal.Upsilon
    lamb_obs = lamb[obs]
    lamb_mis = lamb[mis]
    cond_lamb = lamb_mis + Upsilon.loc[mis, list(obs)] @ (
        Upsilon.loc[obs, list(obs)] ** -1
    ) @ (x_obs - lamb_obs)
    cond_Upsilon = (
        Upsilon.loc[mis, mis]
        - Upsilon.loc[mis, obs] @ (Upsilon.loc[obs, obs] ** -1) @ Upsilon.loc[obs, mis]
    )
    full_cond_lamb = pd.concat([x_obs, cond_lamb])
    # if variant is observed, covariance with anything else will be zero
    full_cond_Upsilon = (0 * Upsilon + cond_Upsilon).fillna(0)
    return DegenerateInputNormal(full_cond_lamb, full_cond_Upsilon)


def cond_latent_dist_given_obs_and_cluster(
    x: pd.Series, T: pd.DataFrame, latent_normal: LatentNormal, epsilon: float
) -> LatentNormal:
    cond_input_normal = cond_input_dist_given_obs_and_cluster(
        x, T, latent_normal, epsilon
    )
    cond_latent_normal = project_gaussian(T, cond_input_normal)
    return cond_latent_normal


def test_end_to_end() -> None:
    np.random.seed(1337)
    Xdata = np.array(
        [
            [-1, -1, 1, 1],
            [-1, -1, 1, 1],
            [-1, -1, 1, 1],
            [-1, -1, 1, 1],
            [1, 1, -1, -1],
            [1, 1, -1, -1],
            [1, 1, -1, -1],
            [1, 1, -1, -1],
        ]
    )
    Xdata = Xdata + np.random.normal(0, 0.1, size=Xdata.shape)
    N, K = Xdata.shape
    P = 2
    col_means = Xdata.mean(axis=0)
    assert len(col_means) == K
    Xdata_centered = Xdata - col_means
    y = np.array([0] * 4 + [1] * 4)
    samples = [f"sample{i}" for i in range(1, N + 1)]
    variants = [f"variant{i}" for i in range(1, K + 1)]
    pcs = [f"pc{i}" for i in range(1, P + 1)]
    X = pd.DataFrame(Xdata_centered, index=samples, columns=variants)
    pca = PCA(n_components=P)
    Xpc = pd.DataFrame(pca.fit_transform(X), index=samples, columns=pcs)
    T = pd.DataFrame(pca.components_.T, index=variants, columns=pcs)
    # estimate noise in inverse transform

    assert np.allclose(Xpc, (X - X.mean()) @ T)
    Z_clusters = {}
    for yi in [0, 1]:
        rel_Xpc = Xpc.loc[y == yi, :]
        mu = pd.Series(np.mean(rel_Xpc, axis=0), index=pcs)
        Sigma = pd.DataFrame(np.cov(rel_Xpc, rowvar=0), index=pcs, columns=pcs)
        Z_clusters[yi] = LatentNormal(mu, Sigma)

    epsilon0 = calculate_inv_project_noise(X[y == 0], T, Z_clusters[0])
    epsilon1 = calculate_inv_project_noise(X[y == 1], T, Z_clusters[1])

    x = pd.Series([0, np.nan, np.nan, np.nan], index=variants)
    cond_mu, cond_Sigma = cond_latent_dist_given_obs_and_cluster(
        x, T, Z_clusters[0], epsilon0
    )


class InputLatentPair:
    def __init__(self):
        self.T = None
        self.input_normal = None
        self.latent_normal = None
        self.epsilon = None

    def fit(self, T: pd.DataFrame, X: pd.DataFrame) -> None:
        Z = X @ T
        pcs = Z.columns

        mu = pd.Series(np.mean(Z, axis=0), index=pcs)
        Sigma = pd.DataFrame(np.cov(Z, rowvar=0), index=pcs, columns=pcs)
        self.latent_normal = LatentNormal(mu, Sigma)
        self.epsilon = calculate_inv_project_noise(X, T, self.latent_normal)
        self.input_normal = inv_project_gaussian(T, self.latent_normal, self.epsilon)
        self.T = T

    def transform(self, x: pd.Series) -> LatentNormal:
        """Project a (potentially partially observed) input vector x into latent space"""
        return cond_latent_dist_given_obs_and_cluster(
            x, self.T, self.latent_normal, self.epsilon
        )

    def input_log_proba(self, x: pd.Series) -> float:
        obs = list(x.index[x.notna()])
        lamb, Upsilon = self.input_normal
        return scipy.stats.multivariate_normal(lamb[obs], Upsilon.loc[obs, obs]).logpdf(
            x[obs]
        )


def test_InputLatentPair():
    iris_df = (load_iris(as_frame=True))["frame"]
    y = iris_df["target"]
    X = iris_df.drop("target", axis="columns")
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, stratify=y, random_state=1337
    )
    n_components = 2
    pcs = [f"pc{i}" for i in range(1, n_components + 1)]
    pca = PCA(n_components=n_components)
    pca.fit(train_X)
    T = pd.DataFrame(pca.components_.T, index=train_X.columns, columns=pcs)
    input_latent_pair = InputLatentPair()
    train_X0 = train_X[train_y == 0]
    input_latent_pair.fit(T, train_X0)
    x = pd.Series([0, 0, 0, 0], index=train_X.columns)
    actual_latent = input_latent_pair.transform(x)
    mu, Sigma = input_latent_pair.latent_normal
    expected_latent = LatentNormal(0 * mu, 0 * Sigma)
    assert expected_latent == actual_latent

    x = pd.Series([0, np.nan, np.nan, np.nan], index=train_X.columns)
    actual_latent = input_latent_pair.transform(x)
    expected_latent = LatentNormal(
        pd.Series({"pc1": 0.9941423903326845, "pc2": 0.746744067190767}),
        pd.DataFrame(
            {
                "pc1": {"pc1": 0.1181220256465803, "pc2": -0.038029439621672524},
                "pc2": {"pc1": -0.038029439621672524, "pc2": 0.09156929460328199},
            }
        ),
    )
    assert actual_latent == expected_latent


def _woodbury_inv_general_form(A, U, C, V):
    return inv(A) - inv(A) @ U @ inv(inv(C) + V @ inv(A) @ U) @ V @ inv(A)


def woodbury_inv(epsilon: pd.Series, Sigma: pd.DataFrame, Tau: pd.DataFrame):
    E = np.diag(epsilon)
    return _woodbury_inv_general_form(E, T, Sigma, T.T)


def calc_quad_form(v, E, Sigma, T) -> float:
    """calculate v.T @ (T @ Sigma @ T.T + E)**-1 @ v"""
    return (
        v @ inv(E) @ v
        - v @ inv(E) @ T @ inv((inv(Sigma) + T.T @ inv(E) @ T)) @ T.T @ inv(E) @ v
    )


from functools import reduce


def prod(xs):
    return reduce(lambda x, y: x * y, xs)


def dnormal_sparse(
    x: pd.Series, T, mu: pd.Series, Sigma: pd.DataFrame, Evec: pd.Series
) -> float:
    k = len(x)
    lamb = T @ mu
    v = x - lamb
    p = len(Sigma)
    Einv = Evec**-1.0
    detE = prod(Evec)
    D = detE * det(np.eye(p) + (Sigma @ T.T) @ diag_mult(Einv, T))
    Q = v @ diag_mult(Einv, v) - v @ diag_mult(Einv, T) @ inv(
        inv(Sigma) + T.T @ diag_mult(Einv, T)
    ) @ T.T @ diag_mult(Einv, v)
    Z = (2 * np.pi) ** (k / 2) * np.sqrt(D)
    print(Einv, D, Q, Z)
    return 1 / Z * np.exp(-1 / 2 * Q)


def dnormal_ref(
    x: pd.Series, T, mu: pd.Series, Sigma: pd.DataFrame, E: pd.DataFrame
) -> float:
    k = len(x)
    lamb = T @ mu
    Upsilon = T @ Sigma @ T.T + E
    v = x - lamb
    p = len(Sigma)
    D = det(Upsilon)
    Q = v @ inv(Upsilon) @ v
    Z = (2 * np.pi) ** (k / 2) * np.sqrt(D)
    return 1 / Z * np.exp(-1 / 2 * Q)


def test_dnormal_sparse():
    k = 5
    p = 2
    variants = [f"variant{i}" for i in range(k)]
    pcs = [f"pc{i}" for i in range(p)]
    x = pd.Series(np.random.normal(size=k), index=variants)
    mu = pd.Series(np.random.normal(size=p), index=pcs)
    Sigma = pd.DataFrame(
        random_positive_semidefinite_matrix(len(pcs)), index=pcs, columns=pcs
    )
    T = pd.DataFrame(sample_T(k, p), columns=pcs, index=variants)
    E = 2 * np.ones(k)  # 2 * pd.DataFrame(np.eye(k), columns=variants, index=variants)
    lamb = T @ mu
    Upsilon = T @ Sigma @ T.T + np.diag(E)
    expected = scipy.stats.multivariate_normal(lamb, Upsilon).pdf(x)
    actual = dnormal_sparse(x, T, mu, Sigma, E)
    print(expected, actual)
    assert np.isclose(expected, actual)


def test_calc_quad_form():
    k = 5
    p = 2
    v = np.random.normal(size=k)
    E = np.random.normal(size=(k, k))
    Sigma = np.random.normal(size=(p, p))
    T = np.random.normal(size=(k, p))
    expected = v @ inv(T @ Sigma @ T.T + E) @ v
    actual = calc_quad_form(v, E, Sigma, T)
    assert np.allclose(expected, actual)


def test_woodbury_inv_general_Form():
    n = 5
    k = 3
    A = np.random.normal(size=(n, n))
    U = np.random.normal(size=(n, k))
    C = np.random.normal(size=(k, k))
    V = np.random.normal(size=(k, n))
    expected = inv(A + U @ C @ V)
    actual = _woodbury_inv_general_form(A, U, C, V)
    assert np.allclose(expected, actual)


def diag_mult(d: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Multiply sparse diagonal matrix d by matrix X."""
    return (d * X.T).T


def test_diag_mult_2d():
    X = pd.DataFrame(
        random_positive_semidefinite_matrix(3),
        index=["a", "b", "c"],
        columns=["x", "y", "z"],
    )
    d = np.array([1, 2, 3])
    expected = np.diag(d) @ X
    actual = diag_mult(d, X)
    np.allclose(expected, actual)


def test_diag_mult_1d():
    d = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    expected = np.diag(d) @ v
    actual = diag_mult(d, v)
    assert (expected == actual).all()
