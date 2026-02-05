"""Ensemble-based hydro-geophysical data assimilation tools."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class HydroGeophysObsOperator:
    """
    Observation operator: hydro state -> geophysical prediction.

    Parameters
    ----------
    petro_transform:
        Callable mapping hydro state to geophysical property model.
    forward_operator:
        Callable mapping geophysical property model to predicted data.
    """

    petro_transform: Callable[[np.ndarray], np.ndarray]
    forward_operator: Callable[[np.ndarray], np.ndarray]

    def __call__(self, hydro_state: np.ndarray) -> np.ndarray:
        geophys_model = self.petro_transform(np.asarray(hydro_state, dtype=float))
        return np.asarray(self.forward_operator(geophys_model), dtype=float).ravel()


class EnsembleKalmanFilter:
    """Classic stochastic Ensemble Kalman Filter."""

    def __init__(self, obs_operator: HydroGeophysObsOperator, obs_cov: np.ndarray, inflation: float = 1.0):
        self.obs_operator = obs_operator
        self.obs_cov = np.asarray(obs_cov, dtype=float)
        self.inflation = float(inflation)

    @staticmethod
    def _anomalies(matrix: np.ndarray) -> np.ndarray:
        mean = np.mean(matrix, axis=1, keepdims=True)
        return matrix - mean

    def forecast(self, ensemble: np.ndarray, model_step: Callable[[np.ndarray], np.ndarray], **kwargs) -> np.ndarray:
        """Propagate each ensemble member through the hydro model step."""
        ensemble = np.asarray(ensemble, dtype=float)
        forecasted = np.zeros_like(ensemble)

        for i in range(ensemble.shape[1]):
            forecasted[:, i] = np.asarray(model_step(ensemble[:, i], **kwargs), dtype=float).ravel()

        if self.inflation != 1.0:
            mean = np.mean(forecasted, axis=1, keepdims=True)
            forecasted = mean + self.inflation * (forecasted - mean)

        return forecasted

    def update(self, ensemble: np.ndarray, observations: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Assimilate observations into a hydro-state ensemble."""
        X = np.asarray(ensemble, dtype=float)
        y_obs = np.asarray(observations, dtype=float).ravel()
        n_members = X.shape[1]

        Y = np.column_stack([self.obs_operator(X[:, i]) for i in range(n_members)])
        if Y.shape[0] != y_obs.size:
            raise ValueError("Observation size does not match operator output size.")

        A = self._anomalies(X)
        D = self._anomalies(Y)

        Pxy = (A @ D.T) / max(n_members - 1, 1)
        Pyy = (D @ D.T) / max(n_members - 1, 1) + self.obs_cov

        K = Pxy @ np.linalg.pinv(Pyy)

        if rng is None:
            rng = np.random.default_rng()

        noise = rng.multivariate_normal(np.zeros(y_obs.size), self.obs_cov, size=n_members).T
        innovations = (y_obs.reshape(-1, 1) + noise) - Y

        X_post = X + K @ innovations

        if self.inflation != 1.0:
            mean = np.mean(X_post, axis=1, keepdims=True)
            X_post = mean + self.inflation * (X_post - mean)

        return X_post


class ESMDA:
    """
    Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    Performs multiple smoother updates with inflated observation covariance.
    """

    def __init__(
        self,
        obs_operator: HydroGeophysObsOperator,
        obs_cov: np.ndarray,
        n_steps: int = 4,
        inflation_factors: Optional[List[float]] = None,
    ):
        self.obs_operator = obs_operator
        self.obs_cov = np.asarray(obs_cov, dtype=float)
        self.n_steps = int(n_steps)

        if inflation_factors is None:
            self.inflation_factors = [float(self.n_steps)] * self.n_steps
        else:
            if len(inflation_factors) != self.n_steps:
                raise ValueError("inflation_factors length must equal n_steps.")
            self.inflation_factors = [float(v) for v in inflation_factors]

    def update(
        self,
        ensemble: np.ndarray,
        observations: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """Run ES-MDA updates and return final ensemble with history."""
        if rng is None:
            rng = np.random.default_rng()

        X = np.asarray(ensemble, dtype=float)
        y_obs = np.asarray(observations, dtype=float).ravel()

        history = []

        for alpha in self.inflation_factors:
            R_alpha = alpha * self.obs_cov
            enkf = EnsembleKalmanFilter(
                obs_operator=self.obs_operator,
                obs_cov=R_alpha,
                inflation=1.0,
            )
            X = enkf.update(X, y_obs, rng=rng)
            history.append(X.copy())

        return {
            "ensemble": X,
            "history": np.array(history),
        }
