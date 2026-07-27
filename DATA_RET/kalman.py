"""
kalman.py

Calibration d'un modèle local-level (state-space AR(1) bruité) sur un spread
cointégré, via filtre de Kalman + lisseur RTS + algorithme EM. Fonctions pures,
même convention que features.py : aucun I/O, aucun effet de bord.

Modèle :
    x_t = A + B * x_{t-1} + eps_t,   eps_t ~ N(0, C^2)   (état latent)
    y_t = x_t + eta_t,                eta_t ~ N(0, D^2)   (observation = spread bruité)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


def kalman_filter(
    y: np.ndarray, A: float, B: float, C: float, D: float, x0: float, P0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filtre de Kalman causal (prédiction + correction). Retourne x_hat, P, x_pred, P_pred, K."""
    N_plus_1 = len(y)
    x_hat = np.zeros(N_plus_1)
    P = np.zeros(N_plus_1)
    x_pred = np.zeros(N_plus_1)
    P_pred = np.zeros(N_plus_1)
    K = np.zeros(N_plus_1)

    x_hat[0] = x0
    P[0] = P0

    for t in range(1, N_plus_1):
        x_pred[t] = A + B * x_hat[t - 1]
        P_pred[t] = (B ** 2) * P[t - 1] + (C ** 2)

        S_t = P_pred[t] + (D ** 2)
        K[t] = P_pred[t] / S_t
        residual = y[t] - x_pred[t]

        x_hat[t] = x_pred[t] + K[t] * residual
        P[t] = (1 - K[t]) * P_pred[t]

    return x_hat, P, x_pred, P_pred, K


def kalman_smoother(
    x_hat: np.ndarray, P: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray, B: float, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lisseur RTS (rétrospectif, utilisé uniquement pour l'estimation EM — jamais pour un signal de trading)."""
    N_plus_1 = len(x_hat)
    N = N_plus_1 - 1

    x_smooth = np.copy(x_hat)
    P_smooth = np.copy(P)
    P_cross = np.zeros(N_plus_1)
    J = np.zeros(N_plus_1)

    for t in range(N - 1, -1, -1):
        J[t] = 0.0 if abs(P_pred[t + 1]) < 1e-12 else P[t] * B / P_pred[t + 1]
        x_smooth[t] = x_hat[t] + J[t] * (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P[t] + (J[t] ** 2) * (P_smooth[t + 1] - P_pred[t + 1])

    for t in range(N):
        P_cross[t] = P_smooth[t + 1] * J[t]

    return x_smooth, P_smooth, P_cross


def em_algorithm(
    y: np.ndarray, max_iter: int = 50, tol: float = 1e-5, lambda_reg: float = 0.01, b_max: float = 0.999
) -> Dict[str, float]:
    """
    Calibration EM de (A, B, C, D) sur y_train. b_max borne la vitesse de
    réversion autorisée — utile pour le diagnostic "coût en vraisemblance
    d'un B borné" (voir pairs_pipeline.py).
    """
    N_plus_1 = len(y)
    N = N_plus_1 - 1

    total_var = np.var(y)
    B = min(0.95, b_max)
    A = np.mean(y) * (1 - B)
    C2 = total_var * 0.05
    D2 = total_var * 0.50

    x0_hat = y[0]
    P0 = D2
    prev_theta = np.array([A, B, C2, D2])

    for _ in range(max_iter):
        C, D = np.sqrt(max(C2, 1e-6)), np.sqrt(max(D2, 1e-6))
        x_hat, P, x_pred, P_pred, K = kalman_filter(y, A, B, C, D, x0_hat, P0)
        x_smooth, P_smooth, P_cross = kalman_smoother(x_hat, P, x_pred, P_pred, B, K)

        S0 = np.sum(x_smooth[:N])
        S1 = np.sum(x_smooth[1:])
        Phi = np.sum(P_smooth[:N] + x_smooth[:N] ** 2)
        Psi = np.sum(P_cross[:N] + x_smooth[:N] * x_smooth[1:])

        denom_B = (N * Phi - S0 ** 2) + lambda_reg
        if abs(denom_B) < 1e-12:
            break
        B_new = np.clip((N * Psi - S1 * S0 + lambda_reg * 0.95) / denom_B, 0.50, b_max)

        denom_A = N + lambda_reg
        A_new = (S1 - B_new * S0 + lambda_reg * (np.mean(y) * (1 - 0.95))) / denom_A

        term1 = np.sum(P_smooth[1:] + x_smooth[1:] ** 2)
        S_func = term1 - 2 * A_new * S1 - 2 * B_new * Psi + N * (A_new ** 2) + 2 * A_new * B_new * S0 + (B_new ** 2) * Phi
        C2_new = max(S_func / N, 1e-6)

        T_func = np.sum(y ** 2 - 2 * y * x_smooth + P_smooth + x_smooth ** 2)
        D2_new = max(T_func / N_plus_1, 1e-6)

        curr_theta = np.array([A_new, B_new, C2_new, D2_new])
        if np.max(np.abs(curr_theta - prev_theta)) < tol:
            A, B, C2, D2 = A_new, B_new, C2_new, D2_new
            break
        A, B, C2, D2 = A_new, B_new, C2_new, D2_new
        prev_theta = curr_theta

    C, D = np.sqrt(C2), np.sqrt(D2)
    x_hat, P, _, _, _ = kalman_filter(y, A, B, C, D, x0_hat, P0)

    return {"A": A, "B": B, "C": C, "D": D, "filtered_state": x_hat, "filter_variance": P}


def parametric_bootstrap(y_train: np.ndarray, params: Dict[str, float], n_boot: int = 30) -> pd.DataFrame:
    """Bootstrap paramétrique : simule n_boot trajectoires depuis les paramètres estimés, ré-estime EM sur chacune."""
    N = len(y_train)
    A, B, C, D = params["A"], params["B"], params["C"], params["D"]
    boot_estimates = []

    for _ in range(n_boot):
        x_sim = np.zeros(N)
        y_sim = np.zeros(N)
        x_sim[0] = np.random.normal(y_train[0], C)
        y_sim[0] = x_sim[0] + np.random.normal(0, D)

        for t in range(1, N):
            x_sim[t] = A + B * x_sim[t - 1] + np.random.normal(0, C)
            y_sim[t] = x_sim[t] + np.random.normal(0, D)

        try:
            res_boot = em_algorithm(y_sim, max_iter=30)
            boot_estimates.append([res_boot["A"], res_boot["B"], res_boot["C"], res_boot["D"]])
        except Exception:
            continue

    return pd.DataFrame(boot_estimates, columns=["A", "B", "C", "D"])


def log_likelihood(y: np.ndarray, A: float, B: float, C: float, D: float, x0: float, P0: float) -> float:
    """Log-vraisemblance gaussienne du modèle, utile pour comparer différentes bornes de B (LRT informel)."""
    _, _, x_pred, P_pred, _ = kalman_filter(y, A, B, C, D, x0, P0)
    ll = 0.0
    for t in range(1, len(y)):
        S_t = P_pred[t] + D ** 2
        resid = y[t] - x_pred[t]
        ll += -0.5 * (np.log(2 * np.pi * S_t) + (resid ** 2) / S_t)
    return ll


def innovations_whiteness_test(y_train: np.ndarray, params: Dict[str, float], lags=(5, 10, 20)) -> pd.DataFrame:
    """
    Test de Ljung-Box sur les innovations (y[t] - x_pred[t]) du filtre.
    Si p-value > 0.05 à tous les lags, le modèle a extrait toute la structure
    exploitable — condition nécessaire (mais pas suffisante) pour valider
    la spécification, en complément du test OOS de cointégration.
    """
    A, B, C, D = params["A"], params["B"], params["C"], params["D"]
    _, _, x_pred, _, _ = kalman_filter(y_train, A, B, C, D, y_train[0], D ** 2)
    innovations = y_train[1:] - x_pred[1:]
    return acorr_ljungbox(innovations, lags=list(lags), return_df=True)
