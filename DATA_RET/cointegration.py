"""
cointegration.py

Fonctions pures de screening de paires cointégrées (méthode Engle-Granger).
Aucun effet de bord, aucun I/O — suit la même convention que features.py.
Toutes les fonctions opèrent sur des np.ndarray / pd.DataFrame déjà en mémoire ;
la lecture depuis data/raw/ et l'écriture vers data/processed/ sont gérées
par pairs_pipeline.py.
"""

import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def is_integrated_order1(series: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Vérifie qu'une série log-prix est I(1) (racine unitaire non rejetée par ADF).
    Pré-requis théorique d'Engle-Granger : chercher une cointégration entre
    deux séries n'a de sens que si chacune est individuellement non-stationnaire.
    """
    adf_p = adfuller(series, autolag="AIC")[1]
    return adf_p > alpha


def filter_integrated_universe(log_prices: pd.DataFrame, alpha: float = 0.05) -> List[str]:
    """Retourne la liste des tickers I(1), en loggant ceux exclus."""
    integrated, excluded = [], []
    for ticker in log_prices.columns:
        if is_integrated_order1(log_prices[ticker].to_numpy(), alpha=alpha):
            integrated.append(ticker)
        else:
            excluded.append(ticker)
    if excluded:
        logger.warning("Tickers exclus (stationnaires en niveau) : %s", excluded)
    return integrated


def engle_granger_test(y_series: np.ndarray, x_series: np.ndarray) -> Dict[str, float]:
    """
    Teste la cointégration entre y et x :
    - régression OLS log(y) = alpha + beta*log(x) pour estimer le hedge ratio
    - test d'Engle-Granger officiel (statsmodels.coint), dont les valeurs
      critiques sont valides même quand beta est lui-même estimé
    - ADF complémentaire sur les résidus OLS
    """
    X = sm.add_constant(x_series)
    ols_res = sm.OLS(y_series, X).fit()
    beta_hat = ols_res.params[1]
    r2 = ols_res.rsquared

    eg_stat, eg_pvalue, eg_crit = coint(y_series, x_series)
    resid = y_series - X @ ols_res.params
    adf_resid_p = adfuller(resid, autolag="AIC")[1]

    return {
        "beta": beta_hat, "r2": r2,
        "eg_stat": eg_stat, "eg_pvalue": eg_pvalue,
        "eg_crit_5pct": eg_crit[1],
        "adf_resid_pvalue": adf_resid_p,
    }


def screen_sector_pairs(
    log_prices_train: pd.DataFrame,
    sector_universe: Dict[str, List[str]],
    integrated_tickers: List[str],
) -> pd.DataFrame:
    """
    Teste toutes les paires INTRA-secteur (pas cross-sector, pour limiter
    le nombre de tests sans justification économique). Retourne un DataFrame
    trié par p-value croissante, une ligne par paire testée.
    """
    results = []
    for sector, tickers in sector_universe.items():
        candidates = [t for t in tickers if t in integrated_tickers]
        if len(candidates) < 2:
            continue
        for t1, t2 in itertools.combinations(candidates, 2):
            y_ = log_prices_train[t1].to_numpy()
            x_ = log_prices_train[t2].to_numpy()
            res = engle_granger_test(y_, x_)
            results.append({"secteur": sector, "y": t1, "x": t2, **res})

    if not results:
        logger.warning("Aucune paire testable (univers trop restreint après filtre I(1)).")
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("eg_pvalue").reset_index(drop=True)


def apply_fdr_correction(screening_df: pd.DataFrame, alpha_fdr: float = 0.10) -> pd.DataFrame:
    """
    Correction Benjamini-Hochberg sur les p-values du screening.
    Contrôle le taux de fausses découvertes plutôt que le risque global
    d'une seule erreur (moins conservateur que Bonferroni, plus adapté
    à un screening exploratoire multi-paires).
    """
    if screening_df.empty:
        return screening_df
    reject, pvals_bh, _, _ = multipletests(
        screening_df["eg_pvalue"].to_numpy(), alpha=alpha_fdr, method="fdr_bh"
    )
    out = screening_df.copy()
    out["eg_pvalue_bh"] = pvals_bh
    out["significatif_fdr"] = reject
    n_survivors = int(reject.sum())
    logger.info("Correction FDR (alpha=%.2f) : %d/%d paires survivent", alpha_fdr, n_survivors, len(out))
    return out


def validate_oos_cointegration(
    log_prices_test: pd.DataFrame, y_ticker: str, x_ticker: str, beta: float, alpha: float = 0.05
) -> Dict[str, float]:
    """
    Reconstruit le spread sur les données de test AVEC LE BETA FIGÉ
    (jamais ré-estimé) et teste sa stationnarité hors échantillon.
    C'est le test décisif : une cointégration in-sample qui ne survit
    pas ici est un artefact de surajustement, pas un signal exploitable.
    """
    spread_test = log_prices_test[y_ticker].to_numpy() - beta * log_prices_test[x_ticker].to_numpy()
    adf_p_oos = adfuller(spread_test, autolag="AIC")[1]
    try:
        kpss_p_oos = kpss(spread_test, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p_oos = np.nan
    return {
        "adf_pvalue_oos": adf_p_oos,
        "kpss_pvalue_oos": kpss_p_oos,
        "spread_std_oos": float(np.std(spread_test)),
        "cointegration_confirmee_oos": adf_p_oos < alpha,
    }


def run_full_screening(
    log_prices: pd.DataFrame,
    sector_universe: Dict[str, List[str]],
    train_frac: float = 0.70,
    min_r2: float = 0.30,
    alpha_eg: float = 0.05,
    alpha_fdr: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline de screening complet : filtre I(1) -> Engle-Granger -> filtre R²
    -> correction FDR -> validation OOS. Retourne (screening_df complet,
    valid_pairs_df ne contenant que les paires ayant passé TOUS les critères).
    """
    split_idx = int(len(log_prices) * train_frac)
    log_train, log_test = log_prices.iloc[:split_idx], log_prices.iloc[split_idx:]

    integrated = filter_integrated_universe(log_train)
    screening_df = screen_sector_pairs(log_train, sector_universe, integrated)
    if screening_df.empty:
        return screening_df, screening_df

    screening_df = apply_fdr_correction(screening_df, alpha_fdr=alpha_fdr)

    candidates = screening_df[
        (screening_df["eg_pvalue"] < alpha_eg)
        & (screening_df["r2"] > min_r2)
        & (screening_df["significatif_fdr"])
    ].copy()

    oos_rows = []
    for _, row in candidates.iterrows():
        oos_res = validate_oos_cointegration(log_test, row["y"], row["x"], row["beta"])
        oos_rows.append(oos_res)
    if oos_rows:
        oos_df = pd.DataFrame(oos_rows, index=candidates.index)
        candidates = pd.concat([candidates, oos_df], axis=1)
        valid_pairs = candidates[candidates["cointegration_confirmee_oos"]].reset_index(drop=True)
    else:
        valid_pairs = candidates

    logger.info(
        "Screening terminé : %d paires testées, %d passent EG+R²+FDR, %d confirmées OOS",
        len(screening_df), len(candidates), len(valid_pairs),
    )
    return screening_df, valid_pairs
