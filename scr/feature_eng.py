"""
features.py

Module de feature engineering pour QuantPipe.

Toutes les fonctions sont PURES : entrée -> sortie, sans effet de bord
(aucune lecture/écriture disque, aucun état global). Cela facilite les
tests unitaires (voir tests/test_features.py) et la réutilisation dans
différents contextes (notebook, pipeline batch, service temps réel).

Convention : ce module transforme les données "raw" en données "processed".
On ne mélange jamais les deux dans les mêmes dossiers (voir data_collector.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Calcule les log-returns d'une série de prix : r_t = ln(P_t / P_{t-1}).

    Les log-returns sont préférés aux rendements simples car additifs dans
    le temps (utile pour l'agrégation multi-période) et plus symétriques
    (un +x% suivi d'un -x% en log revient exactement au point de départ).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins la colonne `price_col`, indexé par date.
    price_col : str, default "Close"
        Nom de la colonne de prix à utiliser.

    Returns
    -------
    pd.Series
        Série des log-returns, alignée sur l'index de `df`. Le premier
        élément est NaN (pas de P_{t-1} disponible) — décision volontaire :
        c'est à l'appelant de dropna() si besoin, la fonction ne masque
        jamais silencieusement une perte de donnée.
    """
    if price_col not in df.columns:
        raise KeyError(f"Colonne '{price_col}' absente du DataFrame (colonnes disponibles : {list(df.columns)}).")

    prices = df[price_col].astype(float)
    log_returns = np.log(prices / prices.shift(1))
    log_returns.name = f"log_return_{price_col}"
    return log_returns


def compute_simple_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Calcule les rendements simples : r_t = (P_t - P_{t-1}) / P_{t-1}.

    Fournie à titre de comparaison pédagogique avec compute_log_returns
    (voir notebook stats_analysis.ipynb pour la comparaison empirique).
    """
    if price_col not in df.columns:
        raise KeyError(f"Colonne '{price_col}' absente du DataFrame (colonnes disponibles : {list(df.columns)}).")

    prices = df[price_col].astype(float)
    simple_returns = prices.pct_change()
    simple_returns.name = f"simple_return_{price_col}"
    return simple_returns


def rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    Calcule la volatilité réalisée glissante (écart-type glissant des rendements).

    Parameters
    ----------
    returns : pd.Series
        Série de rendements (log-returns recommandés — voir compute_log_returns).
    window : int, default 20
        Taille de la fenêtre glissante (en jours de trading). 20 ≈ 1 mois.
    annualize : bool, default True
        Si True, annualise la volatilité en multipliant par sqrt(252)
        (nombre approximatif de jours de trading par an).

    Returns
    -------
    pd.Series
        Volatilité glissante, alignée sur l'index de `returns`.
        Les `window - 1` premières valeurs sont NaN (fenêtre incomplète).
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    vol.name = f"rolling_volatility_{window}d" + ("_annualized" if annualize else "")
    return vol


def rolling_correlation(
    returns_1: pd.Series,
    returns_2: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Calcule la corrélation glissante entre deux séries de rendements.

    Utile pour observer le phénomène de "correlation breakdown" : la
    corrélation entre actifs n'est pas stable dans le temps et tend à
    augmenter fortement en période de crise (voir notebook pour un exemple
    sur 2008/2020).

    Parameters
    ----------
    returns_1, returns_2 : pd.Series
        Séries de rendements à corréler. Doivent partager un index commun
        (dates) — un realign interne est effectué automatiquement.
    window : int, default 60
        Taille de la fenêtre glissante (en jours de trading). 60 ≈ 3 mois.

    Returns
    -------
    pd.Series
        Corrélation glissante de Pearson, alignée sur l'intersection des
        deux index.
    """
    aligned = pd.concat([returns_1, returns_2], axis=1, join="inner")
    aligned.columns = ["r1", "r2"]

    corr = aligned["r1"].rolling(window=window).corr(aligned["r2"])
    corr.name = f"rolling_correlation_{window}d"
    return corr


def rolling_vs_expanding_volatility(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Compare volatilité glissante (rolling) et volatilité cumulative (expanding).

    - rolling : ne considère que les `window` dernières observations
      (fenêtre qui "glisse" dans le temps, oublie le passé lointain).
    - expanding : considère TOUTES les observations depuis le début
      (fenêtre qui grandit, ne perd jamais d'information mais réagit
      de plus en plus lentement aux changements récents).

    Returns
    -------
    pd.DataFrame
        Colonnes ["rolling", "expanding"], alignées sur l'index de `returns`.
    """
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    expanding_vol = returns.expanding(min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    return pd.DataFrame({"rolling": rolling_vol, "expanding": expanding_vol})


if __name__ == "__main__":
    # Petit exemple d'utilisation autonome (python features.py), sans dépendance
    # réseau : charge un fichier Parquet déjà téléchargé par data_collector.py.
    import pandas as pd

    df = pd.read_parquet("data/raw/AAPL_1d.parquet")

    log_returns = compute_log_returns(df)
    simple_returns = compute_simple_returns(df)
    vol_20d = rolling_volatility(log_returns, window=20)

    print("Log-returns (5 dernières lignes) :")
    print(log_returns.tail())
    print("\nVolatilité annualisée sur 20 jours (5 dernières lignes) :")
    print(vol_20d.tail())