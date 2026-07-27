"""
data_collector.py

Module de collecte de données OHLCV (Open-High-Low-Close-Volume) pour QuantPipe.

Responsabilités :
    - Récupérer des données de marché via yfinance, avec gestion robuste des
      erreurs réseau (retry + backoff exponentiel).
    - Sauvegarder les données brutes (raw) en Parquet, sans aucune transformation.
    - Ne jamais planter le pipeline sur un ticker en échec : logguer et continuer.

Ce module ne fait AUCUNE transformation statistique (log-returns, volatilité, etc.) :
c'est le rôle de features.py. On respecte ici la séparation raw / processed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger("quantpipe.data_collector")


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure un logging propre pour l'ensemble du pipeline (à appeler une
    seule fois, typiquement dans main.py). Volontairement pas de `print`
    dans les modules métier.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _fetch_single_ticker_with_retry(
    ticker: str,
    start: str,
    end: str,
    max_retries: int,
    base_backoff: float,
) -> pd.DataFrame | None:
    """
    Récupère un unique ticker avec retry + backoff exponentiel.

    Retourne None si toutes les tentatives échouent (aucune exception
    n'est propagée à l'appelant : c'est la responsabilité de ce helper
    d'absorber les erreurs réseau transitoires).
    """
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
            )

            if df is not None and not df.empty:
                # Aplatit les colonnes multi-index éventuelles (yfinance >= 0.2)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df

            # DataFrame vide : pas forcément une erreur réseau (ticker delisted,
            # période sans données) -> on retente quand même par sécurité.
            logger.debug("Tentative %d/%d pour '%s' : DataFrame vide.", attempt + 1, max_retries, ticker)

        except Exception as exc:  # noqa: BLE001 - on veut absolument tout capturer ici
            logger.warning(
                "Tentative %d/%d pour '%s' a échoué : %s",
                attempt + 1,
                max_retries,
                ticker,
                exc,
            )

        # Backoff exponentiel avant la prochaine tentative (sauf après la dernière)
        if attempt < max_retries - 1:
            wait_time = base_backoff ** attempt
            time.sleep(wait_time)

    return None


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = 4,
    base_backoff: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """
    Récupère les données OHLCV pour une liste de tickers, avec retries.

    Parameters
    ----------
    tickers : list[str]
        Liste de tickers à récupérer (ex. ["AAPL", "MSFT", "GOOG"]).
    start : str
        Date de début au format "YYYY-MM-DD".
    end : str
        Date de fin au format "YYYY-MM-DD".
    max_retries : int, default 4
        Nombre maximal de tentatives par ticker avant abandon.
    base_backoff : float, default 2.0
        Base du backoff exponentiel en secondes (attente = base_backoff ** i).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionnaire {ticker: DataFrame OHLCV}. Les tickers ayant échoué après
        toutes les tentatives, ou ayant renvoyé des données vides, sont
        simplement absents du dictionnaire (pas d'exception levée) — le
        pipeline continue avec les tickers valides restants.
    """
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df = _fetch_single_ticker_with_retry(ticker, start, end, max_retries, base_backoff)

        if df is None or df.empty:
            logger.warning(
                "Ticker '%s' : aucune donnée exploitable après %d tentatives — "
                "ignoré (possible delisting, erreur API, ou ticker invalide).",
                ticker,
                max_retries,
            )
            continue

        results[ticker] = df
        logger.info("Ticker '%s' : %d lignes récupérées (%s → %s).", ticker, len(df), start, end)

    if not results:
        logger.error("Aucun ticker n'a pu être récupéré sur %d demandés.", len(tickers))

    return results


def save_raw(data: dict[str, pd.DataFrame], raw_dir: str | Path) -> None:
    """
    Sauvegarde les DataFrames OHLCV bruts en Parquet, un fichier par ticker.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionnaire {ticker: DataFrame OHLCV}, tel que retourné par fetch_ohlcv.
    raw_dir : str | Path
        Répertoire de destination (créé s'il n'existe pas). Convention :
        data/raw/ — jamais mélangé avec les données transformées (data/processed/).
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    for ticker, df in data.items():
        out_file = raw_path / f"{ticker}.parquet"
        df.to_parquet(out_file)
        logger.info("Sauvegardé : %s (%d lignes).", out_file, len(df))


if __name__ == "__main__":
    configure_logging()

    tickers = ["AAPL", "MSFT", "GOOG"]
    raw_data = fetch_ohlcv(tickers, start="2023-01-01", end="2024-01-01")
    save_raw(raw_data, raw_dir="data/raw")