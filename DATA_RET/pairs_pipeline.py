"""
pairs_pipeline.py

Orchestrateur du screening de paires cointégrées + calibration Kalman/EM.
Suit la même logique que main.py : lit les Parquet déjà produits dans
data/raw/ (par data_collector.py), écrit les résultats dérivés dans
data/processed/pairs/ — jamais l'inverse, convention raw/processed du projet.

Prérequis : data_collector.py doit avoir déjà téléchargé tous les tickers
de SECTOR_UNIVERSE dans data/raw/<TICKER>.parquet avant de lancer ce module.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from cointegration import run_full_screening
from kalman import em_algorithm, parametric_bootstrap, innovations_whiteness_test
from data_collector import fetch_ohlcv, save_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_PAIRS_DIR = Path("data/processed/pairs")

# Univers sectoriel — étendre librement, doit correspondre aux tickers
# déjà collectés dans data/raw/ par data_collector.py.
SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "Staples": ["KO", "PEP", "PG", "CL", "COST", "WMT", "KMB", "CLX"],
    "Banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC"],
    "Energy_Majors": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Payment_Rails": ["V", "MA", "AXP", "PYPL"],
    "Semis": ["NVDA", "AMD", "AVGO", "QCOM", "TXN", "MU"],
    "Airlines": ["DAL", "UAL", "AAL", "LUV"],
    "Pharma": ["PFE", "MRK", "ABBV", "BMY", "LLY"],
    "Insurance": ["TRV", "ALL", "PGR", "CB"],
}

TRAIN_FRAC = 0.70
MIN_R2 = 0.30
ALPHA_EG = 0.05
ALPHA_FDR = 0.10
N_BOOT = 30


def load_log_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Charge les Close depuis data/raw/<TICKER>.parquet (déjà produits par
    data_collector.py), les convertit en log-prix, aligne sur les dates
    communes. Tickers manquants sur disque : loggés en warning et ignorés
    (même logique de tolérance que data_collector.py pour les échecs réseau).
    """
    series_dict = {}
    for ticker in tickers:
        path = RAW_DIR / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Fichier manquant pour %s (%s) — ticker ignoré. "
                            "Lancer data_collector.py au préalable.", ticker, path)
            continue
        df = pd.read_parquet(path)
        series_dict[ticker] = np.log(df["Close"])

    if not series_dict:
        raise FileNotFoundError("Aucun ticker chargé — vérifier data/raw/ et SECTOR_UNIVERSE.")

    log_prices = pd.concat(series_dict, axis=1).dropna(how="any")
    logger.info("Log-prix chargés : %d tickers, %d dates communes", len(series_dict), len(log_prices))
    return log_prices


def run_kalman_on_pair(log_prices: pd.DataFrame, y_ticker: str, x_ticker: str, beta: float) -> Dict:
    """Calibre Kalman/EM sur le spread cointégré (y - beta*x), avec bootstrap et diagnostic Ljung-Box."""
    spread_full = log_prices[y_ticker].to_numpy() - beta * log_prices[x_ticker].to_numpy()
    dates_full = log_prices.index

    split = int(len(spread_full) * TRAIN_FRAC)
    y_tr, y_te = spread_full[:split], spread_full[split:]

    params = em_algorithm(y_tr)
    boot_df = parametric_bootstrap(y_tr, params, n_boot=N_BOOT)
    ci = boot_df.quantile([0.025, 0.975]).T
    lb_test = innovations_whiteness_test(y_tr, params)

    return {
        "pair": f"{y_ticker}/{x_ticker}", "beta": beta,
        "A": params["A"], "B": params["B"], "C": params["C"], "D": params["D"],
        "B_ci_low": ci.loc["B", 0.025], "B_ci_high": ci.loc["B", 0.975],
        "innovations_bruit_blanc": bool((lb_test["lb_pvalue"] > 0.05).all()),
        "dates_train": dates_full[:split], "dates_test": dates_full[split:],
        "filtered_state_train": params["filtered_state"],
    }


def run_pairs_pipeline(start: str = None, end: str = None) -> None:
    """
    Point d'entrée appelé depuis main.py.

    start, end : bornes optionnelles (format 'YYYY-MM-DD') pour restreindre
    la fenêtre temporelle des log-prix chargés depuis data/raw/ avant le
    screening. Si omis, toute l'historique disponible en local est utilisé.

    Si des tickers de SECTOR_UNIVERSE sont absents de data/raw/ et que
    start/end sont fournis, ils sont automatiquement téléchargés via
    data_collector.py avant le chargement (évite d'avoir à lancer un script
    séparé pour peupler data/raw/ avec l'univers sectoriel).
    """
    PROCESSED_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    all_tickers = sorted({t for group in SECTOR_UNIVERSE.values() for t in group})

    missing_tickers = [t for t in all_tickers if not (RAW_DIR / f"{t}.parquet").exists()]
    if missing_tickers:
        if start is None or end is None:
            raise FileNotFoundError(
                f"{len(missing_tickers)} ticker(s) de SECTOR_UNIVERSE absent(s) de data/raw/ "
                f"({missing_tickers[:5]}{'...' if len(missing_tickers) > 5 else ''}) et aucun "
                "start/end fourni pour les télécharger automatiquement. Fournir start et end, "
                "ou lancer data_collector.py sur ces tickers au préalable."
            )
        logger.info(
            "%d ticker(s) manquant(s) dans data/raw/ — téléchargement automatique via "
            "data_collector.py sur [%s, %s]...", len(missing_tickers), start, end,
        )
        raw_data = fetch_ohlcv(missing_tickers, start=start, end=end)
        if raw_data:
            save_raw(raw_data, raw_dir=RAW_DIR)
        still_missing = [t for t in missing_tickers if t not in raw_data]
        if still_missing:
            logger.warning(
                "%d ticker(s) toujours indisponible(s) après tentative de téléchargement : %s",
                len(still_missing), still_missing,
            )

    log_prices = load_log_prices(all_tickers)

    if start is not None or end is not None:
        before = len(log_prices)
        log_prices = log_prices.loc[start:end]
        logger.info(
            "Filtrage de la fenêtre temporelle [%s, %s] : %d -> %d dates",
            start, end, before, len(log_prices),
        )
        if log_prices.empty:
            raise ValueError(
                f"Aucune date dans la fenêtre [{start}, {end}] — vérifier que "
                "data_collector.py a bien téléchargé des données sur cette période."
            )

    logger.info("Lancement du screening Engle-Granger + FDR + validation OOS...")
    screening_df, valid_pairs = run_full_screening(
        log_prices, SECTOR_UNIVERSE,
        train_frac=TRAIN_FRAC, min_r2=MIN_R2, alpha_eg=ALPHA_EG, alpha_fdr=ALPHA_FDR,
    )

    screening_path = PROCESSED_PAIRS_DIR / "screening_results.parquet"
    screening_df.to_parquet(screening_path)
    logger.info("Résultats du screening sauvegardés : %s", screening_path)

    if valid_pairs.empty:
        logger.warning(
            "Aucune paire n'a passé les 4 critères (EG, R², FDR, validation OOS). "
            "Élargir SECTOR_UNIVERSE ou la fenêtre temporelle avant de relancer."
        )
        return

    logger.info("%d paire(s) validée(s) — lancement Kalman/EM :", len(valid_pairs))
    kalman_summary = []
    for _, row in valid_pairs.iterrows():
        res = run_kalman_on_pair(log_prices, row["y"], row["x"], row["beta"])
        logger.info(
            "  %s : A=%.6f B=%.4f (IC95%%=[%.4f, %.4f]) C=%.6f D=%.6f innovations_blanches=%s",
            res["pair"], res["A"], res["B"], res["B_ci_low"], res["B_ci_high"],
            res["C"], res["D"], res["innovations_bruit_blanc"],
        )
        kalman_summary.append({k: v for k, v in res.items()
                                if k not in ("dates_train", "dates_test", "filtered_state_train")})

        pair_filename = res["pair"].replace("/", "_") + "_kalman.parquet"
        pd.DataFrame({
            "date": res["dates_train"],
            "filtered_state": res["filtered_state_train"],
        }).to_parquet(PROCESSED_PAIRS_DIR / pair_filename)

    pd.DataFrame(kalman_summary).to_parquet(PROCESSED_PAIRS_DIR / "kalman_summary.parquet")
    logger.info("Pipeline paires terminé. Résultats dans %s", PROCESSED_PAIRS_DIR)


if __name__ == "__main__":
    run_pairs_pipeline()