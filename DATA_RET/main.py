from __future__ import annotations

import logging
from pathlib import Path

from data_collector import configure_logging, fetch_ohlcv, save_raw
from features import compute_log_returns, rolling_volatility

logger = logging.getLogger("quantpipe.main")

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN"]
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VOLATILITY_WINDOW = 20


def run_pipeline() -> None:
    logger.info("=== Démarrage du pipeline QuantPipe ===")

    raw_data = fetch_ohlcv(TICKERS, start=START_DATE, end=END_DATE)
    if not raw_data:
        logger.error("Aucune donnée collectée — arrêt du pipeline.")
        return

    save_raw(raw_data, raw_dir=RAW_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, df in raw_data.items():
        try:
            log_returns = compute_log_returns(df)
            volatility = rolling_volatility(log_returns, window=VOLATILITY_WINDOW)
            features_df = log_returns.to_frame().join(volatility)
            out_file = PROCESSED_DIR / f"{ticker}_features.parquet"
            features_df.to_parquet(out_file)
            logger.info("Features sauvegardées pour '%s' -> %s", ticker, out_file)
        except Exception as exc:
            logger.error("Échec du calcul des features pour '%s' : %s", ticker, exc)
            continue

    logger.info("Pipeline terminé (%d/%d tickers traités)", len(raw_data), len(TICKERS))


if __name__ == "__main__":
    configure_logging()
    run_pipeline()
