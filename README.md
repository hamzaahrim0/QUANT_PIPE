# QuantPipe

Pipeline de données financières dockerisé pour l'analyse quantitative.

Collecte des données OHLCV (Yahoo Finance) → feature engineering (log-returns, volatilité) → tests statistiques (stationnarité, autocorrélation, fat tails) → le tout reproductible via Docker.

## Architecture

```
QUANT_PIPE/
├──DATA_RET/
|    ├── data_collector.py       # Collecte OHLCV (yfinance) + retry/backoff + logging
|    ├── features.py              # Feature engineering pur (log-returns, volatilité, corrélations)
|    ├── main.py                  # Orchestrateur du pipeline (appelé par Docker)
├── data/
│   ├── raw/                 # Données brutes, jamais transformées (non versionné)
│   └── processed/           # Features calculées (non versionné)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── .dockerignore
```

**Principe de séparation raw / processed** : `data/raw/` ne contient que ce qui sort tel quel de Yahoo Finance, sans aucune transformation. `data/processed/` contient uniquement les features dérivées (log-returns, volatilité). On ne mélange jamais les deux — convention standard en data engineering qui permet de recalculer les features à tout moment sans re-télécharger les données.

## Concepts clés implémentés

- **Stationnarité** : un prix brut n'est pas stationnaire (tendance + variance changeante) ; on le transforme en log-returns pour stabiliser la moyenne. Vérifié empiriquement via le test ADF dans le notebook.
- **Log-returns vs rendements simples** : `r_log = ln(P_t / P_t-1)`, additifs dans le temps (utile pour l'agrégation) et symétriques (contrairement aux rendements simples).
- **Bruit de marché** : l'ACF/PACF des rendements ne montre quasiment aucune autocorrélation exploitable — cohérent avec l'hypothèse d'efficience faible des marchés.
- **Fat tails** : la kurtosis des rendements réels dépasse généralement 3 (vs loi normale), visible sur le QQ-plot — implication directe sur le risque sous-estimé par les modèles gaussiens (VaR, etc.).

## Installation locale (sans Docker)

```bash
git clone https://github.com/hamzaahrim0/QUANT_PIPE.git
cd QUANT_PIPE

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt

python main.py                              # lance le pipeline complet
jupyter notebook stats_analysis.ipynb       # exploration statistique
```

## Installation avec Docker (recommandé pour la reproductibilité)

```bash
# Build de l'image
docker build -t quantpipe .

# Lancement via docker-compose (recommandé)
docker compose up --build
```

**Pourquoi un volume est indispensable** : le conteneur est éphémère — tout ce qui est écrit à l'intérieur sans volume disparaît à sa suppression. Le volume `./data:/app/data` (défini dans `docker-compose.yml`) fait persister `data/raw/` et `data/processed/` sur la machine hôte, indépendamment du cycle de vie du conteneur.

Pour vérifier la persistance :
```bash
docker compose up --build
docker compose down
ls data/processed/   # les fichiers .parquet doivent toujours être là
```

## Pipeline — ce qu'il fait

1. **Collecte** (`data_collector.py`) : télécharge l'historique OHLCV pour une liste de tickers (`AAPL`, `MSFT`, `GOOG`, `AMZN` par défaut, configurable dans `main.py`), avec retry + backoff exponentiel en cas d'échec réseau. Un ticker en échec est ignoré (loggé en warning), le pipeline continue avec les autres.
2. **Sauvegarde brute** : chaque ticker est stocké en Parquet dans `data/raw/<TICKER>.parquet`.
3. **Feature engineering** (`features.py`) : calcul des log-returns et de la volatilité glissante annualisée (fenêtre de 20 jours) via des fonctions pures (aucun effet de bord).
4. **Sauvegarde des features** : résultat stocké dans `data/processed/<TICKER>_features.parquet`.
5. **Analyse statistique** (`stats_analysis.ipynb`) : test ADF (prix vs rendements), ACF/PACF, kurtosis et QQ-plot vs loi normale.

## Exemple de sortie du pipeline

```
=== Démarrage du pipeline QuantPipe ===
Ticker 'AAPL' : 1509 lignes récupérées (2018-01-01 → 2024-01-01).
Sauvegardé : data/raw/AAPL.parquet (1509 lignes).
Features sauvegardées pour 'AAPL' -> data/processed/AAPL_features.parquet
Pipeline terminé (4/4 tickers traités)
```

## Roadmap / prochaines étapes

- Tests unitaires (pytest) sur `features.py`
- Test de significativité de l'exposant de Hurst
- Cointégration et construction de séries mean-reverting artificielles
- Stratégie de trading mean-reverting (pairs trading)
- VaR paramétrique vs historique (lien avec les fat tails observées)

## Outils

Python · pandas · numpy · yfinance · pyarrow · statsmodels · scipy · matplotlib · Docker · Git
