# ============================================================
# PIPELINE COMPLET XGBOOST - LAG 168H + MODEL CARD
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("../Data/Silver/silver_eco-counter-data.csv")

# Nouveau dossier pour ne pas écraser le modèle 24h
OUTPUT_DIR = Path("../models/xgboost_168h")
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_PATH = OUTPUT_DIR / "xgboost_model_168h.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
MODEL_CARD_PATH = OUTPUT_DIR / "model_card.md"


TRAIN_RATIO = 0.80

FEATURES = [
    "lag_168h",
    "day_of_week",
    "matin",
    "midi",
    "soir",
    "nuit",
]

TARGET = "counts"


# ============================================================
# 2. CHARGEMENT DU SILVER
# ============================================================



def main() -> None:
    """Exécute le pipeline complet XGBoost avec un lag de 168 heures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
        errors="coerce"
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("CHARGEMENT DU DATASET")
    print("=" * 70)

    print(f"Nombre de lignes : {len(df):,}")
    print(f"Première date     : {df['date'].min()}")
    print(f"Dernière date     : {df['date'].max()}")

    print("\nColonnes :")
    print(df.columns.tolist())
    hour = df["date"].dt.hour

    df["matin"] = ((hour >= 6) & (hour < 11)).astype(int)
    df["midi"] = ((hour >= 11) & (hour < 15)).astype(int)
    df["soir"] = ((hour >= 15) & (hour < 21)).astype(int)
    df["nuit"] = ((hour >= 21) | (hour < 6)).astype(int)

    # ============================================================
    # 3. VÉRIFICATION DES COLONNES
    # ============================================================

    required_columns = [
        "date",
        "counts",
        "matin",
        "midi",
        "soir",
        "nuit",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes du Silver : {missing_columns}"
        )


    # ============================================================
    # 4. CONTRÔLES QUALITÉ
    # ============================================================

    duplicate_dates = int(
        df["date"].duplicated().sum()
    )

    invalid_dates = int(
        df["date"].isna().sum()
    )

    missing_counts = int(
        df["counts"].isna().sum()
    )

    time_diff = df["date"].diff()

    non_hourly_intervals = int(
        (
            time_diff.dropna()
            != pd.Timedelta(hours=1)
        ).sum()
    )

    print("\n" + "=" * 70)
    print("CONTRÔLES")
    print("=" * 70)

    print(f"Dates invalides                 : {invalid_dates}")
    print(f"Dates dupliquées                : {duplicate_dates}")
    print(f"Counts manquants                : {missing_counts}")
    print(f"Intervalles différents de 1 h  : {non_hourly_intervals}")


    # ============================================================
    # 5. FEATURE ENGINEERING
    # ============================================================

    # ------------------------------------------------------------
    # LAG 168 HEURES = 7 JOURS
    # ------------------------------------------------------------
    # Les données étant horaires et complètes :
    # 168 lignes = 168 heures = 7 jours

    df["lag_168h"] = df["counts"].shift(168)


    # ------------------------------------------------------------
    # Jour de la semaine
    # ------------------------------------------------------------

    date_local = df["date"].dt.tz_convert(
        "Europe/Paris"
    )

    df["day_of_week"] = (
        date_local.dt.dayofweek
    )


    # ============================================================
    # 6. DATASET POUR LE MODÈLE
    # ============================================================

    data = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    print(f"Observations Silver      : {len(df):,}")
    print(f"Observations utilisables : {len(data):,}")
    print(f"Observations perdues     : {len(df) - len(data):,}")

    print("\nVariables utilisées :")

    for feature in FEATURES:
        print(f"  - {feature}")


    # ============================================================
    # 7. SPLIT TEMPOREL 80 / 20
    # ============================================================

    split_idx = int(
        len(data) * TRAIN_RATIO
    )

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()

    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    dates_train = (
        data["date"]
        .iloc[:split_idx]
        .copy()
    )

    dates_test = (
        data["date"]
        .iloc[split_idx:]
        .copy()
    )


    print("\n" + "=" * 70)
    print("SPLIT TEMPOREL")
    print("=" * 70)

    print(
        f"Train : {len(X_train):,} observations "
        f"({len(X_train) / len(data):.1%})"
    )

    print(
        f"Test  : {len(X_test):,} observations "
        f"({len(X_test) / len(data):.1%})"
    )

    print(
        f"\nTrain : "
        f"{dates_train.min()} -> {dates_train.max()}"
    )

    print(
        f"Test  : "
        f"{dates_test.min()} -> {dates_test.max()}"
    )


    # ============================================================
    # 8. BASELINE J-7
    # ============================================================

    # La baseline prédit directement la valeur observée
    # à la même heure une semaine auparavant.

    y_pred_baseline = (
        X_test["lag_168h"]
        .to_numpy()
    )

    baseline_mae = mean_absolute_error(
        y_test,
        y_pred_baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred_baseline
        )
    )

    baseline_r2 = r2_score(
        y_test,
        y_pred_baseline
    )


    print("\n" + "=" * 70)
    print("BASELINE J-7")
    print("=" * 70)

    print(f"MAE  : {baseline_mae:.2f}")
    print(f"RMSE : {baseline_rmse:.2f}")
    print(f"R²   : {baseline_r2:.3f}")


    # ============================================================
    # 9. XGBOOST
    # ============================================================

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )


    # ============================================================
    # 10. ENTRAÎNEMENT
    # ============================================================

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_train, y_train),
            (X_test, y_test),
        ],

        verbose=False,
    )


    # ============================================================
    # 11. PRÉDICTIONS
    # ============================================================

    y_pred = model.predict(
        X_test
    )


    # ============================================================
    # 12. MÉTRIQUES XGBOOST
    # ============================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )


    print("\n" + "=" * 70)
    print("XGBOOST - LAG 168H")
    print("=" * 70)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")


    # ============================================================
    # 13. COMPARAISON BASELINE / XGBOOST
    # ============================================================

    comparison = pd.DataFrame({
        "Modèle": [
            "Baseline J-7",
            "XGBoost lag 168h",
        ],

        "MAE": [
            baseline_mae,
            mae,
        ],

        "RMSE": [
            baseline_rmse,
            rmse,
        ],

        "R²": [
            baseline_r2,
            r2,
        ],
    })

    print("\n" + "=" * 70)
    print("COMPARAISON")
    print("=" * 70)

    print(
        comparison.round(3).to_string(index=False)
    )


    # ============================================================
    # 14. AMÉLIORATION PAR RAPPORT À LA BASELINE
    # ============================================================

    mae_improvement = (
        (baseline_mae - mae)
        / baseline_mae
        * 100
    )

    rmse_improvement = (
        (baseline_rmse - rmse)
        / baseline_rmse
        * 100
    )

    print(
        f"Évolution MAE par rapport à la baseline J-7 : "
        f"{mae_improvement:+.2f} %"
    )

    print(
        f"Évolution RMSE par rapport à la baseline J-7 : "
        f"{rmse_improvement:+.2f} %"
    )


    # ============================================================
    # 15. DATAFRAME DES PRÉDICTIONS
    # ============================================================

    predictions_df = pd.DataFrame({

        "date":
            dates_test.reset_index(drop=True),

        "reality":
            y_test.reset_index(drop=True),

        "prediction":
            y_pred,

        "baseline_168h":
            y_pred_baseline,
    })


    predictions_df["residual"] = (
        predictions_df["reality"]
        - predictions_df["prediction"]
    )

    predictions_df["absolute_error"] = (
        predictions_df["residual"].abs()
    )


    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )


    # ============================================================
    # 16. CONVERGENCE
    # ============================================================

    evals_result = (
        model.evals_result()
    )

    train_rmse_history = (
        evals_result["validation_0"]["rmse"]
    )

    test_rmse_history = (
        evals_result["validation_1"]["rmse"]
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        train_rmse_history,
        label="Train"
    )

    ax.plot(
        test_rmse_history,
        label="Test"
    )

    ax.set_title(
        "Convergence du modèle XGBoost — Lag 168h"
    )

    ax.set_xlabel(
        "Nombre d'arbres"
    )

    ax.set_ylabel(
        "RMSE"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "convergence.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # 17. PRÉDICTIONS VS RÉALITÉ
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(30, 6)
    )

    ax.plot(
        dates_test,
        y_test,
        label="Réalité",
        linewidth=0.8
    )

    ax.plot(
        dates_test,
        y_pred,
        label="XGBoost lag 168h",
        linewidth=0.8,
        alpha=0.8
    )

    ax.set_title(
        "Nombre de passages cyclistes : prédiction vs réalité — Lag 168h"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Nombre de passages"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "prediction_vs_realite.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # 18. ZOOM SUR 7 JOURS
    # ============================================================

    n_zoom = min(
        24 * 7,
        len(y_test)
    )

    zoom_dates = (
        dates_test.iloc[-n_zoom:]
    )

    zoom_real = (
        y_test.iloc[-n_zoom:]
    )

    zoom_prediction = (
        y_pred[-n_zoom:]
    )


    fig, ax = plt.subplots(
        figsize=(20, 6)
    )

    ax.plot(
        zoom_dates,
        zoom_real,
        label="Réalité",
        linewidth=1.5
    )

    ax.plot(
        zoom_dates,
        zoom_prediction,
        label="XGBoost lag 168h",
        linewidth=1.5
    )

    ax.set_title(
        "Prédiction vs réalité — 7 derniers jours — Lag 168h"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Nombre de passages"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "prediction_zoom_7_days.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # 19. RÉSIDUS
    # ============================================================

    residuals = (
        y_test.to_numpy()
        - y_pred
    )


    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.3,
        s=10
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Résidus du modèle XGBoost — Lag 168h"
    )

    ax.set_xlabel(
        "Valeur prédite"
    )

    ax.set_ylabel(
        "Résidu (réalité - prédiction)"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "residus.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # 20. HISTOGRAMME DES RÉSIDUS
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.hist(
        residuals,
        bins=50
    )

    ax.set_title(
        "Distribution des résidus — Lag 168h"
    )

    ax.set_xlabel(
        "Résidu"
    )

    ax.set_ylabel(
        "Nombre d'observations"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "histogramme_residus.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # 21. IMPORTANCE DES VARIABLES
    # ============================================================

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
        name="Importance"
    )

    importance = (
        importance
        .sort_values(ascending=False)
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    importance.sort_values().plot.barh(
        ax=ax
    )

    ax.set_title(
        "Importance des variables — XGBoost lag 168h"
    )

    ax.set_xlabel(
        "Importance"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "feature_importance.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    print("\nImportance des variables :")

    print(
        importance
        .round(4)
        .to_frame()
        .to_string()
    )


    # ============================================================
    # 22. ANALYSE PAR PÉRIODE DE LA JOURNÉE
    # ============================================================

    evaluation = (
        data
        .iloc[split_idx:]
        .copy()
        .reset_index(drop=True)
    )

    evaluation["prediction"] = y_pred

    evaluation["residual"] = (
        evaluation[TARGET]
        - evaluation["prediction"]
    )


    periods = [
        "matin",
        "midi",
        "soir",
        "nuit",
    ]

    period_results_list = []


    for period in periods:

        subset = evaluation[
            evaluation[period] == 1
        ]

        if len(subset) == 0:
            continue

        period_mae = mean_absolute_error(
            subset[TARGET],
            subset["prediction"]
        )

        period_rmse = np.sqrt(
            mean_squared_error(
                subset[TARGET],
                subset["prediction"]
            )
        )

        period_r2 = r2_score(
            subset[TARGET],
            subset["prediction"]
        )

        period_results_list.append({

            "Période":
                period,

            "Observations":
                int(len(subset)),

            "MAE":
                float(period_mae),

            "RMSE":
                float(period_rmse),

            "R2":
                float(period_r2),
        })


    period_results = pd.DataFrame(
        period_results_list
    )


    print("\n" + "=" * 70)
    print("PERFORMANCES PAR PÉRIODE")
    print("=" * 70)

    print(
        period_results
        .round(3)
        .to_string(index=False)
    )


    # ============================================================
    # 23. GRAPHIQUE MAE PAR PÉRIODE
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        period_results["Période"],
        period_results["MAE"]
    )

    ax.set_title(
        "Erreur absolue moyenne selon la période — Lag 168h"
    )

    ax.set_xlabel(
        "Période"
    )

    ax.set_ylabel(
        "MAE"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "mae_par_periode.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # 24. SAUVEGARDE DU MODÈLE
    # ============================================================

    joblib.dump(
        model,
        MODEL_PATH
    )


    # ============================================================
    # 25. SAUVEGARDE DES MÉTRIQUES JSON
    # ============================================================

    metrics = {

        "model": "XGBoost",

        "lag_hours": 168,

        "lag_days": 7,

        "dataset": {

            "silver_rows":
                int(len(df)),

            "model_rows":
                int(len(data)),

            "train_rows":
                int(len(X_train)),

            "test_rows":
                int(len(X_test)),

            "first_date":
                str(df["date"].min()),

            "last_date":
                str(df["date"].max()),
        },

        "baseline_168h": {

            "MAE":
                float(baseline_mae),

            "RMSE":
                float(baseline_rmse),

            "R2":
                float(baseline_r2),
        },

        "xgboost": {

            "MAE":
                float(mae),

            "RMSE":
                float(rmse),

            "R2":
                float(r2),
        },

        "improvement_percent": {

            "MAE":
                float(mae_improvement),

            "RMSE":
                float(rmse_improvement),
        },

        "performance_by_period":
            period_results.to_dict(
                orient="records"
            ),
    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )


    # ============================================================
    # 26. TABLEAUX MARKDOWN
    # ============================================================

    period_table = (
        period_results.copy()
    )

    period_table["MAE"] = (
        period_table["MAE"].round(2)
    )

    period_table["RMSE"] = (
        period_table["RMSE"].round(2)
    )

    period_table["R2"] = (
        period_table["R2"].round(3)
    )


    period_table_md = (
        "| Période | Observations | MAE | RMSE | R² |\n"
        "|---|---:|---:|---:|---:|\n"
    )

    for _, row in period_table.iterrows():

        period_table_md += (
            f"| {row['Période']} "
            f"| {int(row['Observations'])} "
            f"| {row['MAE']:.2f} "
            f"| {row['RMSE']:.2f} "
            f"| {row['R2']:.3f} |\n"
        )


    importance_table_md = (
        "| Variable | Importance |\n"
        "|---|---:|\n"
    )

    for variable, value in importance.items():

        importance_table_md += (
            f"| {variable} "
            f"| {value:.4f} |\n"
        )


    # ============================================================
    # 27. PARAMÈTRES MODEL CARD
    # ============================================================

    params = model.get_params()

    execution_date = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


    # ============================================================
    # 28. CRÉATION DE LA MODEL CARD
    # ============================================================

    model_card = f"""
    # Model Card — Prédiction du trafic cycliste — Lag 168h

    > Document généré automatiquement le {execution_date}

    ---

    ## 1. Présentation

    Ce modèle a pour objectif de prédire le nombre de passages cyclistes
    observés par heure sur le compteur étudié.

    Le problème est traité comme un problème de **régression sur série
    temporelle**.

    Le modèle utilisé est un **XGBoost Regressor**.

    La particularité de cette expérience est l'utilisation d'un
    **lag de 168 heures**, soit une semaine.

    La variable cible est `counts`.

    ---

    ## 2. Données

    Le modèle utilise le dataset Silver produit lors de l'étape de
    préparation des données.

    ### Période disponible

    - Première observation : `{df["date"].min()}`
    - Dernière observation : `{df["date"].max()}`

    ### Volumétrie

    | Dataset | Observations |
    |---|---:|
    | Silver | {len(df):,} |
    | Après création des features | {len(data):,} |
    | Train | {len(X_train):,} |
    | Test | {len(X_test):,} |

    Les 168 premières observations sont exclues car elles ne disposent
    pas d'une valeur historique située une semaine auparavant.

    ---

    ## 3. Variables explicatives

    | Variable | Description |
    |---|---|
    | `lag_168h` | Nombre de passages observé 168 heures auparavant |
    | `day_of_week` | Jour de la semaine, de 0 (lundi) à 6 (dimanche) |
    | `matin` | Indicateur de la période du matin |
    | `midi` | Indicateur de la période du midi |
    | `soir` | Indicateur de la période du soir |
    | `nuit` | Indicateur de la période de nuit |

    Les données étant horaires et complètes, le lag de 168 heures est
    calculé en décalant la série de 168 lignes.

    **lag_168h(t) = counts(t - 168 heures)**

    168 heures correspondent à **7 jours**.

    Cette variable permet donc au modèle de connaître la fréquentation
    observée à la même position dans la série une semaine auparavant.

    ---

    ## 4. Séparation Train / Test

    La séparation est réalisée chronologiquement.

    | Jeu | Proportion |
    |---|---:|
    | Train | {TRAIN_RATIO:.0%} |
    | Test | {1 - TRAIN_RATIO:.0%} |

    ### Train

    `{dates_train.min()}` → `{dates_train.max()}`

    ### Test

    `{dates_test.min()}` → `{dates_test.max()}`

    Les données ne sont pas mélangées afin de respecter leur structure
    temporelle.

    ---

    ## 5. Baseline J-7

    Une baseline hebdomadaire est utilisée comme référence.

    Pour chaque heure, la prédiction correspond directement au nombre
    de passages observé **168 heures auparavant**, soit une semaine.

    **prédiction(t) = counts(t - 168 h)**

    ### Performances de la baseline

    | Métrique | Valeur |
    |---|---:|
    | MAE | {baseline_mae:.2f} |
    | RMSE | {baseline_rmse:.2f} |
    | R² | {baseline_r2:.3f} |

    Cette baseline est particulièrement pertinente pour le trafic
    cycliste car elle permet de tester directement l'existence d'une
    régularité hebdomadaire dans la série.

    ---

    ## 6. XGBoost

    Le modèle utilisé est `XGBRegressor`.

    ### Hyperparamètres

    | Hyperparamètre | Valeur |
    |---|---:|
    | n_estimators | {params["n_estimators"]} |
    | learning_rate | {params["learning_rate"]} |
    | max_depth | {params["max_depth"]} |
    | subsample | {params["subsample"]} |
    | colsample_bytree | {params["colsample_bytree"]} |
    | objective | {params["objective"]} |
    | random_state | {params["random_state"]} |

    ---

    ## 7. Performances

    ### Comparaison avec la baseline J-7

    | Modèle | MAE | RMSE | R² |
    |---|---:|---:|---:|
    | Baseline J-7 | {baseline_mae:.2f} | {baseline_rmse:.2f} | {baseline_r2:.3f} |
    | XGBoost lag 168h | {mae:.2f} | {rmse:.2f} | {r2:.3f} |

    ### Évolution par rapport à la baseline

    - MAE : **{mae_improvement:+.2f} %**
    - RMSE : **{rmse_improvement:+.2f} %**

    Une valeur positive signifie que XGBoost réduit l'erreur par rapport
    à la baseline J-7.

    ---

    ## 8. Convergence

    ![Convergence](figures/convergence.png)

    Le graphique représente l'évolution de la RMSE au cours de
    l'entraînement du modèle.

    ---

    ## 9. Prédictions vs réalité

    ### Ensemble du jeu de test

    ![Prédictions vs réalité](figures/prediction_vs_realite.png)

    ### Zoom sur sept jours

    ![Zoom sur 7 jours](figures/prediction_zoom_7_days.png)

    Le zoom permet d'observer la capacité du modèle à reproduire les
    variations horaires et hebdomadaires du trafic.

    ---

    ## 10. Analyse des résidus

    Le résidu est défini comme :

    **résidu = valeur réelle - valeur prédite**

    ![Résidus](figures/residus.png)

    Un résidu positif correspond à une sous-estimation.

    Un résidu négatif correspond à une surestimation.

    ### Distribution

    ![Distribution des résidus](figures/histogramme_residus.png)

    ---

    ## 11. Performances par période

    {period_table_md}

    ![MAE par période](figures/mae_par_periode.png)

    Cette analyse permet d'identifier les périodes de la journée pour
    lesquelles les prédictions sont les plus ou les moins précises.

    ---

    ## 12. Importance des variables

    {importance_table_md}

    ![Importance des variables](figures/feature_importance.png)

    L'importance des variables indique leur utilisation relative par
    XGBoost.

    Elle ne doit pas être interprétée comme une relation causale.

    ---

    ## 13. Critères de validation

    Le modèle est évalué à l'aide de :

    - la MAE ;
    - la RMSE ;
    - le R² ;
    - une baseline hebdomadaire J-7 ;
    - l'analyse de la convergence ;
    - la comparaison prédiction / réalité ;
    - l'analyse des résidus ;
    - l'analyse des performances par période ;
    - l'importance des variables.

    ---

    ## 14. Intérêt du lag 168h

    Le lag de 168 heures correspond à une semaine complète.

    Il permet de fournir au modèle une information historique située au
    même moment du cycle hebdomadaire précédent.

    Cette expérience pourra être comparée au modèle utilisant un lag de
    24 heures afin d'étudier si une information quotidienne ou
    hebdomadaire est la plus pertinente pour la série étudiée.

    ---

    ## 15. Limites

    Le modèle reste volontairement simple.

    Il ne prend notamment pas en compte :

    - la météo ;
    - les précipitations ;
    - la température ;
    - les jours fériés ;
    - les vacances scolaires ;
    - les événements locaux ;
    - les travaux ;
    - d'autres variables de saisonnalité.

    Une extension possible serait de construire un modèle utilisant
    simultanément les lags **24h et 168h**.

    ---

    ## 16. Utilisation prévue

    Ce modèle est destiné à l'étude expérimentale de la fréquentation
    cycliste et à la comparaison de différentes stratégies de
    prédiction temporelle.

    Il n'est pas destiné à une utilisation critique ou à une prise de
    décision automatisée.

    ---

    ## 17. Artefacts générés

    Le pipeline génère :

    - `xgboost_model_168h.joblib` ;
    - `metrics.json` ;
    - `predictions.csv` ;
    - `model_card.md` ;
    - les graphiques associés dans le dossier `figures`.

    ---

    *Model Card générée automatiquement lors de l'entraînement.*
    """


    # ============================================================
    # 29. EXPORT DE LA MODEL CARD
    # ============================================================

    MODEL_CARD_PATH.write_text(
        model_card.strip(),
        encoding="utf-8"
    )


    # ============================================================
    # 30. RÉCAPITULATIF
    # ============================================================

    print("\n" + "=" * 70)
    print("PIPELINE LAG 168H TERMINÉ")
    print("=" * 70)

    print(
        f"""
    Modèle :
    {MODEL_PATH}

    Métriques :
    {METRICS_PATH}

    Prédictions :
    {PREDICTIONS_PATH}

    Model Card :
    {MODEL_CARD_PATH}

    Graphiques :
    {FIGURE_DIR}
    """
    )

    print("Performances finales :")

    print(
        f"Baseline J-7     -> "
        f"MAE={baseline_mae:.2f} | "
        f"RMSE={baseline_rmse:.2f} | "
        f"R²={baseline_r2:.3f}"
    )

    print(
        f"XGBoost lag 168h -> "
        f"MAE={mae:.2f} | "
        f"RMSE={rmse:.2f} | "
        f"R²={r2:.3f}"
    )


if __name__ == "__main__":
    main()
