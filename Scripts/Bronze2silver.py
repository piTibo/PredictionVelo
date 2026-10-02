from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BRONZE_PATH = Path("../Data/Bronze/eco-counter-data.csv")
SILVER_PATH = Path("../Data/Silver/silver_eco-counter-data.csv")
DOC_PATH = Path("../Doc/bronze_to_silver.md")

COLUMNS_TO_KEEP = ["date", "counts"]


# ============================================================
# Chargement
# ============================================================

def load_bronze(path: Path) -> pd.DataFrame:
    """Charge les données Bronze sans les modifier."""
    return pd.read_csv(path, delimiter=";")


# ============================================================
# Transformation Bronze -> Silver
# ============================================================

def transform_to_silver(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    df = df.copy()

    stats = {
        "rows_bronze": len(df),
        "invalid_dates": 0,
        "invalid_counts": 0,
        "negative_counts": 0,
        "duplicate_dates": 0,
    }

    # Vérification du schéma
    missing_columns = set(COLUMNS_TO_KEEP) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires absentes : {missing_columns}"
        )

    # --------------------------------------------------------
    # Sélection des colonnes
    # --------------------------------------------------------

    df = df[COLUMNS_TO_KEEP].copy()

    # False par défaut : une ligne valide n'a pas été modifiée
    df["modified"] = False

    # --------------------------------------------------------
    # Conversion de la date en UTC
    # --------------------------------------------------------

    original_date = df["date"].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    invalid_date_mask = (
        df["date"].isna()
        & original_date.notna()
    )

    stats["invalid_dates"] = int(
        invalid_date_mask.sum()
    )

    df.loc[invalid_date_mask, "modified"] = True

    # Sans date, l'observation est inutilisable
    df = df.dropna(subset=["date"]).copy()

    # --------------------------------------------------------
    # Conversion de counts
    # --------------------------------------------------------

    original_counts = df["counts"].copy()

    df["counts"] = pd.to_numeric(
        df["counts"],
        errors="coerce",
    )

    invalid_counts_mask = (
        df["counts"].isna()
        & original_counts.notna()
    )

    stats["invalid_counts"] = int(
        invalid_counts_mask.sum()
    )

    df.loc[invalid_counts_mask, "modified"] = True

    # --------------------------------------------------------
    # Valeurs négatives
    # --------------------------------------------------------

    negative_mask = df["counts"] < 0

    stats["negative_counts"] = int(
        negative_mask.sum()
    )

    df.loc[negative_mask, "counts"] = pd.NA
    df.loc[negative_mask, "modified"] = True

    # --------------------------------------------------------
    # Doublons temporels
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=["date"],
        keep="first",
    )

    stats["duplicate_dates"] = int(
        duplicate_mask.sum()
    )

    df = df.loc[~duplicate_mask].copy()

    # --------------------------------------------------------
    # Tri
    # --------------------------------------------------------

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Statistiques finales
    # --------------------------------------------------------

    stats["rows_silver"] = len(df)

    stats["missing_counts"] = int(
        df["counts"].isna().sum()
    )

    stats["modified_rows"] = int(
        df["modified"].sum()
    )

    if not df.empty:

        stats["date_min"] = df["date"].min()
        stats["date_max"] = df["date"].max()

        time_diff = df["date"].diff()

        stats["gaps_over_1h"] = int(
            (time_diff > pd.Timedelta(hours=1)).sum()
        )

        stats["gaps_over_24h"] = int(
            (time_diff > pd.Timedelta(hours=24)).sum()
        )

    return df, stats


# ============================================================
# Documentation
# ============================================================

def generate_documentation(
    stats: dict,
    output_path: Path,
) -> None:

    execution_time = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    markdown = f"""# Lineage des données — Bronze → Silver

## 1. Objectif

Ce document décrit la transformation du dataset Bronze contenant
les données brutes de comptage cycliste vers le dataset Silver
utilisé pour la modélisation.

Ce document est généré automatiquement lors de l'exécution du
pipeline Bronze → Silver.

Date d'exécution : **{execution_time}**

---

## 2. Transformations appliquées

### Sélection des variables

Le dataset source contient les données de comptage ainsi que
plusieurs métadonnées relatives au capteur.

Le dataset étudié ne concernant qu'un seul capteur, ces métadonnées
sont constantes et ne sont pas utiles à la modélisation.

Les seules variables conservées sont :

- `date`
- `counts`

Une variable `modified` est ajoutée afin de tracer les observations
ayant nécessité une correction.

Les autres colonnes, notamment `isoDate`, `status`, `ID`, `name`,
`sens`, `counter` et `geo`, sont supprimées.

### Dates

La colonne `date` est convertie au format `datetime` et normalisée
en UTC.

Cette normalisation permet de disposer d'une référence temporelle
unique indépendamment des changements d'heure été/hiver.

Les dates impossibles à interpréter sont considérées comme invalides.
Une observation sans date valide est supprimée.

### Comptages

La colonne `counts` est convertie en valeur numérique.

Un comptage négatif étant impossible, toute valeur négative est
considérée comme invalide et remplacée par une valeur manquante.

### Valeurs manquantes

Les valeurs manquantes de `counts` sont conservées dans le Silver.

Elles peuvent notamment correspondre à une absence de mesure ou à
une panne du capteur.

Aucune imputation n'est réalisée lors du passage Bronze → Silver.
Une éventuelle stratégie d'imputation appartient au pipeline de
préparation des données propre au modèle.

### Doublons

L'unicité temporelle est contrôlée à partir de la colonne `date`.

En cas de doublon, seule la première observation est conservée.

### Tri

Le dataset Silver est trié chronologiquement selon `date`.

---

# 3. Log de transformation

## Volumétrie

| Indicateur | Valeur |
|---|---:|
| Lignes Bronze | {stats["rows_bronze"]} |
| Lignes Silver | {stats["rows_silver"]} |
| Lignes modifiées | {stats["modified_rows"]} |

## Qualité des données

| Contrôle | Nombre détecté |
|---|---:|
| Dates invalides | {stats["invalid_dates"]} |
| Comptages non numériques | {stats["invalid_counts"]} |
| Comptages négatifs | {stats["negative_counts"]} |
| Valeurs `counts` manquantes | {stats["missing_counts"]} |
| Dates dupliquées | {stats["duplicate_dates"]} |

## Continuité temporelle

| Contrôle | Nombre détecté |
|---|---:|
| Intervalles > 1 heure | {stats.get("gaps_over_1h", 0)} |
| Intervalles > 24 heures | {stats.get("gaps_over_24h", 0)} |

## Période couverte

- Première observation : `{stats.get("date_min", "N/A")}`
- Dernière observation : `{stats.get("date_max", "N/A")}`

---

## 4. Schéma Silver

| Variable | Type | Description |
|---|---|---|
| `date` | datetime UTC | Date et heure de l'observation |
| `counts` | numérique | Nombre de passages cyclistes |
| `modified` | booléen | Indique si l'observation a été corrigée |

---

*Document généré automatiquement par le pipeline Bronze → Silver.*
"""

    # Création du dossier de documentation si nécessaire
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        markdown,
        encoding="utf-8",
    )


# ============================================================
# Pipeline principal
# ============================================================

def main() -> None:

    # Chargement Bronze
    bronze = load_bronze(BRONZE_PATH)

    # Transformation
    silver, stats = transform_to_silver(bronze)

    # Création du dossier Silver
    SILVER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Export du dataset
    silver.to_csv(
        SILVER_PATH,
        index=False,
    )

    # Génération automatique de la documentation
    generate_documentation(
        stats,
        DOC_PATH,
    )

    print("Pipeline Bronze -> Silver terminé.")
    print(f"Dataset : {SILVER_PATH}")
    print(f"Documentation : {DOC_PATH}")


if __name__ == "__main__":
    main()