# Lineage des données — Bronze → Silver

## 1. Objectif

Ce document décrit la transformation du dataset Bronze contenant
les données brutes de comptage cycliste vers le dataset Silver
utilisé pour la modélisation.

Ce document est généré automatiquement lors de l'exécution du
pipeline Bronze → Silver.

Date d'exécution : **2026-10-02 08:56:49 UTC**

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
| Lignes Bronze | 70565 |
| Lignes Silver | 70565 |
| Lignes modifiées | 0 |

## Qualité des données

| Contrôle | Nombre détecté |
|---|---:|
| Dates invalides | 0 |
| Comptages non numériques | 0 |
| Comptages négatifs | 0 |
| Valeurs `counts` manquantes | 37 |
| Dates dupliquées | 0 |

## Continuité temporelle

| Contrôle | Nombre détecté |
|---|---:|
| Intervalles > 1 heure | 0 |
| Intervalles > 24 heures | 0 |

## Période couverte

- Première observation : `2018-06-18 00:00:00+00:00`
- Dernière observation : `2026-07-06 04:00:00+00:00`

---

## 4. Schéma Silver

| Variable | Type | Description |
|---|---|---|
| `date` | datetime UTC | Date et heure de l'observation |
| `counts` | numérique | Nombre de passages cyclistes |
| `modified` | booléen | Indique si l'observation a été corrigée |

---

*Document généré automatiquement par le pipeline Bronze → Silver.*
