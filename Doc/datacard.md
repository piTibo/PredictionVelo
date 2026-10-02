
# Datacard — Eco-Display Place de Bretagne

## Présentation du jeu de données

**Source :** Rennes Métropole  
**Jeu de données :** eco-counter-data.csv  

Le jeu de données original regroupe les mesures issues de différents compteurs de fréquentation installés sur le territoire rennais. Dans le cadre de cette étude, les données ont été filtrées afin de conserver uniquement les relevés du compteur « Eco-Display Place de Bretagne ».

---

## Dictionnaire des données

| Variable | Type | Description |
|---|---|---|
| date | datetime | Date et heure du relevé |
| isoDate | texte | Représentation alternative de la date |
| counts | float | Nombre de passages enregistrés |
| status | float | Statut du relevé |
| ID | int | Identifiant du point de comptage |
| name | texte | Nom du point de comptage |
| sens | int | Code du sens / type de comptage |
| counter | texte | Identifiant du compteur |
| geo | coordonnées | Coordonnées géographiques du compteur |



---

# Statistiques générées automatiquement

## Dimensions

- **Nombre de lignes :** 70,565
- **Nombre de variables :** 9
- **Nombre de doublons :** 0

---

## Couverture temporelle

- **Première observation :** 2018-06-18 00:00:00+00:00
- **Dernière observation :** 2026-07-06 04:00:00+00:00
- **Durée couverte :** 2940 days 04:00:00
- **Fréquence des relevés :** 1 heure

---

## Valeurs manquantes

| Variable | Nombre | Pourcentage |
|---|---:|---:|
| date | 0 | 0.000 % |
| isoDate | 0 | 0.000 % |
| counts | 37 | 0.052 % |
| status | 37 | 0.052 % |
| ID | 0 | 0.000 % |
| name | 0 | 0.000 % |
| sens | 0 | 0.000 % |
| counter | 0 | 0.000 % |
| geo | 0 | 0.000 % |


---

## Analyse de la variable `counts`

| Statistique | Valeur |
|---|---:|
| Nombre de valeurs | 70,528 |
| Moyenne | 93.06 |
| Écart-type | 89.08 |
| Minimum | 0 |
| Q1 (25 %) | 16 |
| Médiane (50 %) | 76 |
| Q3 (75 %) | 140 |
| Maximum | 852 |

### Valeurs atypiques selon le z-score

| Seuil | Nombre de valeurs | Pourcentage |
|---|---:|---:|
| abs(z) > 3 | 995 | 1.41 % |
| abs(z) > 4 | 173 | 0.25 % |
| abs(z) > 5 | 65 | 0.09 % |


Ces seuils permettent d'identifier des observations statistiquement éloignées de la moyenne. Elles ne doivent cependant pas être considérées automatiquement comme erronées : des comptages élevés peuvent correspondre à de réels pics de fréquentation. De plus, la distribution de counts étant asymétrique, le z-score doit être interprété avec prudence.

---

## Qualité des données

37 observations présentent une valeur manquante simultanément pour counts et status. Leur analyse temporelle montre qu'elles ne sont pas réparties aléatoirement. Huit correspondent au changement annuel vers l'heure d'été (dernier dimanche de mars, de 2019 à 2026). Les 29 autres forment une période continue du 21 juin 2024 à 00:00 UTC au 22 juin 2024 à 04:00 UTC, suggérant une interruption ponctuelle de la collecte. La cause exacte de cette interruption ne peut pas être déterminée à partir du jeu de données seul.

---

## Projets de Machine Learning envisagés

### Projet réalisable 1
La ville de Rennes cherche à proposer des stands de réparation de vélos gratuits à des endroits stratégiques afin de sensibiliser les cyclistes à l'entretien de leur vélo.
À partir de l'historique des comptages, un modèle pourrait prédire la fréquentation cycliste en fonction du jour et de l'heure afin d'identifier les créneaux où un stand de réparation pourrait toucher le plus grand nombre de cyclistes.
Ce modèle pourrait par la suite être enrichi avec d'autres données, comme la météo, les jours fériés ou les vacances scolaires.
### Projet réalisable 2
La ville de Rennes voudrait informer les différentes entreprises effectuant des livraisons auprès des commerces du centre-ville des créneaux les plus adaptés pour réaliser leurs livraisons. En effet, les véhicules de livraison ne peuvent pas toujours stationner de façon optimale sur la chaussée et peuvent occasionnellement empiéter sur les aménagements cyclables.
À partir des données de comptage, un modèle pourrait prédire les périodes de faible fréquentation cycliste et ainsi identifier les créneaux pendant lesquels les livraisons occasionneraient le moins d'interactions avec les cyclistes.

### Projet difficilement réalisable
Il pourrait être intéressant de proposer aux familles des créneaux horaires plus adaptés pour circuler à vélo avec des enfants, en cherchant notamment à éviter les périodes de forte fréquentation.
Cependant, la fréquentation cycliste ne constitue pas à elle seule une mesure du niveau de sécurité. Le dataset ne contient notamment aucune information sur les accidents, le trafic automobile, la vitesse des véhicules ou les caractéristiques des infrastructures cyclables.
Ce projet nécessiterait donc de croiser ces données avec d'autres sources avant de pouvoir proposer des créneaux sur un véritable critère de sécurité.

---

## Limites du jeu de données

Les données proviennent de compteurs installés au niveau d'aménagements cyclables. Les cyclistes n'empruntant pas la zone couverte par le dispositif de comptage, notamment ceux circulant sur la chaussée ou sur les trottoirs, peuvent ne pas être comptabilisés. Les valeurs observées ne représentent donc pas nécessairement l'intégralité du trafic cycliste à proximité du compteur.

Les habitudes observées à cet endroit ne sont pas nécessairement représentatives de l'ensemble de Rennes.

le dataset ne contient pas directement des facteurs susceptibles d'influencer fortement la fréquentation (météo, jours fériés, vacances, événements, travaux, etc.). Cela limite notamment les performances et l'interprétation d'un futur modèle prédictif.
