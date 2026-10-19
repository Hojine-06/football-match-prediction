## Conclusions de l'analyse exploratoire

Le jeu de données contient **26791 matchs** issus de **8 championnats** et **10 saisons**.

- **Répartition des résultats** : 44.3 % de victoires à domicile, 24.7 % de nuls et 31.0 % de victoires à l'extérieur. Le modèle de référence (toujours prédire « victoire à domicile ») atteint donc environ **44.3 %** d'accuracy.
- **Avantage du terrain** : les équipes à domicile marquent en moyenne **1.56 buts** contre **1.25** à l'extérieur. La part de victoires à domicile est la plus élevée en **La Liga** (45.4 %) et la plus faible en **Serie A** (42.7 %).
- **Évolution dans le temps** : la saison 2020-2021 n'est pas disponible dans les données.
- **Buts** : la moyenne est de **2.80 buts par match**, du plus défensif (**Primeira Liga**, 2.60) au plus offensif (**Eredivisie**, 3.07).
- **Nuls** : la part de matchs nuls va de 23.3 % en **Premier League** à 26.2 % en **La Liga**. Cette classe, la moins nette statistiquement, sera la plus difficile à prédire.
- **Données disponibles** : les tirs sont renseignés à plus de 90 % pour **8 saison(s) sur 10** (taux global de valeurs présentes : 94 %). Les premières variables seront donc construites à partir des buts et des résultats, disponibles sur toute la période.
- **Implication pour la modélisation** : le modèle devra dépasser la référence d'environ 44 % d'accuracy, en gardant un oeil sur la log-loss et sur le F1 de la classe « nul », plutôt que sur l'accuracy seule.
