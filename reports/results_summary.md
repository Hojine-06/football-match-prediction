## Synthèse des résultats (test : saison 2024-2025, 2640 matchs)

- **Modèle retenu** (log-loss de validation la plus basse) : **logreg**.
- **Accuracy** : 49.9 % contre 42.8 % pour la référence (+7.0 points).
- **Log-loss** : 1.015 contre 1.076 pour la référence (-0.061).
- **Matchs nuls** : seulement 0.0 % des nuls réels sont repérés (F1 = 0.00). La classe « nul » reste très difficile à prédire.
- **Ligues** : la meilleure accuracy est en **Ligue 1** (54.2 %), la plus faible en **Pro League** (46.7 %). Avec environ 294 à 380 matchs par ligue, ces écarts restent fragiles.

### Limites
- Les variables reposent uniquement sur les buts et les résultats passés (pas de blessures, compositions, effectifs, ni matchs de coupe).
- Les équipes récemment promues ont peu d'historique.
- Une seule saison de test : les performances peuvent varier d'une saison à l'autre.
- Le football comporte une part importante de hasard : un gain modeste par rapport à la référence est un résultat réaliste.
