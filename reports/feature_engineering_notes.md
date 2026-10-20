## Conclusions de l'ingénierie des variables

- **Variables les plus corrélées au résultat** : `diff_gd` (+0.30), `diff_venue_points` (+0.28) et `diff_points` (+0.28). Les différences entre l'équipe à domicile et l'équipe à l'extérieur (forme, buts, différence de buts) sont les plus informatives.
- **Corrélations modestes** : la plus forte atteint 0.30 en valeur absolue et la médiane des variables est de 0.19. Aucune variable ne détermine à elle seule l'issue d'un match, ce qui est cohérent avec la part de hasard propre au football.
- **Test anti-fuite** : **0 écart** sur 300 matchs tirés au hasard : les moyennes recalculées à la main sur les matchs précédents sont identiques à celles du fichier. Aucune information du match à prédire (ni future) n'est utilisée.
- **Conséquence pour la suite** : un modèle ne pourra améliorer la référence que modestement. Il faudra le juger sur la log-loss et sur des données chronologiquement postérieures à l'entraînement.
