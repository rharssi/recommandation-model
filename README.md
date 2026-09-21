# Modèle de recommandation MDJS — Réseau de points de vente x jeux

Base de données relationnelle dont **toutes les valeurs de vente sont reconstruites par simulation**, avec une seule et même méthode appliquée aux 9 codes-jeu (y compris le code 4). Aucune valeur numérique n'est recopiée depuis `data_entrainement.csv` / `data_test.csv` : ces fichiers servent uniquement de **plan structurel** (quels retailers, quelles régions, quel découpage entraînement/test, quels jeux vend chaque retailer).

## Fichiers

| Fichier | Contenu |
|---|---|
| **`data_final.sql`** | **Script unique, idempotent, prêt à l'emploi** — purge, DDL, données, vues, dans l'ordre. À exécuter tel quel du début à la fin sur PostgreSQL. |
| `generate_branch_data.py` | Script Python ayant généré `data_final.sql` (dimensions, faits, éclatement par agence) |

## 1. Ce qui a changé par rapport à la version précédente

La version précédente avait une table de faits au grain **retailer × jeu** (13 384 lignes), avec une dimension produit (`jeux`) réduite à un simple code sans nom ni catégorie, et aucune notion de branche sous le retailer.

Deux niveaux ont été ajoutés, chacun sur le même principe (une table de dimension "niveau 1" qui regroupe plusieurs entités "niveau 0") :

### Dimension BRANCHE : `agences` (niveau 0) sous `points_de_vente` (niveau 1)

**Ce que veut dire "agence" ici** : un retailer (`points_de_vente`, identifié par `RETAILER_CODE`) est l'entité commerciale — le titulaire du contrat MDJS. Une **agence** est un point de vente physique concret que ce retailer exploite. Un retailer peut exploiter une ou plusieurs agences sous la même identité commerciale (ex. deux comptoirs physiques différents, même contrat).

**À savoir avant d'utiliser cette dimension** : elle est **entièrement inventée** pour cette maquette, pas dérivée d'une vraie répartition MDJS par agence :
- Le nombre d'agences par retailer est tiré au hasard : 85% des retailers en ont 1, 12% en ont 2, 3% en ont 3 (2066 agences générées pour 1795 retailers).
- Les ventes de chaque retailer sont réparties entre ses agences de façon arbitraire (poids aléatoires, loi de Dirichlet), **pas** selon une vraie performance par agence — cette donnée n'existe nulle part dans les fichiers source.
- Le grain de la table de faits `ventes_retailers` passe de retailer × jeu à **agence × jeu** : 15 403 lignes (au lieu de 13 384). Une contrainte de clé étrangère composite `(agence_id, retailer_code)` garantit qu'une vente ne peut jamais être rattachée à une agence n'appartenant pas au bon retailer.
- Chaque ligne agence × jeu a été obtenue en éclatant la ligne retailer × jeu d'origine par la méthode des plus forts restes (répartition entière exacte) : **en resommant les agences d'un retailer, on retombe exactement sur les chiffres déjà validés** (répartition par GAME_CODE, résultats de clustering, etc. — rien ne change à l'échelle retailer).

### Dimension PRODUIT : `categories_jeu` (niveau 1) au-dessus de `jeux` (niveau 0)

`jeux.libelle_jeu` (nom du jeu, ex. "Pari Basket Live") et la nouvelle table `categories_jeu` (3 catégories : *Loterie à tirage*, *Grattage*, *Pari sportif*) enrichissent la dimension produit, auparavant réduite au seul `GAME_CODE` brut sans nom ni catégorie.

**À savoir aussi** : ces noms de jeux et catégories sont **illustratifs/fictifs** — je n'ai aucune source pour le vrai catalogue commercial MDJS. Ils servent uniquement à donner une dimension produit navigable (ex. pour un visuel Power BI groupé par catégorie), pas à représenter les vrais produits MDJS.

## 2. Méthode de génération (`generate_branch_data.py`)

1. **Retailer × jeu** (identique à la version précédente) : pour chaque retailer, une ligne "ancre" tirée de sa structure d'origine fixe un ordre de grandeur plausible ; pour chaque jeu qu'il vend, une nouvelle ligne est générée par un facteur aléatoire log-normal appliqué à cet ordre de grandeur.
2. **Éclatement en agences** (nouveau) : pour chaque retailer, 1 à 3 agences sont générées (tirage 85/12/3%). Chaque ligne retailer × jeu est ensuite répartie entre ces agences :
   - colonnes de **volume** (ventes brutes/nettes, tickets vendus/annulés, trend_score, moyenne mobile) : réparties par la méthode des plus forts restes — la somme des agences redonne exactement la valeur d'origine ;
   - colonnes de **taux/comportement** (taux de forclusion, croissance, activité...) : identiques sur toutes les agences d'un même retailer, car ce ne sont pas des volumes divisibles.
3. Une vérification programmatique (avant génération du SQL) confirme que la resommation par retailer × jeu redonne exactement les valeurs d'origine.

## 3. Modèle relationnel

```
regions (niveau 2) ───< points_de_vente (niveau 1) ───< agences (niveau 0)
                                                             │
                                                             ▼
                                                      ventes_retailers >─── jeux (niveau 0) >─── categories_jeu (niveau 1)
```

- **regions** : 12 régions du Maroc.
- **categories_jeu** (niveau 1) : 3 familles de jeux (illustratives).
- **jeux** (niveau 0) : 9 codes-jeu (1, 2, 3, 4, 5, 6, 7, 9, 10), rattachés à une catégorie.
- **points_de_vente** (niveau 1) : 1795 retailers.
- **agences** (niveau 0) : 2066 agences (85% des retailers en ont 1, 12% en ont 2, 3% en ont 3).
- **ventes_retailers** (table de faits, grain agence × jeu, **15 403 lignes**).

## 4. Vues analytiques

| Vue | Grain | Rôle |
|---|---|---|
| `vue_analytique_ventes` | retailer × jeu (13 384 lignes) | Ré-agrège les agences pour rester compatible avec tout le feature engineering ci-dessous — **aucun changement de résultat** par rapport à la version précédente |
| `vue_analytique_ventes_agence` | agence × jeu (15 403 lignes) | Le détail fin, avec code/nom d'agence et catégorie de jeu |
| `feat_performance_agence` | agence | Nombre de jeux actifs, volume, taux de forclusion moyen — par agence |

Les vues de feature engineering (`feat_popularite_jeu_region`, `feat_portefeuille_retailer`, `feat_segment_performance_retailer`, `feat_recommandation_candidats`, `feat_momentum_jeu`, `feature_table_recommandation`) sont **inchangées dans leur logique** — elles s'appuient sur `vue_analytique_ventes`, qui reproduit fidèlement l'ancien grain retailer × jeu. Seul ajout : `categorie_jeu` est désormais disponible dans ces vues.

## 5. Vérifications effectuées

- Resommation exacte des agences vers le grain retailer × jeu (vérifiée en Python avant génération du SQL, puis en SQL après exécution : mêmes totaux par `GAME_CODE` au chiffre près qu'avant l'ajout des agences).
- 0 ligne orpheline sur toutes les clés étrangères, y compris la FK composite `(agence_id, retailer_code)`.
- Script testé de bout en bout sur **PostgreSQL 16** (pas seulement SQLite) : booléens corrigés en `true`/`false` (une valeur `0`/`1` brute est acceptée par SQLite/MySQL mais rejetée par PostgreSQL, qui type strictement `BOOLEAN`).
- Idempotence vérifiée sur 3 exécutions consécutives (mêmes comptages à chaque fois : 12 régions, 3 catégories, 9 jeux, 1795 retailers, 2066 agences, 15 403 ventes).

## 6. Export CSV

Pour exporter le détail le plus complet (agence × jeu, avec région, nom d'agence, jeu et catégorie) en CSV depuis PostgreSQL :

```bash
psql -d mdjs -c "\copy (SELECT * FROM vue_analytique_ventes_agence) TO 'data_final.csv' WITH (FORMAT CSV, HEADER)"
```

Pour retrouver l'ancien grain retailer × jeu (compatible avec les scripts Python existants) :

```bash
psql -d mdjs -c "\copy (SELECT * FROM vue_analytique_ventes) TO 'data_final_retailer.csv' WITH (FORMAT CSV, HEADER)"
```
