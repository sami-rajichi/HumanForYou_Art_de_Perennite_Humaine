# Analyse Stratégique pour Réduire le Turnover chez HumanForYou

## Project Overview

L'entreprise pharmaceutique indienne HumanForYou, avec environ 4000 employés, cherche à réduire son taux de rotation annuel d'environ 15%. Cette initiative vise à atténuer les retards dans les projets, réduire les coûts de recrutement et optimiser la formation des nouveaux employés. Notre rôle est d'analyser les données pour identifier les causes du turnover et développer des modèles pour améliorer la rétention des employés.

## Objectif Principal du Projet

L'objectif principal est d'identifier les principaux facteurs de rotation du personnel et de développer des modèles analytiques. Ces modèles fourniront des pistes d'amélioration pour inciter les employés à rester chez HumanForYou, visant ainsi à optimiser la rétention du personnel.

## Plan

### 1. Prétraitement des Données
   - Analyse des ensembles de données pré-traitées.
   - Équilibrage des données avec la technique SMOTE.

### 2. Exploration des Données
   - Analyse des tendances et des corrélations.
   - Identification des facteurs potentiels de rotation.

### 3. Modélisation et Analyse
   - Comparaison de plusieurs modèles (logistic regression, naive bayes, decision tree, random forest, gradient boosting, svm).
   - Sélection du modèle optimal (Random Forest) et affinage avec gridsearchcv ou randomized search.

### 4. Evaluation du Modèle
   - Mesure de la précision du modèle final.
   - Interprétation des résultats.

### 5. Documentation du Code
   - Commentaires détaillés dans chaque fichier de code.
   - Explication des choix de conception, des paramètres, et des résultats obtenus.

### 6. Rapport Final
   - Présentation des résultats de l'analyse.
   - Recommandations pour améliorer la rétention des employés.

## Dataset

### Ensemble de Données Pré-traitées des Employés

Cet ensemble de données comprend plusieurs fichiers CSV, notamment :
- **new_employee_survey.csv :** Réponses traitées aux enquêtes des anciens employés.
- **new_general_data.csv :** Données générales pré-traitées sur l'ensemble des employés.
- **new_manager_survey.csv :** Réponses traitées aux enquêtes des gestionnaires.
- **in_out_time.csv :** Données pré-traitées sur les heures d'arrivée et de départ.

Ces données offrent une vue complète sur l'expérience des employés, l'intégration, et les informations relatives au temps de travail.

## Analysis and Modeling

Le jeu de données a été équilibré avec SMOTE. Nous avons utilisé plusieurs modèles (logistic regression, naive bayes, decision tree, random forest, gradient boosting, svm). Random forest a montré une précision de plus de 96%. Ensuite, nous avons affiné le modèle avec gridsearchcv ou randomized search, atteignant une précision de 99%.

## Code Documentation

Le code d'implémentation est documenté en français pour assurer une compréhension claire. Voici un aperçu des principales étapes :
- **Preprocessing :** Prétraitement des données, équilibrage avec SMOTE.
- **Data Exploration :** Analyse des tendances et des corrélations.
- **Model Selection :** Comparaison de plusieurs modèles.
- **Model Evaluation :** Mesure de la précision du modèle final.

Chaque fichier de code contient des commentaires détaillés expliquant les choix de conception, les paramètres, et les résultats obtenus.

## Contributeurs

- [Sami RAJICHI](https://www.linkedin.com/in/sami-rajichi/)
- [Mazen BOUJEZZA](https://www.linkedin.com/in/mazen-boujezza-677560199/)
- [Wiem MEDIMAGH](https://www.linkedin.com/in/wiem-medimagh-b8310a213/)

## License

Ce projet est sous licence [Apache License 2.0](LICENSE).
