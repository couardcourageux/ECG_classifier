# README

## Introduction

Le code est découpé en section, d'abord l'importation des librairies, la définition des fonctions utilisées dans le reste du code. Enfin, les différentes phases d'éxécution pour les CNN, avec 2 topologies différentes, les simpleRNN, et les LSTM.

--- 
## Construire un modèle

create CNN permet de fournir un modèle d'un cnn, avec une ou 2 couche.

create simple rnn permet de définir un réseau 2 couches avec des SimpleRnn

Create complex Rnn permet la même chose avec des lstm.

display_model et compile_model permettent d'afficher le sumary du modèle en argument, et de compiler le modèle, plus facilement, avec des pré-réglages.

train_rnn permet de lancer l'entrainement d'un rnn 

---

## Nettoyage des données

get_train_test_cnn permet fournit les x et y, train comme test, avec les  dimensions adaptées pour les Cnn .

get_train_test_rnn fait la même chose pour les Rnn.

---

## Etude

get_cnn_models et get_rnn_models fournissent une liste de résultats, d'évaluations plutot des modéles créés avec les hyper-paramètres entrés en argument.

plot_results_from_json permet là d'afficher une 'matrice' d'évaluation de l'éfficacité des hyper-paramètres selon la loss et la metrics.

