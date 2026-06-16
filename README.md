
Rapport de Projet : Classification de Sentiments sur les Tweets de la COVID-19 avec BERT
Ce projet a été réalisé dans le cadre du Devoir Pratique numero 3. Il propose une solution d'apprentissage profond pour classifier la polarite emotionnelle de tweets lies a la pandemie de COVID-19 en 3 classes (Negative, Neutral, Positive) a l'aide d'un modele BERT, complete par une interface utilisateur interactive Gradio.

1. Presentation du Dataset

Source : Dataset officiel Coronavirus Tweets NLP Text Classification, comprenant des tweets collectes au debut de la pandemie mondiale.
Classes (3) :

* Negative (regroupant Extremely Negative et Negative)
* Neutral
* Positive (regroupant Extremely Positive et Positive)

Statistiques et Volume :

* Volume total initial : environ 41 157 tweets.
* Volume utilise pour l'entrainement securise car l'ordinateur utilise ne suportait pas plus : 10 000 tweets.
* Decoupage des donnees : 80% pour l'entrainement (8 000 exemples) et 20% pour la validation (2 000 exemples).

Exemples du Dataset :

* Negatif : "All store shelves are completely empty, this pandemic panic is getting insane!"
* Neutre : "The government announced new social distancing measures starting tomorrow."
* Positif : "So incredibly thankful for our frontline medical workers doing an amazing job!"



2. Description du Modele et Choix Techniques

Model de base : google-bert/bert-base-uncased. Ce choix repose sur la technique du Transfer Learning. Le modele maitrise deja la syntaxe et les representations semantiques complexes de l'anglais.
Tokenizer : BertTokenizerFast associe au modele, configure avec une longueur maximale de sequence fixee a max_length=128, une troncaturation automatique (truncation=True) et un rembourrage adaptatif (padding="max_length").
Tete de Classification : Une couche lineaire PyTorch (torch.nn.Linear) a ete greffee en sortie du jeton special [CLS] (qui represente le sens global de la phrase) pour projeter l'espace latent vers nos 3 classes cibles.

   

3. Etapes de Realisation et Difficultes Rencontrees

Etapes cles :

   1. Creation de dataset.py pour charger, nettoyer (via utils.py) et tokeniser le fichier CSV en tenseurs PyTorch.
   2. Definition de la classe SentimentModel dans model.py pour encapsuler BERT et la nouvelle couche lineaire.
   3. Ecriture du pipeline d'entrainement robuste dans train.py avec calcul des metriques et sauvegarde automatique du meilleur modele (best_model.pt).
   4. Developpement de l'interface graphique de demonstration dans demo.py avec Gradio.

Difficultes rencontrees et Solutions :

* Surchauffe Materielle (Thermal Throttling) : Lors des premiers tests sur l'integralite du dataset, la carte graphique RTX 3050 a atteint sa limite critique de 86C, provoquant l'extinction brutale du PC portable.
Solution : Nous avons reduit le volume de calcul a un echantillon stratifie de 10 000 lignes, augmente le batch_size a 16 pour diviser par deux le nombre d'iterations, et limite l'entrainement a 2 epoques. La convergence de BERT etant tres rapide, le modele reste performant tout en protegeant le materiel.
* Blocage invisible sous Windows : PyTorch bloquait l'execution silencieusement juste apres le chargement du modele.
Solution : Ajout obligatoire du bloc de protection du multiprocessing "if name == 'main':" et forcage de l'argument num_workers=0 dans les DataLoaders.


4. Resultats et Metriques Finales

Metriques de l'Epoque 1  :

* Train Loss : 0,6929
* Validation Loss : 0,5245
* Validation Accuracy :80,85%
* Validation F1-Score (Weighted) :80,81%

Metriques de l'Epoque Finale (Epoque 2) :

* Train Loss : 0.3616
* Validation Loss : 0,4839
* Validation Accuracy : 84,7%
* Validation F1-Score (Weighted) : 84,6%

Preuves Visuelles(voir repository) :
(Les captures d'ecran sont stockees dans le dossier du projet aux emplacements specifies ci-dessous)

* resultats de l'entrainement
* Courbe de Loss et Barre d'entrainement (TQDM) : 
* Matrice de Confusion : 
* Interface Gradio en Fonctionnement : 

5. Captures d'ecran de la demo Gradio en fonctionnement(voir repository)

6. Instructions d'Installation et d'Execution
   2. Installation de l'environnement :
   Activez votre environnement virtuel local (venv) puis installez les modules necessaires avec la commande :
   pip install torch transformers scikit-learn tqdm gradio pandas numpy
   3. Execution de l'entrainement :
   Pour lancer l'entrainement et generer le fichier de poids optimise best_model.pt :
   python train.py
   4. Execution de la Demo Gradio :
   Une fois le fichier best_model.pt genere ou place a la racine du projet, demarrez l'interface Web locale :
   python demo.py

7. Repartition du travail au sein du binome

Afin de respecter la consigne de collaboration equitable et visible sur GitHub, les taches ont ete segmentees ainsi :

Etudiant HAFING Davy :


* Developpement, securisation Windows, et optimisation thermique du script d'entrainement principal (train.py).
* Creation des fonctions communes de traitement et de log (utils.py).
* Redaction du Readme


Etudiant YAO Metanha :

* Mise en place de la structure de donnees (dataset.py) et de l'architecture du reseau de neurones (model.py).
* Conception et developpement de l'interface graphique de demonstration utilisateur (demo.py).
* Integration du chargement des poids du modele (load_state_dict) en mode inference CPU/GPU.
* Preparation du livrable, des visualisations de resultats et tests de l'application.


MERCI POUR VOTRE ATTENTION !!!
