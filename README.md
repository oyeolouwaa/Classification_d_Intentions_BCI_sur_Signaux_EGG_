# 🧠 Classification d'Intentions BCI sur Signaux EEG

Projet de Brain-Computer Interface (BCI) visant à analyser, filtrer et classifier des signaux électroencéphalographiques (EEG) pour la reconnaissance d'intentions motrices (Motor Imagery).

---

## 🚀 À propos du projet
Ce projet implémente un pipeline complet de traitement de données neurophysiologiques en Python (via la bibliothèque `MNE`) couplé à une interface de démonstration web interactive. 

L'objectif est d'identifier différents états mentaux ou intentions à partir de l'activité cérébrale enregistrée sur plusieurs canaux :
* **T0** : État de repos / Baseline
* **T1** : Imagerie motrice (ex. : main gauche)
* **T2** : Imagerie motrice (ex. : main droite)

---

## 🛠️ Technologies & Stack Technique
* **Langage :** Python 3.14 / JavaScript
* **Traitement du signal :** `MNE-Python`, `NumPy`, `SciPy`
* **Visualisation :** `Matplotlib`, `Chart.js`
* **Interface Web (Frontend) :** HTML5, Tailwind CSS, JavaScript (pour la démo interactive)

---

## 📂 Structure du Projet

```text
Classification_d_Intentions_BCI_sur_Signaux_EGG/
│
├── data/                    # Dossier contenant les fichiers de données brutes (.edf)
├── src/                     # Scripts modulaires (prétraitement, features, modèles)
├── main.py                  # Script principal d'exécution et de filtrage MNE
├── index.html               # Interface web de démonstration en temps réel
├── requirements.txt         # Dépendances du projet
└── README.md                # Documentation du projet