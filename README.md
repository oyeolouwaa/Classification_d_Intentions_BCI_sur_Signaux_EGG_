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
|
├── eegmmidb-1.0.0/                     # Dossier contenant les fichiers de données brutes (.edf)
├── templates/                          # Dossier contenant l'interface web (HTML, Tailwind, Chart.js) pour visualiser le signal et les prédictions en direct.
    └── index.html                      # Interface web de démonstration en temps réel
├── main.py                             # Script principal d'exécution et de filtrage MNE 
├── app.py                              # Script principal Flask gérant le backend, le chargement du signal EEG et les routes de l'API.
├── requirements.txt                    # Dépendances du projet
└── README.md                           # Documentation du projet
