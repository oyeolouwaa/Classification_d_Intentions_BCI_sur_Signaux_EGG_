import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP

# 1. Téléchargement et chargement automatique et sécurisé par MNE
print("Téléchargement/Chargement des fichiers EDF...")
paths = mne.datasets.eegbci.load_data(subjects=[1], runs=[4, 8, 12])

# On lit directement les chemins renvoyés par la fonction MNE
raw_files = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in paths]
raw = mne.concatenate_raws(raw_files)

print("----------> RAW ", raw)

# 2. Nettoyage du signal (Filtre passe-bande Mu/Beta : 8-30 Hz)
raw.filter(8., 30., fir_design='firwin')

# 3. Récupération des événements (stimuli)
events, event_id = mne.events_from_annotations(raw)

# 4. Extraire des fenêtres de temps autour des stimuli (de 1s à 2s après la consigne)
epochs = mne.Epochs(raw, events, event_id, tmin=1.0, tmax=2.0, proj=True, baseline=None, preload=True, verbose=False)

# ==========================================
# VISUALISATION 2 : LES EPOCHS
# ==========================================
epochs.plot_image(picks='eeg', combine='mean')

# Récupération des données d'époques pour le CSP : forme (n_epochs, n_channels, n_times)
X = epochs.get_data()
y = epochs.events[:, -1]  # Les labels (classes de mouvements)

print(f"Dimensions de X (epochs, canaux, temps) : {X.shape}")
print(f"Dimensions de y (labels) : {y.shape}")

# ==========================================
# VISUALISATION 3 : LA MATRICE X et y
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique A : Signal temporel pour 1 seul essai (Canal 0)
axes[0].plot(X[0, 0, :])  
axes[0].set_title("Signal temporel pour 1 seul essai (Canal 0)")
axes[0].set_xlabel("Échantillons temporels")
axes[0].set_ylabel(r"Amplitude ($\mu V$)")

# Graphique B : Distribution des classes y
unique_classes, counts = np.unique(y, return_counts=True)
axes[1].bar([str(c) for c in unique_classes], counts, color=['skyblue', 'salmon'])
axes[1].set_title("Répartition des classes dans y (Labels)")
axes[1].set_xlabel("Code de classe")
axes[1].set_ylabel("Nombre d'essais")

plt.tight_layout()
plt.show()

# 5. Pipeline de Machine Learning de pointe pour BCI : CSP + LDA
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

pipeline = Pipeline([
    ('CSP', csp),
    ('LDA', lda)
])

# 6. Évaluation rigoureuse par Validation Croisée (5-Fold)
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print("========================================")
print(f"Précision moyenne du modèle BCI : {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
print("========================================")



"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP

# 1. Chargement des données (attention : 'subject' au singulier)
path = mne.datasets.eegbci.load_data(subjects=1, runs=[4, 8, 12])

raw_files = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in path]
raw = mne.concatenate_raws(raw_files)

raw.plot(n_channels=10, duration=10.0, scalings='auto')

# 2. Nettoyage du signal (Filtre passe-bande Mu/Beta : 8-30 Hz)
raw.filter(8., 30., fir_design='firwin')

# 3. Récupération des événements (stimuli)
events, event_id = mne.events_from_annotations(raw)

# 4. Extraire des fenêtres de temps autour des stimuli (de 1s à 2s après la consigne)
epochs = mne.Epochs(raw, events, event_id, tmin=1.0, tmax=2.0, proj=True, baseline=None, preload=True, verbose=False)
epochs.plot_image(picks=['C3', 'C4'], combine='mean')


# Récupération directe des données d'époques pour le CSP : forme (n_epochs, n_channels, n_times)
X = epochs.get_data()
y = epochs.events[:, -1]  # Les labels (classes de mouvements)

print(f"Dimensions de X (epochs, canaux, temps) : {X.shape}")
print(f"Dimensions de y (labels) : {y.shape}")


# Visualisons graphiquement à quoi ressemble la matrice X (les essais vs le temps/canaux)
# et la distribution des classes y (les labels).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique A : Une heatmap représentant les signaux de la première époque (canal 0)
axes[0].plot(X[0, 0, :])  # Canal 0, toutes les valeurs temporelles de l'essai 0
axes[0].set_title("Signal temporel pour 1 seul essai (Canal 0)")
axes[0].set_xlabel("Échantillons temporels")
axes[0].set_ylabel("Amplitude ($\mu V$)")

# Graphique B : Distribution des classes y (vérifier l'équilibre des classes)
unique_classes, counts = np.unique(y, return_counts=True)
axes[1].bar([str(c) for c in unique_classes], counts, color=['skyblue', 'salmon'])
axes[1].set_title("Répartition des classes dans y (Labels)")
axes[1].set_xlabel("Code de classe (ex: 2 = main gauche, 3 = main droite)")
axes[1].set_ylabel("Nombre d'essais")

plt.tight_layout()
plt.show()



# 5. Pipeline de Machine Learning de pointe pour BCI : CSP + LDA
# Le CSP extrait les composantes spatiales optimales pour l'imagination motrice
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

pipeline = Pipeline([
    ('CSP', csp),
    ('LDA', lda)
])

# 6. Évaluation rigoureuse par Validation Croisée (5-Fold)
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print("========================================")
print(f"Précision moyenne du modèle BCI : {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
print("========================================")

"""

"""
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = mne.datasets.eegbci.load_data (subjects = 1, runs = [4, 8, 12])

raw_files = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in path]

raw = mne.concatenate_raws(raw_files)

print ("----------> RAW " , raw)

raw.filter(8., 30., fir_design='firwin')




# -----

# Récupérer les événements (stimuli)
events, event_id = mne.events_from_annotations(raw)

# Extraire des fenêtres de temps autour des stimuli (ex: de 1s à 2s après l'affichage de la consigne)
epochs = mne.Epochs(raw, events, event_id, tmin=1.0, tmax=2.0, proj=True, baseline=None, preload=True)

# Extraire la puissance spectrale (Power Spectral Density - PSD) comme "feature"
# C'est la quantité d'énergie cérébrale dans les bandes de fréquences clés
psds, freqs = epochs.compute_psd(method='welch', fmin=8., fmax=30.0).get_data(return_freqs=True)

# `psds` a la forme : (nb_epochs, nb_canaux, nb_frequences)
# On le reshape en un tableau 2D simple pour du Machine Learning tabulaire : (nb_epochs, nb_canaux * nb_frequences)
X = psds.reshape(len(psds), -1)
y = epochs.events[:, -1]  # Les labels (classes de mouvements)

# ------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entraînement d'une Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Évaluation
y_pred = clf.predict(X_test)
print(f"Précision du modèle BCI : {accuracy_score(y_test, y_pred) * 100:.2f}%")

"""









