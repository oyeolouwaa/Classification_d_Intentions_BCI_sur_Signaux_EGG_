from flask import Flask, render_template, jsonify
import mne
import numpy as np

app = Flask(__name__)

print("Chargement et traitement du fichier EDF...")
# Assure-toi que ton fichier .edf est bien dans un dossier data/ ou ajuste le chemin
raw = mne.io.read_raw_edf(r"eegmmidb-1.0.0\S001\S001R04.edf", preload=True)

# Application de ton filtre passe-bande (8 - 30 Hz)
raw.filter(8.0, 30.0, fir_design='firwin')

# Récupération des données du premier canal en microvolts
data_channel = raw[0, :][0][0] * 1e6 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/eeg-data')
def eeg_data():
    start_idx = np.random.randint(0, len(data_channel) - 40)
    chunk = data_channel[start_idx : start_idx + 40].tolist()
    
    return jsonify({
        'signal': chunk,
        'intent': np.random.choice(["T0 : Repos", "T1 : Imagerie Main Gauche", "T2 : Imagerie Main Droite"]),
        'probabilities': [
            int(np.random.randint(50, 85)),
            int(np.random.randint(5, 30)),
            int(np.random.randint(5, 30))
        ]
    })

@app.route('/api/frequency-data')
def frequency_data():
    # Fréquences cibles (ex: de 5 Hz à 34 Hz couvrant les bandes Mu et Beta)
    frequencies = list(range(5, 35))
    # Simulation de la puissance spectrale centrée sur les rythmes cérébraux
    power_spectrum = [np.exp(-((f-10)**2)/15) * 40 + np.random.uniform(2, 8) for f in frequencies]
    
    return jsonify({
        'frequencies': frequencies,
        'power': power_spectrum
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)