import joblib
import pandas as pd
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title='Estimation de Prix de Smartphones', page_icon='📱', layout='centered'
)

# Style CSS personnalisé (thème sombre élégant)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    p, label, .stMarkdown {
        color: #9ca3af !important;
    }
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border-color: #374151 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0,6rem 1,5rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    .stAlert, div[data-testid="stNotification"] {
        background-color: #111827 !important;
        border: 1px solid #10b981 !important;
        color: #34d399 !important;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chargement du modèle et des colonnes de référence
modele = joblib.load('modele_rf_smartphone.pkl')
model_columns = joblib.load('colonnes_features.pkl')

st.title('📱 Estimation de Prix de Smartphones')
st.write(
    "Renseignez les caractéristiques techniques de l'appareil pour estimer son prix sur le marché."
)

# Saisies utilisateur
launch_year = st.number_input(
    'Année de sortie', min_value=2018, max_value=2026, value=2023, step=1
)
storage_gb = st.selectbox('Stockage (Go)', [64, 128, 256, 512])
ram_gb = st.selectbox('RAM (Go)', [4, 6, 8, 12, 16])
battery_mah = st.slider('Batterie (mAh)', 2000, 8000, 4500, step=100)

marques_disponibles = [
    'Xiaomi',
    'Vivo',
    'Apple',
    'Google',
    'OnePlus',
    'Realme',
    'Motorola',
    'Samsung',
    'Oppo',
    'Other',
]
brand = st.selectbox('Marque', marques_disponibles)

if st.button('Estimer le prix'):
  # 1. Création d'un DataFrame avec les valeurs de base
  input_data = pd.DataFrame(
      {
          'launch_year': [launch_year],
          'storage_gb': [storage_gb],
          'ram_gb': [ram_gb],
          'battery_mah': [battery_mah],
      }
  )

  # 2. Ajout dynamique des colonnes One-Hot pour correspondre exactement à l'entraînement
  for col in model_columns:
    if col.startswith('brand_nettoyee_'):
      nom_marque = col.replace('brand_nettoyee_', '')
      input_data[col] = 1 if brand == nom_marque else 0

  # S'assure que les colonnes sont dans le même ordre exact
  input_data = input_data[model_columns]

  # 3. Prédiction
  prix_estime = modele.predict(input_data)[0]
  st.success(f'💰 Prix estimé : {prix_estime:,.2f} INR')
