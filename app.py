import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Analizador de Calidad del Vino",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Analizador de Calidad del Vino")
st.markdown("Evalúa si un vino es **Bueno** o **Malo** según sus componentes químicos usando un modelo entrenado con Random Forest.")

# --- Cargar modelo automáticamente ---
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    try:
        mdl = joblib.load("RF_Model.sav")  # Carga directa
        return mdl, None
    except Exception as e:
        return None, str(e)

model, err = cargar_modelo()
if err:
    st.error(f"Error al cargar modelo: {err}")
elif model:
    st.success("✅ Modelo cargado exitosamente")

# --- Formulario de entrada de datos ---
st.header("📊 Ingresar valores químicos del vino")
col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")
    sulfitos = st.number_input("Sulfitos", min_value=0.0, format="%.2f")
    acidez_volatil = st.number_input("Acidez volátil", min_value=0.0, format="%.2f")
    diox_total = st.number_input("Dióxido azufre total", min_value=0.0, format="%.2f")

with col2:
    densidad = st.number_input("Densidad", min_value=0.0, format="%.5f")
    cloruros = st.number_input("Cloruros", min_value=0.0, format="%.2f")
    ph = st.number_input("pH", min_value=0.0, format="%.2f")
    acidez_fija = st.number_input("Acidez fija", min_value=0.0, format="%.2f")

with col3:
    acido_citrico = st.number_input("Ácido cítrico", min_value=0.0, format="%.2f")
    azucar = st.number_input("Azúcar residual", min_value=0.0, format="%.2f")
    diox_libre = st.number_input("Dióxido azufre libre", min_value=0.0, format="%.2f")

# --- Botón para predecir ---
if st.button("🔍 Evaluar Calidad"):
    if not model:
        st.warning("No se pudo cargar el modelo.")
    else:
        entrada = np.array([[alcohol, sulfitos, acidez_volatil, diox_total, densidad,
                             cloruros, ph, acidez_fija, acido_citrico, azucar, diox_libre]])
        
        pred = model.predict(entrada)
        clase = "👍 Bueno" if pred[0] == 1 else "👎 Malo"

        st.subheader(f"Resultado: {clase}")
