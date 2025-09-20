import streamlit as st
from ultralytics import YOLO
import numpy as np, cv2, os, requests
from PIL import Image

# Ruta local donde guardaremos el modelo
DEFAULT_MODEL = "models/last.pt"

# URL directa del asset en tu Release público (reemplaza con la tuya)
MODEL_URL = "https://github.com/<TU_USUARIO>/<TU_REPO>/releases/download/v1.0/last.pt"

@st.cache_resource
def load_model(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        st.warning("Descargando modelo… (sólo la primera vez)")
        with requests.get(MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
    return YOLO(path)
