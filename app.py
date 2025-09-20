# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np, cv2, os, requests, json
from PIL import Image

# ===================== Config de página =====================
st.set_page_config(page_title="Detector de enfermedades de cacao", page_icon="🌿", layout="centered")
st.title("🌿 Detector de enfermedades de cacao basado en un modelo YOLO")
st.markdown(
    "Sube una imagen de una planta/mazorcas de cacao y el modelo detectará objetos y te mostrará un conteo por clase.\n\n"
    "**Nota:** si usas el modelo por defecto `yolov11m.pt`, mostrará clases generales COCO. "
    "Para ver *healthy/monilia/fitofthora*, carga tu peso entrenado (`last.pt` o `best.pt`)."
)

# ===================== Parámetros del modelo =====================
# Ruta local donde guardaremos/leeremos el modelo
DEFAULT_MODEL = "models/last.pt"   # cambia a "models/best.pt" si quieres
# URL directa al asset en tu Release público (ajusta a tu repo/tag/nombre)
# Ejemplo: https://github.com/<usuario>/<repo>/releases/download/<tag>/<archivo>
MODEL_URL = "https://github.com/jorgedelgado13/Examen-Final/releases/download/v1.0/last.pt"

# Si prefieres iniciar con YOLO COCO por defecto, coméntalo arriba y descomenta esto:
# DEFAULT_MODEL = "yolov11m.pt"
# MODEL_URL = None  # no se usa

# (Opcional) ruta a un archivo con nombres de clase si tu .pt no los trae embebidos
CLASSES_JSON = "models/classes.json"  # {"0":"healthy_cob","1":"monilia_cob","2":"fitofthora_cob"}

@st.cache_resource
def load_model(path: str, url: str | None):
    """Carga el modelo. Si no existe localmente y se provee url, lo descarga una sola vez."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if url and not os.path.exists(path):
        st.info("Descargando modelo… (solo la primera vez)")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    # Si el archivo no existe y no hay URL, cae al modelo por defecto COCO
    if not os.path.exists(path):
        st.warning(f"No se encontró `{path}`. Cargando modelo por defecto `yolov11m.pt`.")
        return YOLO("yolov11m.pt")
    return YOLO(path)

model = load_model(DEFAULT_MODEL, MODEL_URL)

# Obtener nombres de clases desde el modelo o desde classes.json
NAMES = getattr(model, "names", None)
if not NAMES or not isinstance(NAMES, dict) or len(NAMES) == 0:
    if os.path.exists(CLASSES_JSON):
        try:
            with open(CLASSES_JSON, "r", encoding="utf-8") as f:
                raw = json.load(f)
            NAMES = {int(k): v for k, v in raw.items()}
        except Exception:
            NAMES = None
    else:
        NAMES = None

# ===================== Controles de inferencia =====================
with st.sidebar:
    st.header("⚙️ Parámetros")
    conf  = st.slider("Confianza (conf)", 0.05, 0.90, 0.25, 0.05)
    iou   = st.slider("IoU", 0.30, 0.90, 0.60, 0.05)
    imgsz = st.slider("Tamaño de entrada (imgsz)", 320, 1280, 640, 64,
                      help="El modelo reducirá/ajustará internamente la imagen a este tamaño.")

# ===================== Cargar imagen =====================
file = st.file_uploader("📤 Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

run = False
image = None
if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Vista previa", use_column_width=True)
    run = st.button("🔍 Evaluar imagen")

# ===================== Inferencia =====================
def infer(image_pil: Image.Image):
    """Corre predicción, dibuja cajas y devuelve (imagen_anotada, resumen_clases)."""
    # YOLO hará el resize al imgsz solicitado internamente
    img = np.array(image_pil)  # RGB
    res = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    # Visualización (res.plot devuelve BGR)
    plotted = res.plot()
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # Conteo por clase
    counts = {}
    for b in res.boxes:
        cls = int(b.cls.item())
        name = NAMES.get(cls, str(cls)) if NAMES else str(cls)
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    # Si estás usando tu modelo de cacao, calcula % infectadas (monilia + fitofthora)
    infected = counts.get("monilia_cob", 0) + counts.get("fitofthora_cob", 0)
    pct_inf = (infected / total * 100) if total else 0.0

    resumen = {**counts, "total": total}
    # Agrega % infectadas si aplica (si existen esas clases)
    if "monilia_cob" in counts or "fitofthora_cob" in counts:
        resumen["pct_infected"] = f"{pct_inf:.1f}%"
    return plotted_rgb, resumen

if run and image is not None:
    vis, summary = infer(image)
    st.image(vis, caption="Detecciones", use_column_width=True)
    st.subheader("📊 Reporte")
    st.json(summary)

st.caption("Sugerencia: para maximizar *recall* de enfermas, prueba con conf 0.20–0.25.")
