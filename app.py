# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np, cv2, os, requests, json
import pandas as pd
from PIL import Image

# ===================== Config de página =====================
st.set_page_config(page_title="Detector de enfermedades de cacao", page_icon="🌿", layout="centered")
st.title("🌿 Detector de enfermedades de cacao basado en un modelo YOLO")
st.markdown(
    "Sube una imagen de una planta/mazorcas de cacao y el modelo detectará objetos y te mostrará un reporte.\n\n"
    "**Nota:** si usas el modelo por defecto `yolov11m.pt`, mostrará clases COCO. "
    "Para ver *healthy/monilia/fitofthora*, usa tu peso entrenado (`last.pt` o `best.pt`)."
)

# ===================== Parámetros del modelo =====================
DEFAULT_MODEL = "models/best.pt"   # cambia a "models/best.pt" si quieres
MODEL_URL = "https://github.com/jorgedelgado13/Examen-Final/releases/download/v2.0/best.pt"
CLASSES_JSON = "models/classes.json"  # {"0":"healthy_cob","1":"monilia_cob","2":"fitofthora_cob"}

@st.cache_resource
def load_model(path: str, url: str | None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if url and not os.path.exists(path):
        st.info("Descargando modelo… (solo la primera vez)")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    if not os.path.exists(path):
        st.warning(f"No se encontró `{path}`. Cargando modelo por defecto `yolov11m.pt`.")
        return YOLO("yolov11m.pt")
    return YOLO(path)

model = load_model(DEFAULT_MODEL, MODEL_URL)

# Nombres de clases desde el modelo o desde classes.json
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
    img = np.array(image_pil)  # RGB; YOLO hará resize a imgsz
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
    infected = counts.get("monilia_cob", 0) + counts.get("fitofthora_cob", 0)
    pct_inf = (infected / total * 100) if total else 0.0

    # Diccionario “resumen”
    resumen = {**counts, "total": total}
    if "monilia_cob" in counts or "fitofthora_cob" in counts:
        resumen["pct_infected"] = round(pct_inf, 1)

    # ---- Tabla (DataFrame) ----
    rows = [{"clase": k, "conteo": v} for k, v in sorted(counts.items(), key=lambda x: x[0])]
    rows.append({"clase": "total", "conteo": total})
    if "pct_infected" in resumen:
        rows.append({"clase": "pct_infected", "conteo": f"{resumen['pct_infected']}%"})
    df = pd.DataFrame(rows)

    return plotted_rgb, resumen, df

if run and image is not None:
    vis, summary, df = infer(image)
    st.image(vis, caption="Detecciones", use_column_width=True)

    st.subheader("📊 Reporte (tabla)")
    st.dataframe(df, use_container_width=True)

    # Descargas
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name="reporte_cacao.csv", mime="text/csv")

    json_str = json.dumps(summary, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Descargar JSON", data=json_str, file_name="reporte_cacao.json", mime="application/json")

    st.subheader("🧾 Resumen (JSON para copiar)")
    st.code(json_str, language="json")

    # ===================== OpenAI al final =====================
    st.subheader("🧠 Recomendaciones del experto cacaotero (OpenAI)")

    only_healthy = (summary.get("total", 0) > 0) and \
                   all(k in ("healthy_cob", "total", "pct_infected") for k in summary.keys()) and \
                   summary.get("healthy_cob", 0) == summary.get("total", 0)

    if only_healthy:
        st.success("✅ Aprobado: todas las mazorcas aparecen sanas en esta imagen.")
    else:
        use_ai = st.toggle("Consultar a OpenAI para condiciones a revisar y posibles tratamientos", value=True)
        if use_ai:
            # ----- Lee API key y modelo desde Secrets o entorno -----
            OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
            OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

            if not OPENAI_API_KEY:
                st.error("Falta configurar `OPENAI_API_KEY` en los *Secrets* de Streamlit o en variables de entorno.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_API_KEY)

                    system_msg = (
                        "Eres un experto cacaotero con 20 años de experiencia en manejo de enfermedades "
                        "del cacao (moniliasis/Moniliophthora roreri y fitóftora/Phytophthora spp.). "
                        "Das recomendaciones prácticas y priorizas bioseguridad, manejo integrado y "
                        "enfoque preventivo. Responde en español claro y conciso."
                    )
                    user_msg = (
                        "Con base en el siguiente reporte del detector de imágenes, indica qué condiciones "
                        "debería revisar para evitar la propagación y sugiere tratamientos o medidas "
                        "recomendadas si corresponde. Si no hay evidencia de enfermedad, indica que está aprobado.\n\n"
                        f"REPORTE_JSON:\n{json_str}\n\n"
                        "Incluye una sección de: 1) Verificaciones en campo (clima, sombreamiento, ventilación, "
                        "higiene de herramientas, manejo de residuos), 2) Manejo cultural (poda, densidad, "
                        "recolección de mazorcas enfermas), 3) Manejo químico/biológico (si aplica), "
                        "4) Señales de alerta para re-inspección."
                    )

                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                    advice = resp.choices[0].message.content
                    st.markdown(advice)
                except Exception as e:
                    st.error(f"Error consultando OpenAI: {e}")

st.caption("Sugerencia: para maximizar *recall* de enfermas, prueba con conf 0.20–0.25.")
