"""Núcleo del MVP: preprocesamiento, clasificador con umbral τ, RAG con cita y derivación con folio.

Sigue el diagrama de arquitectura (lámina 14): consulta -> preprocesamiento -> clasificador de
intención -> (≥ τ y consulta frecuente) asistente con RAG | (< τ o caso personal) asesor con folio
-> registro anonimizado.
"""
import csv
import os
import re
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import requests
from dotenv import load_dotenv

RAIZ = Path(__file__).parent
load_dotenv(RAIZ / ".env")

TAU = 0.50                          # umbral de derivación (lámina 10)
SIEMPRE_DERIVAR = {"caso_personal"}  # intenciones que nunca responde el asistente
SIMILITUD_MINIMA = 0.35             # por debajo, los documentos no traen la respuesta
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
REGISTRO = RAIZ / "registro" / "consultas.csv"
COLUMNAS = ["id", "fecha_hora", "sesion", "evento", "consulta_anonimizada", "intencion",
            "confianza", "ruta", "motivo", "fuentes", "folio", "latencia_ms"]

TEMAS = ["Requisitos de admisión", "Costos y becas", "Plan de estudios", "Fechas y convocatoria",
         "Certificados y constancias", "Modalidad y horarios", "Seriación de materias",
         "Revalidación y equivalencias"]

# ---------- Preprocesamiento (lámina 6, pasos 3 y 5) ----------

_PATRONES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<correo>"),
    (re.compile(r"\bA0\d{7}\b", re.I), "<matricula>"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "<telefono>"),
    (re.compile(r"\b((?i:me llamo|mi nombre es|soy|atte\.?|saludos,?))\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)"),
     r"\1 <nombre>"),
    (re.compile(r"\b((?i:me llamo|mi nombre es))\s+\w+"), r"\1 <nombre>"),
]


def anonimizar(texto: str) -> str:
    """Enmascara correos, matrículas, teléfonos y nombres antes de guardar o enviar la consulta."""
    for patron, reemplazo in _PATRONES:
        texto = patron.sub(reemplazo, texto)
    return texto


_ABREVIATURAS = {"q": "que", "xq": "porque", "pq": "porque", "tmb": "tambien", "x": "por",
                 "info": "informacion", "k": "que", "porfa": "por favor"}
_SALUDOS = re.compile(r"\b(hola|buen(o|a)s?( dias| tardes| noches)?|buen dia|saludos|gracias|"
                      r"muchas gracias|que tal|atte)\b")


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, abreviaturas expandidas, números a «num», sin saludos."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w<>\s]", " ", texto)
    texto = " ".join(_ABREVIATURAS.get(p, p) for p in texto.split())
    texto = re.sub(r"\d+", "num", texto)
    return " ".join(_SALUDOS.sub(" ", texto).split())


# ---------- Clasificador de intención ----------

_clasificador = None


def clasificar(texto_anonimizado: str):
    global _clasificador
    if _clasificador is None:
        _clasificador = joblib.load(RAIZ / "modelo" / "clasificador.joblib")
    probas = _clasificador.predict_proba([normalizar(texto_anonimizado)])[0]
    i = probas.argmax()
    return _clasificador.classes_[i], float(probas[i])


# ---------- RAG: índice de documentos oficiales ----------

_indice = None


def _cargar_indice():
    """Parte cada .md de docs/ por secciones «## » y calcula sus embeddings."""
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    fragmentos = []
    for archivo in sorted((RAIZ / "docs").glob("*.md")):
        titulo, seccion, lineas = archivo.stem, "", []
        for linea in archivo.read_text(encoding="utf-8").splitlines() + ["## "]:
            if linea.startswith("# "):
                titulo = linea[2:].strip()
            elif linea.startswith("## "):
                if "".join(lineas).strip():
                    fragmentos.append({"archivo": archivo.name, "titulo": titulo,
                                       "seccion": seccion, "texto": "\n".join(lineas).strip()})
                seccion, lineas = linea[3:].strip(), []
            else:
                lineas.append(linea)
    vectores = modelo.encode([f"{f['titulo']}. {f['seccion']}. {f['texto']}" for f in fragmentos],
                             normalize_embeddings=True)
    return modelo, fragmentos, vectores


def buscar(consulta: str, k: int = 3):
    global _indice
    if _indice is None:
        _indice = _cargar_indice()
    modelo, fragmentos, vectores = _indice
    q = modelo.encode([consulta], normalize_embeddings=True)[0]
    similitudes = vectores @ q  # coseno, porque los vectores están normalizados
    orden = similitudes.argsort()[::-1][:k]
    return [{**fragmentos[i], "similitud": float(similitudes[i])} for i in orden]


PROMPT_SISTEMA = """Eres el asistente virtual de Admisiones y Control Escolar de posgrado.
Respondes en español, en no más de 5 líneas, a aspirantes y alumnos.
Reglas:
1. Usa SOLO la información de los fragmentos que te doy. No uses conocimiento propio.
2. Termina siempre con la fuente: «Fuente: <archivo> § <sección>».
3. Si los fragmentos no contienen la respuesta completa, responde exactamente NO_SE y nada más.
4. No inventes cifras, fechas ni requisitos. Si un fragmento dice EJEMPLO, menciónalo."""


def generar_respuesta(consulta: str, fragmentos):
    contexto = "\n\n".join(f"[{f['archivo']} § {f['seccion']}]\n{f['texto']}" for f in fragmentos)
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-5.5"),
              "temperature": 0,
              "messages": [{"role": "system", "content": PROMPT_SISTEMA},
                           {"role": "user", "content": f"Fragmentos:\n{contexto}\n\nPregunta: {consulta}"}]},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------- Derivación al asesor ----------

def nuevo_folio() -> str:
    return f"ADM-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def acuse(folio: str) -> str:
    return (f"Tu consulta se turnó a un asesor de Admisiones con el folio **{folio}**. "
            "Guárdalo para dar seguimiento; el asesor te responderá en el tiempo comprometido "
            "por el área (EJEMPLO, no oficial: 1 día hábil).")


# ---------- Registro anonimizado ----------

def registrar(fila: dict):
    REGISTRO.parent.mkdir(exist_ok=True)
    nuevo = not REGISTRO.exists()
    with REGISTRO.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow({c: fila.get(c, "") for c in COLUMNAS})


# ---------- Orquestación ----------

def atender(texto: str, sesion: str = "") -> dict:
    inicio = time.perf_counter()
    consulta = anonimizar(texto)
    intencion, confianza = clasificar(consulta)
    salida = {"id": uuid.uuid4().hex[:12], "intencion": intencion, "confianza": confianza,
              "ruta": "derivada", "respuesta": "", "fuentes": [], "folio": ""}

    if intencion in SIEMPRE_DERIVAR:
        salida["motivo"] = "caso_personal"
    elif confianza < TAU:
        salida["motivo"] = "confianza_menor_a_tau"
    else:
        fragmentos = [f for f in buscar(consulta) if f["similitud"] >= SIMILITUD_MINIMA]
        if not fragmentos:
            salida["motivo"] = "sin_documento"
        elif not os.getenv("OPENROUTER_API_KEY"):
            salida["motivo"] = "sin_api_key"
        else:
            texto_llm = generar_respuesta(consulta, fragmentos)
            if "NO_SE" in texto_llm:
                salida["motivo"] = "documentos_no_responden"
            else:
                salida.update(ruta="respuesta", motivo="rag", respuesta=texto_llm,
                              fuentes=[f"{f['archivo']} § {f['seccion']}" for f in fragmentos])

    if salida["ruta"] == "derivada":
        salida["folio"] = nuevo_folio()
        salida["respuesta"] = acuse(salida["folio"])

    salida["latencia_ms"] = round((time.perf_counter() - inicio) * 1000)
    registrar({**salida, "fecha_hora": datetime.now().isoformat(timespec="seconds"),
               "sesion": sesion, "evento": "consulta", "consulta_anonimizada": consulta,
               "confianza": round(confianza, 3), "fuentes": " | ".join(salida["fuentes"])})
    return salida


def no_me_sirvio(consulta_id: str, sesion: str = "") -> str:
    """F5: registra el reporte y abre un folio para que un asesor revise la respuesta."""
    folio = nuevo_folio()
    registrar({"id": consulta_id, "fecha_hora": datetime.now().isoformat(timespec="seconds"),
               "sesion": sesion, "evento": "no_me_sirvio", "ruta": "derivada",
               "motivo": "reporte_usuario", "folio": folio})
    return folio
