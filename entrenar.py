"""Genera el mock-dataset sintético del MVP, entrena el clasificador y guarda sus métricas.

Réplica del procedimiento de la PoC (láminas 6, 10, 12 y 13) con datos regenerados:
las métricas que produce este script son las del MVP y se reportan aparte de las de la PoC.
"""
import json
import random
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, make_pipeline

from asistente import RAIZ, SIEMPRE_DERIVAR, TAU, anonimizar, normalizar

SEMILLA, VARIANTES = 21, 12
random.seed(SEMILLA)

PREFIJOS = ["", "Hola, ", "Buenas tardes. ", "Buen día, me llamo Carlos Pérez. ", "Soy alumno, ",
            "Me enteré por redes. ", "Trabajo tiempo completo y ", "Una pregunta: ",
            "Soy Mariana Ruiz, aspirante. ", "Disculpe, "]
SUFIJOS = ["", " Gracias.", " Saludos, Ana López.", " Mi correo es juan.p@correo.com",
           " mi cel 81 1234 5678", " Mi matrícula es A01234567.", " Quedo atento."]
CANALES = ["correo", "whatsapp", "formulario"]


def ruido(texto: str) -> str:
    """Imita WhatsApp: sin acentos ni signos, abreviaturas y errores de dedo."""
    if random.random() < 0.5:
        texto = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    if random.random() < 0.4:
        texto = texto.replace("¿", "").replace("?", "").lower()
    if random.random() < 0.3:
        texto = texto.replace(" que ", " q ").replace("por favor", "porfa")
    if random.random() < 0.3 and len(texto) > 10:
        i = random.randrange(len(texto) - 1)
        texto = texto[:i] + texto[i + 1] + texto[i] + texto[i + 2:]
    return texto


def generar_dataset() -> pd.DataFrame:
    plantillas = pd.read_csv(RAIZ / "datos" / "plantillas.csv")
    filas = []
    for p in plantillas.itertuples():
        for _ in range(VARIANTES):
            canal = random.choice(CANALES)
            texto = random.choice(PREFIJOS) + p.texto + random.choice(SUFIJOS)
            if canal == "whatsapp" or random.random() < 0.2:
                texto = ruido(texto)
            filas.append({"plantilla": p.plantilla, "canal": canal, "intencion": p.intencion,
                          "texto": anonimizar(texto)})
    return pd.DataFrame(filas).drop_duplicates("texto")


def nuevo_modelo():
    rasgos = FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 1))),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
    ])
    return make_pipeline(rasgos, LogisticRegression(C=2, max_iter=3000))


def evaluar(modelo, prueba: pd.DataFrame) -> dict:
    X = prueba["texto"].map(normalizar)
    probas = modelo.predict_proba(X)
    clases = modelo.classes_
    pred = clases[probas.argmax(1)]
    conf = probas.max(1)
    top2 = (np.argsort(probas, 1)[:, -2:] == np.searchsorted(clases, prueba["intencion"])[:, None]).any(1)
    responde = (conf >= TAU) & ~np.isin(pred, list(SIEMPRE_DERIVAR))
    complejos = prueba["intencion"].isin(SIEMPRE_DERIVAR).to_numpy()
    return {
        "n_prueba": int(len(prueba)),
        "f1_macro": round(f1_score(prueba["intencion"], pred, average="macro"), 3),
        "accuracy": round(accuracy_score(prueba["intencion"], pred), 3),
        "roc_auc_ovr": round(roc_auc_score(prueba["intencion"], probas, multi_class="ovr", labels=clases), 3),
        "top2": round(top2.mean(), 3),
        "autoservicio": round(responde.mean(), 3),
        "acierto_de_lo_respondido": round((pred[responde] == prueba["intencion"].to_numpy()[responde]).mean(), 3),
        "casos_personales_derivados": round((~responde[complejos]).mean(), 3),
    }


if __name__ == "__main__":
    datos = generar_dataset()
    datos.to_csv(RAIZ / "datos" / "mock_dataset_mvp.csv", index=False)

    # Prueba con redacciones nunca vistas: 2 plantillas por intención fuera del entrenamiento.
    prueba_ids = datos.groupby("intencion")["plantilla"].unique().map(
        lambda ps: random.sample(list(ps), 2)).explode()
    es_prueba = datos["plantilla"].isin(prueba_ids)
    modelo = nuevo_modelo().fit(datos.loc[~es_prueba, "texto"].map(normalizar), datos.loc[~es_prueba, "intencion"])
    metricas = {"origen": "mock-dataset del MVP (regenerado), NO es el de la PoC",
                "n_total": int(len(datos)), "plantillas": int(datos["plantilla"].nunique()),
                "tau": TAU, **evaluar(modelo, datos[es_prueba])}

    # Modelo final con todos los datos.
    final = nuevo_modelo().fit(datos["texto"].map(normalizar), datos["intencion"])
    (RAIZ / "modelo").mkdir(exist_ok=True)
    joblib.dump(final, RAIZ / "modelo" / "clasificador.joblib")
    (RAIZ / "modelo" / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
