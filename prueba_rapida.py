"""Prueba rápida: 9 consultas de ejemplo y qué ruta toma cada una. Ejecutar: python prueba_rapida.py"""
import sys

from asistente import RAIZ, atender

sys.stdout.reconfigure(encoding="utf-8")

CONSULTAS = [
    "¿Cuáles son los requisitos para entrar a la maestría?",
    "cuanto cuesta cada materia y se puede pagar en mensualidades",
    "¿Cómo pido una constancia de estudios para la beca de mi empresa?",
    "¿Cuántas materias me pueden revalidar de otra maestría?",
    "Me cobraron dos veces la colegiatura, soy Ana Torres, mi correo es ana@mail.com",
    "No me aparece mi calificación de la materia de estadística",
    "¿Tienen estacionamiento en el campus?",
    "¿Dan descuento por convenio con mi empresa?",
    "oye y eso como va?",
]

filas = ["| # | Consulta | Intención | Confianza | Ruta | Motivo | Documentos consultados o folio |",
         "|---|---|---|---|---|---|---|"]
for n, c in enumerate(CONSULTAS, 1):
    r = atender(c, sesion="prueba")
    destino = " · ".join(r["fuentes"]) if r["ruta"] == "respuesta" else r["folio"]
    filas.append(f"| {n} | {c} | {r['intencion']} | {r['confianza']:.2f} | {r['ruta']} | {r['motivo']} | {destino} |")
    print(f"\n[{n}] {c}\n    -> {r['ruta']} ({r['motivo']}), {r['intencion']} {r['confianza']:.2f}\n    {r['respuesta']}")

(RAIZ / "resultados_prueba.md").write_text("# Resultados de la prueba rápida\n\n" + "\n".join(filas) + "\n",
                                           encoding="utf-8")
print("\nTabla guardada en resultados_prueba.md")
