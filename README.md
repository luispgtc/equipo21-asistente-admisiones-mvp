# MVP · Asistente virtual de Admisiones y Control Escolar

Equipo 21 · Soluciones con aplicación de tecnología · Proyecto Integrador Parte 2 (Prototipado y prueba).

Un asistente de chat para **Alejandro, el aspirante ocupado**. Responde al momento las preguntas frecuentes sobre requisitos, costos, revalidación y constancias, citando el documento oficial de donde sale cada respuesta. Lo que no puede responder con seguridad lo turna a un asesor y le entrega un folio.

**Demo en línea (disponible del 22 al 29 de noviembre de 2026):** https://equipo21-asistente-admisiones.azurewebsites.net

## Integrantes

| Integrante | Rol | Correo |
|---|---|---|
| Mónica Susana Gaytán Ramírez | Product Owner y datos | A01840295@tec.mx |
| César Daniel Matías Bayona | Ingeniero de ML | A01840391@tec.mx |
| Kevin Rogelio Rodriguez Magaña | Arquitecto de solución | A01840418@tec.mx |
| Luis Pablo Guerrero Garza | Scrum Master y QA | A01840428@tec.mx |

## Cómo funciona

```
Consulta del aspirante
  -> 1. Anonimiza: correos, teléfonos, matrículas y nombres -> <correo>, <telefono>…
  -> 2. Normaliza: minúsculas, sin acentos, abreviaturas, números -> «num»
  -> 3. Clasifica la intención (9 clases) y obtiene su confianza
       ├─ caso_personal, o confianza < τ (0.50) ──────────────> DERIVA: folio + acuse
       └─ confianza ≥ τ y consulta frecuente
            -> 4. Busca los 3 fragmentos más parecidos en el documento de esa intención (o en todo docs/)
                 ├─ ninguno con similitud ≥ 0.35 ──────────────> DERIVA: folio + acuse
                 └─ 5. El LLM redacta con esos fragmentos
                      ├─ responde «NO_SE» ──────────────────────> DERIVA: folio + acuse
                      └─ RESPONDE con «Fuente: archivo § sección»
  -> 6. Registra la consulta anonimizada, la ruta y la latencia en registro/consultas.csv
Botón «Esta respuesta no me sirvió» -> registra el reporte y abre un folio para revisión
```

## Qué hace

1. **Clasificador de intención.** TF-IDF de palabras y de caracteres (3 a 5) con regresión logística, 9 intenciones y C = 2, como en la lámina 10. Decide si la consulta la responde el asistente o la turna al asesor, con el umbral **τ = 0.50**.
2. **Asistente con RAG**, cuando la confianza es ≥ τ y no se trata de un caso personal:
   - busca en el documento de `docs/` que corresponde a la intención (o en todos, si no hay uno) por similitud coseno, con embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`);
   - el LLM redacta la respuesta vía OpenRouter, sólo con los fragmentos recuperados, y **siempre cita la fuente**;
   - si los documentos no traen la respuesta, contesta que no sabe y deriva.
3. **Derivación al asesor**, cuando la confianza es < τ, la intención es `caso_personal`, ningún documento es relevante o el LLM responde que no sabe. Genera un folio `ADM-AAAAMMDD-XXXXXX` y muestra un acuse.
4. **Interfaz de chat** en Streamlit:
   - mensaje de bienvenida con los temas que cubre (F2);
   - respuesta con su fuente (F3);
   - acuse con folio (F4);
   - botón **«Esta respuesta no me sirvió»** (F5), que registra el reporte y abre un folio para revisión.
5. **Registro anonimizado** en `registro/consultas.csv`, para las pruebas con usuarios y sus KPI. Antes de guardar o de enviar nada al LLM, se enmascaran correos, teléfonos, matrículas y nombres. Se guardan la intención, la confianza, la ruta, el motivo de la derivación, el folio y la latencia.

## Cómo correrlo

```powershell
python -m venv .venv            # Python 3.12
.venv\Scripts\activate
pip install -r requirements.txt  # incluye el índice de torch para CPU
copy .env.example .env          # y pon tu OPENROUTER_API_KEY
python entrenar.py              # opcional: el modelo ya viene en modelo/
python prueba_rapida.py         # 9 consultas de ejemplo -> resultados_prueba.md
streamlit run app.py            # abre http://localhost:8501
```

La primera consulta tarda unos segundos, porque descarga y carga el modelo de embeddings.

## Cómo encaja con las láminas de arquitectura

| Bloque de la lámina 14 | En este MVP |
|---|---|
| Portal web · widget de chat | `app.py` (Streamlit) |
| API orquestadora: registra la consulta y asigna folio | `asistente.atender()` |
| Preprocesamiento: limpia, anonimiza y normaliza | `asistente.anonimizar()` y `asistente.normalizar()` (lámina 6, pasos 3 y 5) |
| Clasificador de intención, 9 intenciones, τ = 0.50 | `modelo/clasificador.joblib`, entrenado por `entrenar.py` |
| Asistente con RAG y liga a la fuente | `asistente.buscar()` y `asistente.generar_respuesta()` sobre `docs/` |
| Documentos oficiales | `docs/*.md` (**EJEMPLO, no oficial**) |
| CRM de admisiones: caso con folio | `asistente.nuevo_folio()` y `acuse()` (simulado; no hay conexión a un CRM) |
| Registro anonimizado y botón «no me sirvió» | `registro/consultas.csv` y `asistente.no_me_sirvio()` |
| Reentrenamiento mensual | `entrenar.py` se puede volver a correr; el etiquetado mensual no está automatizado |

Fuera del alcance del MVP quedan WhatsApp Business, el correo, PostgreSQL, el CRM real y MLflow (lámina 15).

## Datos y métricas: dos fuentes distintas

**Métricas de la PoC (láminas 12 y 13).** F1-macro 0.59, accuracy 0.62, ROC-AUC 0.92, 32 % de autoservicio y 97 % de derivación de casos complejos. Vienen de la PoC con el mock-dataset sintético de 1,374 consultas. Ese dataset y ese notebook **no forman parte de este repositorio**.

**Métricas de este MVP (`modelo/metricas.json`).** El MVP regenera su propio mock-dataset sintético con el mismo procedimiento:
- 90 plantillas en `datos/plantillas.csv`;
- variantes con saludos, firmas, datos personales (que se anonimizan) y ruido de WhatsApp;
- 1,053 consultas en total;
- prueba con 2 plantillas por intención que el modelo nunca vio (205 consultas).

| Métrica | MVP (mock-dataset regenerado) |
|---|---|
| F1-macro | 0.483 |
| Accuracy | 0.498 |
| ROC-AUC uno contra el resto | 0.897 |
| Top-2 | 0.698 |
| Autoservicio con τ = 0.50 | 17.6 % |
| Acierto de lo que responde | 94.4 % |
| Casos personales derivados | 100 % |

Las dos fuentes no se mezclan. Ninguna de las dos sustituye la validación con consultas reales (lámina 23).

## Prueba rápida

`python prueba_rapida.py` corre 9 consultas y guarda la tabla en `resultados_prueba.md`:
- **5 se responden con RAG:** requisitos, costos, constancias, revalidación y descuento por convenio;
- **2 casos personales se derivan:** cobro doble y calificación faltante;
- **1 consulta sin documento se deriva:** estacionamiento;
- **1 consulta ambigua se deriva por confianza < τ.**

Las respuestas del LLM pueden variar un poco entre corridas.

## Limitaciones

- Los documentos de `docs/` son **EJEMPLO, no oficiales**: las cifras son ficticias. El equipo debe sustituirlos por los documentos oficiales de Admisiones.
- La anonimización es por reglas (regex). Detecta correos, teléfonos, matrículas y nombres después de «soy», «me llamo» o «mi nombre es», pero no cualquier nombre suelto.
- El folio no se envía a un CRM. El tiempo de respuesta del acuse es un ejemplo.
- El texto que va a OpenRouter es la consulta ya anonimizada más los fragmentos de `docs/`.

## Licencia

[MIT](LICENSE) © 2026 Equipo 21. Los documentos de `docs/` son ejemplos ficticios y no representan información oficial de ninguna institución.
