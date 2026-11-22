"""Interfaz de chat del MVP (widget del portal, láminas 14 y 19). Ejecutar: streamlit run app.py"""
import uuid

import streamlit as st

from asistente import TEMAS, atender, no_me_sirvio

st.set_page_config(page_title="Asistente de Admisiones", page_icon="🎓")
st.title("Asistente de Admisiones y Control Escolar")

if "sesion" not in st.session_state:
    st.session_state.sesion = uuid.uuid4().hex[:8]
    st.session_state.mensajes = []

# F2. Primer uso: bienvenida con los temas que cubre.
with st.chat_message("assistant"):
    st.markdown("¡Hola! Respondo al momento dudas sobre:\n\n" + "\n".join(f"- {t}" for t in TEMAS)
                + "\n\nLos casos personales (pagos, calificaciones, bajas) los turno a un asesor con folio. "
                "No escribas tu nombre, correo ni teléfono.")


def mostrar(i, m):
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])
        if m.get("fuentes"):
            st.caption("Documentos consultados: " + " · ".join(m["fuentes"]))
        if m["rol"] == "assistant" and m.get("ruta") == "respuesta":
            if m.get("folio_reporte"):
                st.info(f"Gracias. Un asesor revisará esta respuesta con el folio **{m['folio_reporte']}**.")
            elif st.button("Esta respuesta no me sirvió", key=f"nms_{i}"):
                m["folio_reporte"] = no_me_sirvio(m["id"], st.session_state.sesion)
                st.rerun()


for i, m in enumerate(st.session_state.mensajes):
    mostrar(i, m)

if pregunta := st.chat_input("Escribe tu pregunta"):
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.spinner("Buscando en los documentos oficiales…"):
        r = atender(pregunta, st.session_state.sesion)
    st.session_state.mensajes.append({"rol": "assistant", "texto": r["respuesta"], "id": r["id"],
                                      "ruta": r["ruta"], "fuentes": r["fuentes"]})
    st.rerun()
