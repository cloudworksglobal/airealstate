import sqlite3
import streamlit as st
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# ------------------------------------------------------------------
# 1. HERRAMIENTAS DEL AGENTE (Consultas a Catálogo)
# ------------------------------------------------------------------

@tool
def buscar_propiedades(zona: str, presupuesto_max: int) -> str:
    """Consulta el inventario inmobiliario según zona y presupuesto máximo."""
    inventario = [
        {"id": 101, "titulo": "Dpto 2 amb muy luminoso", "zona": "Palermo", "precio": 110000, "ambientes": 2},
        {"id": 102, "titulo": "Casa con patio y cochera", "zona": "Belgrano", "precio": 240000, "ambientes": 4},
        {"id": 103, "titulo": "Monoambiente moderno centro", "zona": "Palermo", "precio": 85000, "ambientes": 1},
        {"id": 104, "titulo": "Departamento 3 amb apto profesional", "zona": "Caballito", "precio": 130000, "ambientes": 3},
    ]
    
    resultados = [
        p for p in inventario 
        if p["zona"].lower() in zona.lower() and p["precio"] <= presupuesto_max
    ]
    
    if not resultados:
        return "No hay propiedades que coincidan con esos criterios exactos en el catálogo."
    
    return f"Inmuebles encontrados: {resultados}"

tools = [buscar_propiedades]

# ------------------------------------------------------------------
# 2. MODELO Y GRAFO DE LANGGRAPH
# ------------------------------------------------------------------

llm = ChatOllama(model="llama3.2", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot_node(state: AgentState):
    system_prompt = SystemMessage(
        content=(
            "Sos un Agente Inmobiliario virtual para el sitio web de la empresa. "
            "Atendé al usuario con amabilidad. Cuando tengas datos suficientes (zona y presupuesto), "
            "ejecutá la herramienta 'buscar_propiedades' para mostrarle opciones del catálogo real. "
            "Mantené tus respuestas breves y directas."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Construir el grafo
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# Configurar Persistencia SQLite
conn = sqlite3.connect("agente_memoria.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compilar App con Checkpointer de Memoria
app = workflow.compile(checkpointer=memory)

# ------------------------------------------------------------------
# 3. INTERFAZ WEB / WIDGET EN STREAMLIT
# ------------------------------------------------------------------

st.set_page_config(page_title="Widget Inmobiliario IA", page_icon="🏢")

st.title("🏢 Asistente Inmobiliario Inteligente")
st.caption("Agente local impulsado por LangGraph, Ollama y SQLite")

# Identificador de Sesión (Thread ID) para la memoria por cliente
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "cliente_session_1"

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# Recuperar historial previo desde SQLite si existe
state_snapshot = app.get_state(config)
if state_snapshot.values and "messages" in state_snapshot.values:
    messages = state_snapshot.values["messages"]
else:
    messages = []

# Renderizar el historial de conversación en pantalla
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif hasattr(msg, "content") and msg.content:
        # Evitar renderizar mensajes de llamadas internas a tools sin texto visible
        with st.chat_message("assistant"):
            st.write(msg.content)

# Input del Chat
if prompt := st.chat_input("Escribe tu consulta (ej: Busco dpto en Palermo hasta 120000 USD)..."):
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.write(prompt)

    # Invocación del agente
    input_message = HumanMessage(content=prompt)
    output = app.invoke({"messages": [input_message]}, config=config)

    # Mostrar respuesta del agente
    respuesta = output["messages"][-1].content
    with st.chat_message("assistant"):
        st.write(respuesta)