import sqlite3
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# ------------------------------------------------------------------
# 1. HERRAMIENTAS Y GRAFO DE LANGGRAPH
# ------------------------------------------------------------------

@tool
def buscar_propiedades(zona: str, presupuesto_max: int) -> str:
    """Consulta el inventario inmobiliario según zona y presupuesto máximo en dólares."""
    inventario = [
        {"id": 101, "titulo": "Dpto 2 amb muy luminoso", "zona": "Palermo", "precio": 110000},
        {"id": 102, "titulo": "Casa con patio y cochera", "zona": "Belgrano", "precio": 240000},
        {"id": 103, "titulo": "Monoambiente moderno", "zona": "Palermo", "precio": 85000},
        {"id": 104, "titulo": "Piso de categoría con balcón", "zona": "Recoleta", "precio": 310000},
        {"id": 105, "titulo": "Dpto 1 amb a estrenar", "zona": "Caballito", "precio": 70000},
    ]
    resultados = [
        p for p in inventario 
        if p["zona"].lower() in zona.lower() and p["precio"] <= presupuesto_max
    ]
    return f"Inmuebles encontrados: {resultados}" if resultados else "No hay opciones disponibles para esos criterios."

tools = [buscar_propiedades]
llm = ChatOllama(model="llama3.2", temperature=0.1).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot_node(state: AgentState):
    system_prompt = SystemMessage(
        content="Sos un Agente Inmobiliario virtual de CloudWorks Real Estate. Atendé al cliente de forma breve y profesional. "
                "Cuando el cliente te dé la zona y su presupuesto máximo, ejecutá la herramienta 'buscar_propiedades'."
    )
    # Concatenar el prompt de sistema con la lista de mensajes acumulados
    full_messages = [system_prompt] + state["messages"]
    response = llm.invoke(full_messages)
    return {"messages": [response]}

# Construcción del Flujo de LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# Persistencia en base de datos SQLite local
conn = sqlite3.connect("agente_memoria.db", check_same_thread=False)
memory = SqliteSaver(conn)
agent_app = workflow.compile(checkpointer=memory)

# ------------------------------------------------------------------
# 2. CONFIGURACIÓN DE FASTAPI
# ------------------------------------------------------------------

app = FastAPI(title="Agente Inmobiliario API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        input_message = HumanMessage(content=req.message)
        
        output = agent_app.invoke({"messages": [input_message]}, config=config)
        
        # Extraer el último mensaje generado por el bot
        last_msg = output["messages"][-1].content
        if isinstance(last_msg, list):
            last_msg = str(last_msg[0]) if last_msg else ""
            
        return ChatResponse(response=str(last_msg))
    except Exception as e:
        return ChatResponse(response=f"Error en el servidor: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)