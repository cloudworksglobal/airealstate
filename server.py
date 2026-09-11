import sqlite3
import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, TypedDict, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END  # type: ignore[import-not-found]
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """Combina los mensajes del estado sin depender de un módulo opcional de LangGraph."""
    return left + right

# ------------------------------------------------------------------
# 1. BASE DE DATOS LOCAL Y TABLA DE LEADS
# ------------------------------------------------------------------

conn = sqlite3.connect("agente_memoria.db", check_same_thread=False)

def init_db():
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            thread_id TEXT PRIMARY KEY,
            nombre TEXT,
            telefono TEXT,
            zona TEXT,
            presupuesto_max INT,
            plazo_mudanza TEXT,
            score TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

init_db()

# ------------------------------------------------------------------
# 2. HERRAMIENTAS Y MODELOS DE LANGGRAPH
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
llm_extractor = ChatOllama(model="llama3.2", temperature=0.0)

# ------------------------------------------------------------------
# 3. ESTADO Y NODOS DEL GRAFO
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    lead_info: Optional[dict]

def chatbot_node(state: AgentState):
    system_prompt = SystemMessage(
        content="Sos el Agente Virtual de InmoCloud PRO. Atendé al cliente con amabilidad y brevedad. "
                "Cuando tengas la zona y el presupuesto del cliente, ejecutá la herramienta 'buscar_propiedades'. "
                "Si muestra interés en una propiedad, pedile amablemente un teléfono o WhatsApp de contacto para agendar una visita."
    )
    full_messages = [system_prompt] + state["messages"]
    response = llm.invoke(full_messages)
    return {"messages": [response]}

def lead_extractor_node(state: AgentState):
    """Nodo extractor en segundo plano que analiza la conversación y califica al lead."""
    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
    
    extraction_prompt = f"""Analiza la siguiente conversación de un cliente inmobiliario y extrae los datos en formato JSON único y nada más:
Conversación:
{conversation_text}

Debes responder ÚNICAMENTE con un JSON válido con la siguiente estructura:
{{
  "nombre": "nombre extraído o null",
  "telefono": "teléfono extraído o null",
  "zona": "zona de interés o null",
  "presupuesto_max": presupuesto numérico entero o null,
  "plazo_mudanza": "plazo mencionado o null",
  "score": "CALIENTE (si dejó teléfono o presupuesto), TIBIO (si solo consultó zonas), FRIO (si fue un saludo simple)"
}}
"""
    try:
        raw_res = llm_extractor.invoke([HumanMessage(content=extraction_prompt)])
        content = raw_res.content.strip()
        
        # Limpiar bloques de código si el LLM incluye ```json ... ```
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
            
        data = json.loads(content)
        return {"lead_info": data}
    except Exception:
        return {"lead_info": None}

# Construcción de la Topología de LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("lead_extractor", lead_extractor_node)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")
workflow.add_edge("chatbot", "lead_extractor")
workflow.add_edge("lead_extractor", END)

memory = SqliteSaver(conn)
agent_app = workflow.compile(checkpointer=memory)

# ------------------------------------------------------------------
# 4. FASTAPI & ENDPOINTS DE GESTIÓN DE LEADS
# ------------------------------------------------------------------

app = FastAPI(title="InmoCloud PRO API - Lead Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    thread_id: Optional[str] = "default_session"
    message: str

class ChatResponse(BaseModel):
    response: str
    lead_score: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        thread_id = req.thread_id or "default_session"
        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=req.message)
        
        output = agent_app.invoke({"messages": [input_message]}, config=config)
        
        last_msg = output["messages"][-1].content
        if isinstance(last_msg, list):
            last_msg = str(last_msg[0]) if last_msg else ""
            
        lead_data = output.get("lead_info")
        lead_score = None
        
        # Guardar / Actualizar lead en la BD local SQLite si se identificaron datos
        if lead_data:
            lead_score = lead_data.get("score")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO leads (thread_id, nombre, telefono, zona, presupuesto_max, plazo_mudanza, score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    nombre=coalesce(EXCLUDED.nombre, leads.nombre),
                    telefono=coalesce(EXCLUDED.telefono, leads.telefono),
                    zona=coalesce(EXCLUDED.zona, leads.zona),
                    presupuesto_max=coalesce(EXCLUDED.presupuesto_max, leads.presupuesto_max),
                    plazo_mudanza=coalesce(EXCLUDED.plazo_mudanza, leads.plazo_mudanza),
                    score=coalesce(EXCLUDED.score, leads.score)
            """, (
                thread_id,
                lead_data.get("nombre"),
                lead_data.get("telefono"),
                lead_data.get("zona"),
                lead_data.get("presupuesto_max"),
                lead_data.get("plazo_mudanza"),
                lead_score
            ))
            conn.commit()

        return ChatResponse(response=str(last_msg), lead_score=lead_score)
    except Exception as e:
        return ChatResponse(response=f"Error en el servidor: {str(e)}")

@app.get("/leads")
async def get_leads():
    """Endpoint para listar todos los leads calificados desde la BD."""
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, nombre, telefono, zona, presupuesto_max, plazo_mudanza, score, fecha FROM leads ORDER BY fecha DESC")
    rows = cursor.fetchall()
    leads = [
        {
            "thread_id": r[0],
            "nombre": r[1] or "Anónimo",
            "telefono": r[2] or "No provisto",
            "zona": r[3] or "Sin especificar",
            "presupuesto_max": r[4] or 0,
            "plazo_mudanza": r[5] or "Sin especificar",
            "score": r[6] or "FRIO",
            "fecha": r[7]
        }
        for r in rows
    ]
    return {"total": len(leads), "leads": leads}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)