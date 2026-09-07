from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ------------------------------------------------------------------
# 1. HERRAMIENTAS (Lo que el Agente PUEDE HACER)
# ------------------------------------------------------------------

@tool
def buscar_propiedades(zona: str, presupuesto_max: int, ambientes: int = None) -> str:
    """Consulta la base de datos de propiedades disponibles según los criterios del cliente."""
    # Simulamos una base de datos o inventario inmobiliario
    inventario = [
        {"id": 101, "titulo": "Dpto luminoso en Palermo", "zona": "Palermo", "precio": 120000, "ambientes": 2},
        {"id": 102, "titulo": "Casa amplia con jardín", "zona": "Belgrano", "precio": 250000, "ambientes": 4},
        {"id": 103, "titulo": "Monoambiente céntrico", "zona": "Palermo", "precio": 85000, "ambientes": 1},
    ]
    
    resultados = []
    for prop in inventario:
        if prop["zona"].lower() in zona.lower() and prop["precio"] <= presupuesto_max:
            resultados.append(prop)
            
    if not resultados:
        return "No se encontraron propiedades exactas para esos criterios en el catálogo actual."
    
    return f"Propiedades encontradas: {resultados}"

tools = [buscar_propiedades]

# ------------------------------------------------------------------
# 2. MODELO + BINDING DE HERRAMIENTAS
# ------------------------------------------------------------------

# Importante: Para usar Tool Calling nativo con Ollama,
# asegurate de usar modelos compatibles como llama3.1, llama3.2 o qwen2.5
llm = ChatOllama(
    model="llama3.2",
    temperature=0.1
)

# Conectamos las herramientas directamente al modelo de lenguaje
llm_with_tools = llm.bind_tools(tools)

# ------------------------------------------------------------------
# 3. ESTADO DEL AGENTE
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------------------------------------------------------
# 4. NODOS DEL GRAFO
# ------------------------------------------------------------------

def chatbot_node(state: AgentState):
    """Nodo donde el agente piensa, responde al usuario o decide invocar una herramienta."""
    system_prompt = SystemMessage(
        content=(
            "Sos un Agente Inmobiliario Autónomo de primer nivel. "
            "Tu meta es asesorar al cliente. Cuando el cliente te proporcione suficiente información "
            "(como zona deseada y presupuesto), DEBES ejecutar la herramienta 'buscar_propiedades' "
            "para consultar el inventario real. Si faltan datos, hacé preguntas cortas y amables."
        )
    )
    
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ------------------------------------------------------------------
# 5. CONSTRUCCIÓN DEL GRAFO DE LANGGRAPH
# ------------------------------------------------------------------

workflow = StateGraph(AgentState)

# Agregar nodos
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

# Flujo de ejecución
workflow.add_edge(START, "chatbot")

# Borde condicional: si el chatbot solicita ejecutar una herramienta, va a 'tools'.
# De lo contrario, termina el turno y le responde al usuario.
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# Compilar la aplicación
app = workflow.compile()

# ------------------------------------------------------------------
# 6. EJECUCIÓN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("🤖 Agente Inmobiliario Activo (Con capacidad de ejecución de herramientas)\n")
    state = {"messages": []}
    
    while True:
        user_input = input("Cliente: ")
        if user_input.lower() in ["salir", "exit"]:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        output = app.invoke(state)
        
        # Obtenemos el último mensaje generado por el agente
        respuesta_final = output["messages"][-1].content
        print(f"\nAgente: {respuesta_final}\n")
        state = output