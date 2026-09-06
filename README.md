Set-Content -Path README.md -Value @"
# AI Real Estate Agent - LangGraph & Ollama

Agente de IA autónomo para el sector inmobiliario que utiliza modelos de lenguaje locales (vía Ollama), orquestación con LangGraph, persistencia en SQLite y una interfaz API en FastAPI con un widget flotante en JavaScript.

## 🛠️ Tecnologías

* **LLM Local:** Ollama (llama3.2)
* **Orquestación de Agentes:** LangGraph + LangChain
* **API Backend:** FastAPI / Uvicorn
* **Memoria Persistente:** SQLite (SqliteSaver)
* **Frontend:** Widget nativo en JS / HTML5

## 🚀 Instalación y Uso

1. **Clonar repositorio:**
   ```bash
   git clone [https://github.com/cloudworksglobal/airealstate.git](https://github.com/cloudworksglobal/airealstate.git)
   cd airealstate
