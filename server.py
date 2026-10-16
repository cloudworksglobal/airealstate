import os
import requests
from fastapi import FastAPI, Request, Response, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

VERIFY_TOKEN = "mi_token_secreto_123"
PHONE_NUMBER_ID = "1334309566437960"
META_ACCESS_TOKEN = "EAAp5xOeI09IBStPHQa3QEY1qklVrxKn2ek9wKZC66iLZCNMj9TeEEYz1E1P7CByP1ZCwq2cPdAhKJoI335W5SiMuTz8CyZAZCPvWsT0sxLZCrF3Eg3JXaXyyJrs2AKrjvDOOv0tr6odbZAz7fGGSfog5ZCxPt9YIyKyQF8f9pONql3r5OUJeVVBZBE0xEWjTeZAqmRaFMJh5wSzZAvGZC80LL8PcOeqpZCHrnj59h8kqeVeXjuCva3BVrfS1YcBYKyYZAQbbFrbkiZCZA8nImNJ7bRIBESwJ9qGE12IzGHE8agZDZD"

class ChatRequest(BaseModel):
    message: str = None
    mensaje: str = None
    client_id: str = None
    cliente_id: str = None

def consultar_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:0.5b",
                "prompt": f"Sos un atento asistente de una inmobiliaria en Buenos Aires. Responde de forma muy breve y profesional: {prompt}",
                "stream": False,
                "options": {
                    "num_predict": 60,
                    "temperature": 0.3
                }
            },
            timeout=15
        )
        return r.json().get("response", "No pude procesar la respuesta.")
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return f"Error en consulta local: {e}"

def procesar_y_responder_whatsapp(cliente_id: str, mensaje: str):
    print(f"[OLLAMA] Procesando mensaje de {cliente_id}...")
    respuesta_ia = consultar_ollama(mensaje)
    print(f"[OLLAMA] Respuesta generada: {respuesta_ia}")
    
    exito = enviar_mensaje_whatsapp(cliente_id, respuesta_ia)
    
    if not exito and cliente_id.startswith("549"):
        numero_alternativo = "54" + cliente_id[3:]
        print(f"[RETRY] Probando envio a variante sin 9: {numero_alternativo}")
        enviar_mensaje_whatsapp(numero_alternativo, respuesta_ia)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    texto = req.message or req.mensaje or ""
    respuesta_ia = consultar_ollama(texto)
    return {"response": respuesta_ia}

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        print("[OK] WEBHOOK VERIFICADO")
        return Response(content=params.get("hub.challenge"), status_code=200)
    return Response(content="Error", status_code=403)

@app.post("/webhook")
async def receive_whatsapp(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" in entry:
            mensaje = entry["messages"][0]["text"]["body"]
            cliente_id = entry["messages"][0]["from"]
            
            print(f"[INBOUND] WhatsApp de {cliente_id}: {mensaje}")
            background_tasks.add_task(procesar_y_responder_whatsapp, cliente_id, mensaje)
    except Exception as e:
        print(f"[ERROR] {e}")

    return {"status": "ok"}

def enviar_mensaje_whatsapp(to_number: str, text: str) -> bool:
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text}
    }
    r = requests.post(url, json=payload, headers=headers)
    if r.status_code == 200:
        print(f"[OUTBOUND] Enviado exitosamente a {to_number}")
        return True
    else:
        print(f"[ERROR META] Status {r.status_code}: {r.text}")
        return False
