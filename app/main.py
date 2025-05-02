# app/main.py
import openai
from fastapi import FastAPI
from app.config.settings import settings
from app.entrypoints.http.fastapi_endpoints import router as api_router

# ¡IMPORTANTE! asignamos la clave a OpenAI antes de arrancar!
openai.api_key = settings.OPENAI_API_KEY

# Instancia principal de FastAPI
app = FastAPI(title="Asistente Legal de Tránsito")

# Registro de los endpoints del API
app.include_router(api_router)
