# app/main.py
from fastapi import FastAPI
from app.entrypoints.http.fastapi_endpoints import router as api_router

# Instancia principal de FastAPI
app = FastAPI(title="Asistente Legal de Tránsito")

# Registro de los endpoints del API
app.include_router(api_router)
