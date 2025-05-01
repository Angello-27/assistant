from fastapi import FastAPI
from app.entrypoints.http.fastapi_endpoints import router as api_router

app = FastAPI(title="Asistente Legal de Tránsito")

# Incluir el router de endpoints
app.include_router(api_router)
