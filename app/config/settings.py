# app/config/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuración principal del sistema, cargada desde variables de entorno.
    """

    OPENAI_API_KEY: str  # clave de API para OpenAI

    class Config:
        # Archivo de entorno donde se definen variables:
        env_file = ".env"


# Instancia global de configuración
settings = Settings()
