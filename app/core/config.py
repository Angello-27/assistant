import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

settings = Settings()

if not settings.OPENAI_API_KEY:
    raise Exception("La API Key de OpenAI no está configurada en el archivo .env")
