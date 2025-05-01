from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuración principal del sistema cargada desde variables de entorno.
    """

    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
