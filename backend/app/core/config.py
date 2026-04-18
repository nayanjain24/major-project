from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATAFUSION_")

    app_name: str = "DataFusion AI Backend"
    environment: str = "dev"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg2://datafusion:datafusion@db:5432/datafusion"
    storage_dir: str = "./storage"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    encryption_key: str = "CHANGE_ME_32BYTE_KEY________________"


settings = Settings()
