from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Neo4j connection
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Paths
    pdf_path: str = "data/source/essentials-of-human-diseases-and-conditions_compress.pdf"
    processed_dir: str = "data/processed"
    
    
    # LLM / Groq config
    groq_api_key: str | None = None
    groq_model_name: str | None = None
    class Config:
        env_file = "C:\\Users\\User\\Desktop\\agentic-graphrag-med\\data\\source\\.env"

settings = Settings()
