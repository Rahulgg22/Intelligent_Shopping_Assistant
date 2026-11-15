from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PS-12 Conversational Shopping Assistant"

    top_k: int = 5

    google_api_key: str = ""

    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.6-flash"
    gemini_timeout_seconds: float = 10.0

    # Tavily Web Search
    tavily_api_key: str = ""
    tavily_endpoint: str = "https://api.tavily.com/search"
    tavily_search_depth: str = "advanced"
    tavily_max_results: int = 8
    tavily_page_fetch_limit: int = 4
    tavily_timeout_seconds: float = 8.0

    # Serper Google Shopping
    serper_api_key: str = ""
    serper_endpoint: str = "https://google.serper.dev/shopping"
    serper_gl: str = "in"
    serper_hl: str = "en"
    serper_num: int = 10
    serper_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
