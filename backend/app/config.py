import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings"""
    # Database settings
    database_url = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # Security settings
    secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
    algorithm = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Training server settings
    training_server_url = os.getenv("TRAINING_SERVER_URL", "http://ddns.hoo.ink:8000")
    training_server_timeout = int(os.getenv("TRAINING_SERVER_TIMEOUT", "3600"))
    
    # API settings
    api_v1_str = "/api"
    project_name = "QLib AI API"
    
    # Email settings
    smtp_server = os.getenv("SMTP_SERVER", "smtphz.qiye.163.com")
    smtp_port = int(os.getenv("SMTP_PORT", "994"))
    smtp_username = os.getenv("SMTP_USERNAME", "qlib@uszho.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "Moshou99")
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "t")
    sender_email = os.getenv("SENDER_EMAIL", "qlib@uszho.com")
    
    # Email verification settings
    verification_token_expire_minutes = int(os.getenv("VERIFICATION_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
    skip_email_verification = os.getenv("SKIP_EMAIL_VERIFICATION", "False").lower() in ("true", "1", "t")
    
    # CORS settings
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3001,http://localhost:3000,http://localhost:8000,http://127.0.0.1:3001,http://127.0.0.1:3000,http://127.0.0.1:8000")

    # Default production origins always included
    _default_origins = [
        "http://116.62.59.244",
        "http://qlib.hoo.ink",
        "http://ddns.hoo.ink:8000",
    ]

    def get_cors_origins(self) -> list:
        """Parse CORS origins from settings, combining env var and defaults."""
        origins = self.cors_origins
        if isinstance(origins, str):
            parsed = [o.strip() for o in origins.split(",") if o.strip()]
        elif isinstance(origins, list):
            parsed = origins
        else:
            parsed = []
        # Merge with defaults, avoiding duplicates
        for o in self._default_origins:
            if o not in parsed:
                parsed.append(o)
        return parsed

# Create settings instance
settings = Settings()
