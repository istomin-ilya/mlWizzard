from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/lightgbm_ranker.pkl"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "5"))
    live_mode: bool = os.getenv("LIVE_MODE", "false").lower() == "true"


settings = Settings()