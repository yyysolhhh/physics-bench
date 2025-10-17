import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=False)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)
