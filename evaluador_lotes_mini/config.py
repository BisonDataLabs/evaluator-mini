"""Central runtime configuration.

Secrets are loaded from environment variables only. No credential is copied,
logged or serialized into a job manifest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CEIBOS_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    work_dir: Path
    cache_dir: Path
    output_dir: Path
    max_workers: int
    default_buffer_m: int
    snowflake_account: str | None
    snowflake_user: str | None
    snowflake_role: str
    snowflake_warehouse: str
    snowflake_database: str
    snowflake_schema: str
    snowflake_private_key_path: Path | None
    snowflake_private_key_pem: str | None
    snowflake_private_key_b64: str | None
    snowflake_private_key_passphrase: str | None
    legacy_snowflake_connector: Path

    def ensure_directories(self) -> None:
        for path in (self.work_dir, self.cache_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    default_key = CEIBOS_ROOT / "Generales/credenciales/snowflake/rsa_key_lnieto_analytics.p8"
    key_raw = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_path = Path(key_raw).expanduser() if key_raw else default_key
    if not key_path.exists():
        key_path = None

    settings = Settings(
        project_root=PROJECT_ROOT,
        work_dir=_path_from_env("ELM_WORK_DIR", PROJECT_ROOT / "work"),
        cache_dir=_path_from_env("ELM_CACHE_DIR", PROJECT_ROOT / "cache"),
        output_dir=_path_from_env("ELM_OUTPUT_DIR", PROJECT_ROOT / "outputs"),
        max_workers=max(1, min(8, int(os.getenv("ELM_MAX_WORKERS", "2")))),
        default_buffer_m=max(0, int(os.getenv("ELM_DEFAULT_BUFFER_M", "500"))),
        snowflake_account=os.getenv("SNOWFLAKE_ACCOUNT", "IQTGLEW-CEIBOS"),
        snowflake_user=os.getenv("SNOWFLAKE_USER", "LNIETO_CEIBOS"),
        snowflake_role=os.getenv("SNOWFLAKE_ROLE", "CEIBOS_READ_ONLY_ROLE"),
        snowflake_warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        snowflake_database=os.getenv("SNOWFLAKE_DATABASE", "PROCESSED"),
        snowflake_schema=os.getenv("SNOWFLAKE_SCHEMA", "DBT_MARIANA"),
        snowflake_private_key_path=key_path,
        snowflake_private_key_pem=os.getenv("SNOWFLAKE_PRIVATE_KEY_PEM"),
        snowflake_private_key_b64=os.getenv("SNOWFLAKE_PRIVATE_KEY_B64"),
        snowflake_private_key_passphrase=os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
        legacy_snowflake_connector=(CEIBOS_ROOT / "Generales/credenciales/snowflake/conexion.py"),
    )
    settings.ensure_directories()
    return settings
