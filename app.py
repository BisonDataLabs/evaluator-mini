"""Safe public entrypoint: file uploads enabled, Snowflake forcibly disabled."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ["ELM_DEPLOYMENT_MODE"] = "upload_only"
runpy.run_path(Path(__file__).with_name("app_main.py"), run_name="__main__")
