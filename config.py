# config.py
import json
import os

LM_BASE_URL = os.getenv("LM_BASE_URL", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
LM_MODEL = os.getenv("LM_MODEL", "qwen2.5-14b-instruct-1m")
LM_TIMEOUT_S = int(os.getenv("LM_TIMEOUT_S", "900"))
LM_MAX_TOKENS = int(os.getenv("LM_MAX_TOKENS", "1024"))

LM_EXTRA_BODY_RAW = os.getenv("LM_EXTRA_BODY", "")
if LM_EXTRA_BODY_RAW.strip():
    try:
        LM_EXTRA_BODY = json.loads(LM_EXTRA_BODY_RAW)
    except json.JSONDecodeError:
        LM_EXTRA_BODY = None
else:
    LM_EXTRA_BODY = None

TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))