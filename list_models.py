# list_models.py
from openai import OpenAI

import config

client = OpenAI(base_url=config.LM_BASE_URL, api_key=config.LM_API_KEY)
for m in client.models.list().data:
    print(m.id)