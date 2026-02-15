import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "default_secret_key_for_testing_only")
PORT = int(os.getenv("PORT", 8000))
