import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY_ZANIA")

class Config:
    CHAT_MEMORY_DATABASE=os.getenv("CHAT_MEMORY_DATABASE_ZANIA", "")
    HANDBOOK_FILE="document/handbook.pdf"

setting=Config()
# print(setting.CHAT_MEMORY_DATABASE)
