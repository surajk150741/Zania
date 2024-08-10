import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY_ZANIA")

class Config:
    HANDBOOK_FILE="document/handbook.pdf"
    PLAYBOOK_DB="db/UMM_PLAYBOOK_10_2023_Multivector"
    SQL_CONFIG = "services/utils/config.json"
    EXAMPLES = "examples.json"
    UMM_DATABASE=os.environ.get("UMM_DATABASE", "")
    UMM_TABLE_NAME=os.environ.get("UMM_TABLE_NAME", "")

setting=Config()
# print(setting.CHAT_MEMORY_DATABASE)
