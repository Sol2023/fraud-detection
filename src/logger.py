import logging
import os
from datetime import datetime

LOG_FILE = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path = os.path.dirname(LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(os.getcwd(), LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)