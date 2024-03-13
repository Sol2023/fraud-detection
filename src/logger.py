import logging
import os
from datetime import datetime

LOG_FILE = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path = os.path.dirname(LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(os.getcwd(), LOG_FILE)


def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=logging.INFO,
        format="[%(asctime)s]%(lineno)d %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"

    )


if __name__ == "__main__":
    setup_logging()
    logging.info("Start logging")
    # logging.warning("This is a warning message")
    # logging.error("This is an error message")
    # logging.critical("This is a critical message")