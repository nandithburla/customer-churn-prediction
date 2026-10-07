import logging
import os
from datetime import datetime

from src.central_logger import CentralLogHandler


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] line %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


root_logger = logging.getLogger()

central_handler = CentralLogHandler()

root_logger.addHandler(central_handler)