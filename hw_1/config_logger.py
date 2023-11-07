import logging

logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
)

logger = logging.getLogger("debug")
