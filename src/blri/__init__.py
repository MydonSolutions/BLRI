import os
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)

with open(os.path.join(os.path.dirname(__file__), "VERSION"), 'r') as fio:
    __version__ = fio.read()
