import logging
from ._version import __version__

logging.INFO_THROUGHPUT =  logging.DEBUG+1
logging.addLevelName(logging.INFO_THROUGHPUT, "INFO_THROUGHPUT")
logging.INFO_STEPS =  logging.DEBUG+2
logging.addLevelName(logging.INFO_STEPS, "INFO_STEPS")
logging.INFO_BLOCKS = logging.DEBUG+3
logging.addLevelName(logging.INFO_BLOCKS, "INFO_BLOCKS")

def info_throughput(self, message, *args, **kws):
    if self.isEnabledFor(logging.INFO_THROUGHPUT):
        self._log(logging.INFO_THROUGHPUT, message, args, **kws)
logging.Logger.info_throughput = info_throughput

def info_steps(self, message, *args, **kws):
    if self.isEnabledFor(logging.INFO_STEPS):
        self._log(logging.INFO_STEPS, message, args, **kws)
logging.Logger.info_steps = info_steps

def info_blocks(self, message, *args, **kws):
    if self.isEnabledFor(logging.INFO_BLOCKS):
        self._log(logging.INFO_BLOCKS, message, args, **kws)
logging.Logger.info_blocks = info_blocks

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)

