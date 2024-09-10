import sys
import logging


def logging_setup(logfile):
    logger = logging.getLogger('TestLog')
    
    if not logger.handlers:
        handler_sysout = logging.StreamHandler(sys.stdout)
        handler_file = logging.FileHandler(logfile)
        
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        handler_sysout.setFormatter(formatter)
        handler_file.setFormatter(formatter)

        logger.addHandler(handler_sysout)
        logger.addHandler(handler_file)

        logger.setLevel(logging.INFO)
        logger.propagate = False


def info(msg):
    logging.getLogger('TestLog').info(msg)


def warn(msg):
    logging.getLogger('TestLog').warn(msg)


def error(msg):
    logging.getLogger('TestLog').error(msg)


