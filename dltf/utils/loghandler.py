import sys
import logging


def logging_setup(logfile=None, on_stdout=True):
    logger = logging.getLogger('TestLog')
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        if logfile:
            handler_file = logging.FileHandler(logfile)
            handler_file.setFormatter(formatter)
            logger.addHandler(handler_file)
            
        if on_stdout:
            handler_sysout = logging.StreamHandler(sys.stdout)
            handler_sysout.setFormatter(formatter)
            logger.addHandler(handler_sysout)

        logger.setLevel(logging.INFO)
        logger.propagate = False


def info(msg):
    logging.getLogger('TestLog').info(msg)


def warn(msg):
    logging.getLogger('TestLog').warning(msg)


def error(msg):
    logging.getLogger('TestLog').error(msg)


