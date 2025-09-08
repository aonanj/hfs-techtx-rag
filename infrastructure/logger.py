import logging
import sys

_loggers = {}

def setup_logger(name="techtrans", level=logging.INFO, toFile=True, fileName="techtrans.log"):
    """
    Establish an instance of a logger to be used for logging in current context of app

    Args
        name: name of the logger
        level: level of logging info
        toFile: whether to log to file (ignored in cloud environments)
        fileName: log file name (ignored in cloud environments)
    """
    if name in _loggers:
        return _loggers[name]

    # Local development logging
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[%(asctime)s] - %(name)s %(levelname)s %(message)s")

    if toFile:
        fileHandler = logging.FileHandler(fileName)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler(sys.stderr)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    _loggers[name] = logger
    return logger

def get_logger(name="techtrans"):
    return logging.getLogger(name)