import logging


def configure_logger(name, logdir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(message)s')
    file_handler = logging.FileHandler(logdir + '/' + name + '.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
