from __future__ import print_function
import logging
import __builtin__ 

open('cnn.log', 'w').close()
try:
    logger
except NameError:
    logger = logging.getLogger('cnn.log')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('cnn.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)     
    
def print(*args, **kwargs):
    msg = ""
    for i in args:
        msg = msg + str(i) + ' '
    logger.debug(msg)
    return __builtin__.print(*args, **kwargs)