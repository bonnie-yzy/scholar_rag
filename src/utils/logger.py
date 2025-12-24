import logging
import sys
from colorama import Fore, Style, init

init(autoreset=True)

def setup_logger(name="ScholarRAG"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f'{Fore.GREEN}%(asctime)s{Style.RESET_ALL} | '
            f'{Fore.CYAN}%(levelname)s{Style.RESET_ALL} | '
            f'%(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger