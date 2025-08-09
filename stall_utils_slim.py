import base64
import datetime
import hashlib
import logging
import os

from colorama import Fore

TqdmBarFormat = '%s{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.CYAN, Fore.GREEN)

nt_onehot_mapping = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
}
aa_onehot_mapping = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3,
    'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15,
    'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '*': 20, '&': 21, '#': 22,
}
NTCharset = [
    # ' ',
    'A', 'T', 'C', 'G',
]
AACharset = [
    # ' ',
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    '*', '&', '#',
]
assert len(nt_onehot_mapping) == len(NTCharset)
assert len(aa_onehot_mapping) == len(AACharset)
TimeFormat = "%Y%m%d-%H%M%S"


def get_time():
    return datetime.datetime.now().strftime(TimeFormat)


def create_logger(
        logger_name: str = "Stallogger",
        log_dir: str = ".",
        logger_filename: str = "Stalling_" + get_time() + ".log",  # = None,
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger_path = os.path.join(log_dir, logger_filename)
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    # 记录日志信息
    # logger.debug('debug message')
    # logger.info('info message')
    # logger.warning('warning message')
    # logger.error('error message')
    # logger.critical('critical message')
    logger.info("Logger file '%s' created" % os.path.abspath(logger_path))

    return logger


def gen_id(seq: str) -> str:
    encoding = 'utf-8'
    id_seq = base64.b32encode(hashlib.sha256(seq.encode(encoding)).digest()).lower()[:12].decode(encoding)
    return id_seq

