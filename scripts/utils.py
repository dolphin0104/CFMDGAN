import os
import numpy as np
import random
import logging
import torch
from datetime import datetime

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_random_seed(is_gpu=True, manual_seed=1):    
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if is_gpu:
        torch.cuda.manual_seed_all(manual_seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, is_formatter=True, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    if is_formatter:
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', 
            datefmt='%y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        if is_formatter:
            sh.setFormatter(formatter)
        l.addHandler(sh)


def write_log(log, save_path, save_name, refresh=False):
    #print(log)
    log_txt = os.path.join(save_path, '{}.txt'.format(save_name))
    open_type = 'a' if os.path.exists(log_txt) else 'w'
    log_file = open(log_txt, open_type)
    log_file.write(str(log))
    log_file.write('\n')
    if refresh:
        log_file.close()
        log_file = open(log_txt, 'a')