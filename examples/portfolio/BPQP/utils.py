import datetime
import logging
import random

import numpy as np
import torch


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def dict_report(stats, key, value, op="append"):
    if key in stats.keys():
        if op == "append":
            stats[key] = np.append(stats[key], value)
        elif op == "concat":
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value


def write_log(*info):
    # print with UTC+8 time
    time = "[" + str(datetime.datetime.utcnow() + datetime.timedelta(hours=8))[:19] + "] -"
    print(time, *info)

    logging.info("{time} {info}".format(time=time,info=info))
