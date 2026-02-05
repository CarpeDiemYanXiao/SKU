import time
import logging
import datetime
import requests
import functools
from typing import Any, Callable, Optional

import torch.distributed as dist


def logging_setup(path=""):
    # formatter
    fmt = "%(asctime)s %(filename)s pid-%(process)d %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    formatter.converter = lambda _: (datetime.datetime.now() + datetime.timedelta(hours=8)).timetuple()
    # handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="a", delay=True)  # delay=True, 延迟创建文件
    file_handler.setFormatter(formatter)
    # root logger
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    if path:
        logging.getLogger().addHandler(file_handler)


def logging_once(msg: Any, level: int = logging.INFO, **kwargs):
    """打印一次
    在单卡环境正常使用。
    在多卡环境只有在 GPU0 所在进程才会打印。最好是打印相同的内容或者是已经同步过的内容。
    可以选择 logging 级别。
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        if level == logging.DEBUG:
            logging.debug(msg, **kwargs)
        elif level == logging.INFO:
            logging.info(msg, **kwargs)
        elif level == logging.WARNING:
            logging.warning(msg, **kwargs)
        elif level == logging.ERROR:
            logging.error(msg, **kwargs)
        elif level == logging.CRITICAL:
            logging.critical(msg, **kwargs)


def print_once(*args, **kwargs):
    """打印一次
    在单卡环境正常使用。
    在多卡环境只有在 GPU0 所在进程才会打印。最好是打印相同的内容或者是已经同步过的内容。
    可以选择 logging 级别。
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def timeit(func: Callable):
    """函数计数计时装饰器
    记录函数调用时长和次数的装饰器。
    """
    call_count = [0]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        call_count[0] += 1
        c = call_count[0]
        str_count = str(c) + ("st" if c == 1 else "nd" if c == 2 else "th")
        try:
            time_start = time.time()
            logging.info(f"=== The {str_count} Call of [{func.__qualname__}] starts === ")
            retval = func(*args, **kwargs)
            logging.info(f"=== The {str_count} Call of [{func.__qualname__}] finish === " f"time cost: {time.time() - time_start:.2f}s")
            return retval
        except Exception as e:
            logging.error(f"=== The {str_count} Call of [{func.__qualname__}] failed ===", exc_info=e)
            raise e

    return wrapper


class Timer:
    """上下文计时器，单例模式
    记录一段过程的用时。
    Example:
        with Timer("数据处理"):
            process(...)
    """

    def __init__(self, desc=""):
        self.desc = desc
        self.time = time.time()

    def __enter__(self):
        logging.info(f"{self.desc}")

    def __exit__(self, exc_type, exc_value, traceback):
        seconds = time.time() - self.time
        logging.info(f"{self.desc} 结束, 用时 {seconds:.2f} 秒.")


def seatalk_alert(message, url: Optional[str] = None, mentioned_email_list=[]):
    if url is None:
        raise Exception("seatalk url cannot be empty.")
    json_body = {"tag": "text", "text": {"content": message, "mentioned_email_list": mentioned_email_list}}
    headers = {"Content-Type": "application/json;charset=UTF-8", "Accept": "application/json text/plain, */*"}
    try:
        requests.post(url, headers=headers, json=json_body)
    except Exception as e:
        logging.error(f"seatalk alert error!", exc_info=e)
