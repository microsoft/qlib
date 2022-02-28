import logging
import multiprocessing
import threading
import time
import warnings
from queue import Empty
from typing import Iterable

from torch.utils.data import DataLoader
from utilsd.logging import print_log


class DataQueue:
    def __init__(self, maxsize=0):
        self.queue = multiprocessing.Queue(maxsize=maxsize)
        self._done = multiprocessing.Value('i', 0)

    def cleanup(self):
        with self._done.get_lock():
            self._done.value += 1
        for repeat in range(500):
            if repeat >= 1:
                warnings.warn(f'After {repeat} cleanup, the queue is still not empty.', category=RuntimeWarning)
            while not self.queue.empty():
                try:
                    self.queue.get(block=False)
                except Empty:
                    pass
            time.sleep(1.)  # a chance to allow more data to be put in
            if self.queue.empty():
                break
        print_log(f'Remaining items in queue collection done. Empty: {self.queue.empty()}',
                  __name__, level=logging.DEBUG)

    def get(self, block=True):
        if not hasattr(self, '_first_get'):
            self._first_get = True
        if self._first_get:
            timeout = 5.
            self._first_get = False
        else:
            timeout = .5
        while True:
            try:
                return self.queue.get(block=block, timeout=timeout)
            except Empty:
                if self._done.value:
                    raise StopIteration

    def put(self, obj, block=True, timeout=None):
        return self.queue.put(obj, block=block, timeout=timeout)

    def mark_as_done(self):
        with self._done.get_lock():
            self._done.value = 1

    def done(self):
        return self._done.value

    def __del__(self):
        print_log(f'__del__ of {__name__}.DataQueue', __name__, level=logging.DEBUG)
        self.cleanup()


def make_data_producer(dataloader: DataLoader, infinite: bool = False, queue_size=500) -> DataQueue:
    def _worker(dataloader: DataLoader, queue: DataQueue):
        repeat = 10 ** 18 if infinite else 1
        for _ in range(repeat):
            for data in dataloader:
                if queue.done():
                    return
                queue.put(data)
            print_log('Dataloader loop done.', __name__, level=logging.DEBUG)
        queue.mark_as_done()

    queue = DataQueue(queue_size)
    thread = threading.Thread(target=_worker, args=(dataloader, queue), daemon=True)
    thread.start()
    return queue


def make_data_consumer(queue: DataQueue) -> Iterable:
    while True:
        try:
            yield queue.get()
        except StopIteration:
            print_log('Data consumer timed-out from get.', __name__, level=logging.DEBUG)
            return
