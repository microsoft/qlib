import os
import multiprocessing
import threading
import time
import warnings
from queue import Empty
from typing import Sized, Any

from torch.utils.data import DataLoader
from qlib.log import get_module_logger

_logger = get_module_logger(__name__)

class DataQueue:
    """Main process (producer) produces data and stores them in a queue.
    Sub-processes (consumers) can retrieve the data-points from the queue.
    Data-points are generated via reading items from ``dataset``.

    :class:`DataQueue` is ephemeral. You must create a new DataQueue
    when the ``repeat`` is exhausted.

    Parameters
    ----------
    repeat
        Iterate over the data-points for how many times. Use ``-1`` to iterate forever.
    shuffle
        If ``shuffle`` is true, the items will be read in random order.
    producer_num_workers
        Concurrent workers for data-loading.
    queue_maxsize
        Maximum items to put into queue before it jams.
    autoactivate
        Start the producer instantly when creating the queue.
        Otherwise, users need to call ``activate()`` manually before using the queue.

    Examples
    --------
    >>> data_queue = DataQueue(my_dataset)

    In worker:
    >>> for data in data_queue:
    ...     print(data)
    """

    def __init__(self, dataset: Sized[Any],
                 repeat: int = 1,
                 producer_num_workers: int = 0,
                 queue_maxsize: int = 0,
                 autoactivate: bool = False):
        if queue_maxsize == 0:
            queue_maxsize = os.cpu_count()
            _logger.info(f'Automatically set data queue maxsize to {queue_maxsize} to avoid overwhelming.')

        if autoactivate:
            self.activate()

        self.dataset: Sized[Any] = dataset
        self.repeat: int = repeat
        self.producer_num_workers: int = producer_num_workers

        self._activated: bool = False
        self._queue = multiprocessing.Queue(maxsize=queue_maxsize)
        self._done = multiprocessing.Value('i', 0)

    def cleanup(self):
        with self._done.get_lock():
            self._done.value += 1
        for repeat in range(500):
            if repeat >= 1:
                warnings.warn(f'After {repeat} cleanup, the queue is still not empty.', category=RuntimeWarning)
            while not self._queue.empty():
                try:
                    self._queue.get(block=False)
                except Empty:
                    pass
            time.sleep(1.)  # a chance to allow more data to be put in
            if self._queue.empty():
                break
        _logger.debug(f'Remaining items in queue collection done. Empty: {self._queue.empty()}')

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
                return self._queue.get(block=block, timeout=timeout)
            except Empty:
                if self._done.value:
                    raise StopIteration

    def put(self, obj, block=True, timeout=None):
        return self._queue.put(obj, block=block, timeout=timeout)

    def mark_as_done(self):
        with self._done.get_lock():
            self._done.value = 1

    def done(self):
        return self._done.value

    def activate(self):
        thread = threading.Thread(target=self._producer, daemon=True)
        thread.start()
        self._activated = True

    def __del__(self):
        _logger.debug(f'__del__ of {__name__}.DataQueue')
        self.cleanup()

    def __iter__(self):
        if not self._activated:
            raise ValueError('Need to call activate() to launch a daemon worker '
                             'to produce data into data queue before using it.')
        self._consumer()

    def _consumer(self):
        while True:
            try:
                yield self.get()
            except StopIteration:
                _logger.debug('Data consumer timed-out from get.')
                return

    def _producer(self):
        # pytorch dataloader is used here only because we need its sampler and multi-processing
        dataloader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=self.producer_num_workers,
            collate_fn=lambda t: t,  # identity collate fn
        )
        repeat = 10 ** 18 if self.repeat == -1 else self.repeat
        for _rep in range(repeat):
            for data in dataloader:
                if self._queue.done():
                    # Already done.
                    return
                self._queue.put(data)
            _logger.debug(f'Dataloader loop done. Repeat {_rep}.')
        self._queue.mark_as_done()
