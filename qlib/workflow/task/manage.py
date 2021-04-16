# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
A task consists of 3 parts
- tasks description: the desc will define the task
- tasks status: the status of the task
- tasks result information : A user can get the task with the task description and task result.

"""
from bson.binary import Binary
import pickle
from pymongo.errors import InvalidDocument
from bson.objectid import ObjectId
from contextlib import contextmanager
import qlib
from tqdm.cli import tqdm
import time
import concurrent
import pymongo
from qlib.config import C
from .utils import get_mongodb
from qlib import get_module_logger, auto_init
import fire


class TaskManager:
    """TaskManager
    here is what will a task looks like when it created by TaskManager

    .. code-block:: python

        {
            'def': pickle serialized task definition.  using pickle will make it easier
            'filter': json-like data. This is for filtering the tasks.
            'status': 'waiting' | 'running' | 'done'
            'res': pickle serialized task result,
        }

    The tasks manager assume that you will only update the tasks you fetched.
    The mongo fetch one and update will make it date updating secure.

    .. note::

        Assumption: the data in MongoDB was encoded and the data out of MongoDB was decoded
    """

    STATUS_WAITING = "waiting"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_PART_DONE = "part_done"

    ENCODE_FIELDS_PREFIX = ["def", "res"]

    def __init__(self, task_pool: str = None):
        """
        init Task Manager, remember to make the statement of MongoDB url and database name firstly.

        Parameters
        ----------
        task_pool: str
            the name of Collection in MongoDB
        """
        self.mdb = get_mongodb()
        if task_pool is not None:
            self.task_pool = getattr(self.mdb, task_pool)
        self.logger = get_module_logger(self.__class__.__name__)

    def list(self):
        """
        list the all collection(task_pool) of the db

        Returns:
            list
        """
        return self.mdb.list_collection_names()

    def _encode_task(self, task):
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = Binary(pickle.dumps(task[k]))
        return task

    def _decode_task(self, task):
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = pickle.loads(task[k])
        return task

    def _dict_to_str(self, flt):
        return {k: str(v) for k, v in flt.items()}

    def replace_task(self, task, new_task):
        # assume that the data out of interface was decoded and the data in interface was encoded
        new_task = self._encode_task(new_task)
        query = {"_id": ObjectId(task["_id"])}
        try:
            self.task_pool.replace_one(query, new_task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            self.task_pool.replace_one(query, new_task)

    def insert_task(self, task):

        try:
            insert_result = self.task_pool.insert_one(task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            insert_result = self.task_pool.insert_one(task)
        return insert_result

    def insert_task_def(self, task_def):
        """
        insert a task to task_pool

        Parameters
        ----------
        task_def: dict
            the task definition

        Returns
        -------

        """
        task = self._encode_task(
            {
                "def": task_def,
                "filter": task_def,  # FIXME: catch the raised error
                "status": self.STATUS_WAITING,
            }
        )
        insert_result = self.insert_task(task)
        return insert_result

    def create_task(self, task_def_l, dry_run=False, print_nt=False):
        """
        if the tasks in task_def_l is new, then insert new tasks into the task_pool

        Parameters
        ----------
        task_def_l: list
            a list of task
        dry_run: bool
            if insert those new tasks to task pool
        print_nt: bool
            if print new task
        Returns
        -------
        list
            a list of the _id of new tasks
        """
        new_tasks = []
        for t in task_def_l:
            try:
                r = self.task_pool.find_one({"filter": t})
            except InvalidDocument:
                r = self.task_pool.find_one({"filter": self._dict_to_str(t)})
            if r is None:
                new_tasks.append(t)
        self.logger.info(f"Total Tasks: {len(task_def_l)}, New Tasks: {len(new_tasks)}")

        if print_nt:  # print new task
            for t in new_tasks:
                print(t)

        if dry_run:
            return

        _id_list = []
        for t in new_tasks:
            insert_result = self.insert_task_def(t)
            _id_list.append(insert_result.inserted_id)

        return _id_list

    def fetch_task(self, query={}):
        query = query.copy()
        if "_id" in query:
            query["_id"] = ObjectId(query["_id"])
        query.update({"status": self.STATUS_WAITING})
        task = self.task_pool.find_one_and_update(
            query, {"$set": {"status": self.STATUS_RUNNING}}, sort=[("priority", pymongo.DESCENDING)]
        )
        # null will be at the top after sorting when using ASCENDING, so the larger the number higher, the higher the priority
        if task is None:
            return None
        task["status"] = self.STATUS_RUNNING
        return self._decode_task(task)

    @contextmanager
    def safe_fetch_task(self, query={}):
        """
        fetch task from task_pool using query with contextmanager

        Parameters
        ----------
        query: dict
            the dict of query

        Returns
        -------

        """
        task = self.fetch_task(query=query)
        try:
            yield task
        except Exception:
            if task is not None:
                self.logger.info("Returning task before raising error")
                self.return_task(task)
                self.logger.info("Task returned")
            raise

    def task_fetcher_iter(self, query={}):
        while True:
            with self.safe_fetch_task(query=query) as task:
                if task is None:
                    break
                yield task

    def query(self, query={}, decode=True):
        """
        This function may raise exception `pymongo.errors.CursorNotFound: cursor id not found` if it takes too long to iterate the generator

        Parameters
        ----------
        query: dict
            the dict of query
        decode: bool

        Returns
        -------

        """
        query = query.copy()
        if "_id" in query:
            query["_id"] = ObjectId(query["_id"])
        for t in self.task_pool.find(query):
            yield self._decode_task(t)

    def re_query(self, _id):
        t = self.task_pool.find_one({"_id": ObjectId(_id)})
        return self._decode_task(t)

    def commit_task_res(self, task, res, status=None):
        # A workaround to use the class attribute.
        if status is None:
            status = TaskManager.STATUS_DONE
        self.task_pool.update_one({"_id": task["_id"]}, {"$set": {"status": status, "res": Binary(pickle.dumps(res))}})

    def return_task(self, task, status=None):
        if status is None:
            status = TaskManager.STATUS_WAITING
        update_dict = {"$set": {"status": status}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def remove(self, query={}):
        """
        remove the task using query

        Parameters
        ----------
        query: dict
            the dict of query

        """
        query = query.copy()
        if "_id" in query:
            query["_id"] = ObjectId(query["_id"])
        self.task_pool.delete_many(query)

    def task_stat(self, query={}):
        query = query.copy()
        if "_id" in query:
            query["_id"] = ObjectId(query["_id"])
        tasks = self.query(query=query, decode=False)
        status_stat = {}
        for t in tasks:
            status_stat[t["status"]] = status_stat.get(t["status"], 0) + 1
        return status_stat

    def reset_waiting(self, query={}):
        query = query.copy()
        # default query
        if "status" not in query:
            query["status"] = self.STATUS_RUNNING
        return self.reset_status(query=query, status=self.STATUS_WAITING)

    def reset_status(self, query, status):
        query = query.copy()
        if "_id" in query:
            query["_id"] = ObjectId(query["_id"])
        print(self.task_pool.update_many(query, {"$set": {"status": status}}))

    def prioritize(self, task, priority: int):
        """
        set priority for task

        Parameters
        ----------
        task : dict
            The task query from the database
        priority : int
            the target priority
        """
        update_dict = {"$set": {"priority": priority}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def _get_undone_n(self, task_stat):
        return task_stat.get(self.STATUS_WAITING, 0) + task_stat.get(self.STATUS_RUNNING, 0)

    def _get_total(self, task_stat):
        return sum(task_stat.values())

    def wait(self, query={}):
        task_stat = self.task_stat(query)
        total = self._get_total(task_stat)
        last_undone_n = self._get_undone_n(task_stat)
        with tqdm(total=total, initial=total - last_undone_n) as pbar:
            while True:
                time.sleep(10)
                undone_n = self._get_undone_n(self.task_stat(query))
                pbar.update(last_undone_n - undone_n)
                last_undone_n = undone_n
                if undone_n == 0:
                    break

    def __str__(self):
        return f"TaskManager({self.task_pool})"


def run_task(task_func, task_pool, force_release=False, *args, **kwargs):
    """
    While task pool is not empty (has WAITING tasks), use task_func to fetch and run tasks in task_pool

    Parameters
    ----------
    task_func : def (task_def, *args, **kwargs) -> <res which will be committed>
        the function to run the task
    task_pool : str
        the name of the task pool (Collection in MongoDB)
    force_release :
        will the program force to release the resource
    args :
        args
    kwargs :
        kwargs
    """
    tm = TaskManager(task_pool)

    ever_run = False

    while True:
        with tm.safe_fetch_task() as task:
            if task is None:
                break
            get_module_logger("run_task").info(task["def"])
            if force_release:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:  # what this means?
                    res = executor.submit(task_func, task["def"], *args, **kwargs).result()
            else:
                res = task_func(task["def"], *args, **kwargs)
            tm.commit_task_res(task, res)
            ever_run = True

    return ever_run


if __name__ == "__main__":
    # This is for using it in cmd
    # E.g. : `python -m qlib.workflow.task.manage list`
    auto_init()
    fire.Fire(TaskManager)
