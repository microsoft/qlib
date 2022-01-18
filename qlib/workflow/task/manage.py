# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TaskManager can fetch unused tasks automatically and manage the lifecycle of a set of tasks with error handling.
These features can run tasks concurrently and ensure every task will be used only once.
Task Manager will store all tasks in `MongoDB <https://www.mongodb.com/>`_.
Users **MUST** finished the configuration of `MongoDB <https://www.mongodb.com/>`_ when using this module.

A task in TaskManager consists of 3 parts
- tasks description: the desc will define the task
- tasks status: the status of the task
- tasks result: A user can get the task with the task description and task result.
"""
import concurrent
import pickle
import time
from contextlib import contextmanager
from typing import Callable, List

import fire
import pymongo
from bson.binary import Binary
from bson.objectid import ObjectId
from pymongo.errors import InvalidDocument
from qlib import auto_init, get_module_logger
from tqdm.cli import tqdm

from .utils import get_mongodb
from ...config import C


class TaskManager:
    """
    TaskManager

    Here is what will a task looks like when it created by TaskManager

    .. code-block:: python

        {
            'def': pickle serialized task definition.  using pickle will make it easier
            'filter': json-like data. This is for filtering the tasks.
            'status': 'waiting' | 'running' | 'done'
            'res': pickle serialized task result,
        }

    The tasks manager assumes that you will only update the tasks you fetched.
    The mongo fetch one and update will make it date updating secure.

    This class can be used as a tool from commandline. Here are several examples.
    You can view the help of manage module with the following commands:
    python -m qlib.workflow.task.manage -h # show manual of manage module CLI
    python -m qlib.workflow.task.manage wait -h # show manual of the wait command of manage

    .. code-block:: shell

        python -m qlib.workflow.task.manage -t <pool_name> wait
        python -m qlib.workflow.task.manage -t <pool_name> task_stat


    .. note::

        Assumption: the data in MongoDB was encoded and the data out of MongoDB was decoded

    Here are four status which are:

        STATUS_WAITING: waiting for training

        STATUS_RUNNING: training

        STATUS_PART_DONE: finished some step and waiting for next step

        STATUS_DONE: all work done
    """

    STATUS_WAITING = "waiting"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_PART_DONE = "part_done"

    ENCODE_FIELDS_PREFIX = ["def", "res"]

    def __init__(self, task_pool: str):
        """
        Init Task Manager, remember to make the statement of MongoDB url and database name firstly.
        A TaskManager instance serves a specific task pool.
        The static method of this module serves the whole MongoDB.

        Parameters
        ----------
        task_pool: str
            the name of Collection in MongoDB
        """
        self.task_pool: pymongo.collection.Collection = getattr(get_mongodb(), task_pool)
        self.logger = get_module_logger(self.__class__.__name__)
        self.logger.info(f"task_pool:{task_pool}")

    @staticmethod
    def list() -> list:
        """
        List the all collection(task_pool) of the db.

        Returns:
            list
        """
        return get_mongodb().list_collection_names()

    def _encode_task(self, task):
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = Binary(pickle.dumps(task[k], protocol=C.dump_protocol_version))
        return task

    def _decode_task(self, task):
        """
        _decode_task is Serialization tool.
        Mongodb needs JSON, so it needs to convert Python objects into JSON objects through pickle

        Parameters
        ----------
        task : dict
            task information

        Returns
        -------
        dict
            JSON required by mongodb
        """
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = pickle.loads(task[k])
        return task

    def _dict_to_str(self, flt):
        return {k: str(v) for k, v in flt.items()}

    def _decode_query(self, query):
        """
        If the query includes any `_id`, then it needs `ObjectId` to decode.
        For example, when using TrainerRM, it needs query `{"_id": {"$in": _id_list}}`. Then we need to `ObjectId` every `_id` in `_id_list`.

        Args:
            query (dict): query dict. Defaults to {}.

        Returns:
            dict: the query after decoding.
        """
        if "_id" in query:
            if isinstance(query["_id"], dict):
                for key in query["_id"]:
                    query["_id"][key] = [ObjectId(i) for i in query["_id"][key]]
            else:
                query["_id"] = ObjectId(query["_id"])
        return query

    def replace_task(self, task, new_task):
        """
        Use a new task to replace a old one

        Args:
            task: old task
            new_task: new task
        """
        new_task = self._encode_task(new_task)
        query = {"_id": ObjectId(task["_id"])}
        try:
            self.task_pool.replace_one(query, new_task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            self.task_pool.replace_one(query, new_task)

    def insert_task(self, task):
        """
        Insert a task.

        Args:
            task: the task waiting for insert

        Returns:
            pymongo.results.InsertOneResult
        """
        try:
            insert_result = self.task_pool.insert_one(task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            insert_result = self.task_pool.insert_one(task)
        return insert_result

    def insert_task_def(self, task_def):
        """
        Insert a task to task_pool

        Parameters
        ----------
        task_def: dict
            the task definition

        Returns
        -------
        pymongo.results.InsertOneResult
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

    def create_task(self, task_def_l, dry_run=False, print_nt=False) -> List[str]:
        """
        If the tasks in task_def_l are new, then insert new tasks into the task_pool, and record inserted_id.
        If a task is not new, then just query its _id.

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
        List[str]
            a list of the _id of task_def_l
        """
        new_tasks = []
        _id_list = []
        for t in task_def_l:
            try:
                r = self.task_pool.find_one({"filter": t})
            except InvalidDocument:
                r = self.task_pool.find_one({"filter": self._dict_to_str(t)})
            # When r is none, it indicates that r s a new task
            if r is None:
                new_tasks.append(t)
                if not dry_run:
                    insert_result = self.insert_task_def(t)
                    _id_list.append(insert_result.inserted_id)
                else:
                    _id_list.append(None)
            else:
                _id_list.append(self._decode_task(r)["_id"])

        self.logger.info(f"Total Tasks: {len(task_def_l)}, New Tasks: {len(new_tasks)}")

        if print_nt:  # print new task
            for t in new_tasks:
                print(t)

        if dry_run:
            return []

        return _id_list

    def fetch_task(self, query={}, status=STATUS_WAITING) -> dict:
        """
        Use query to fetch tasks.

        Args:
            query (dict, optional): query dict. Defaults to {}.
            status (str, optional): [description]. Defaults to STATUS_WAITING.

        Returns:
            dict: a task(document in collection) after decoding
        """
        query = query.copy()
        query = self._decode_query(query)
        query.update({"status": status})
        task = self.task_pool.find_one_and_update(
            query, {"$set": {"status": self.STATUS_RUNNING}}, sort=[("priority", pymongo.DESCENDING)]
        )
        # null will be at the top after sorting when using ASCENDING, so the larger the number higher, the higher the priority
        if task is None:
            return None
        task["status"] = self.STATUS_RUNNING
        return self._decode_task(task)

    @contextmanager
    def safe_fetch_task(self, query={}, status=STATUS_WAITING):
        """
        Fetch task from task_pool using query with contextmanager

        Parameters
        ----------
        query: dict
            the dict of query

        Returns
        -------
        dict: a task(document in collection) after decoding
        """
        task = self.fetch_task(query=query, status=status)
        try:
            yield task
        except (Exception, KeyboardInterrupt):  # KeyboardInterrupt is not a subclass of Exception
            if task is not None:
                self.logger.info("Returning task before raising error")
                self.return_task(task, status=status)  # return task as the original status
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
        Query task in collection.
        This function may raise exception `pymongo.errors.CursorNotFound: cursor id not found` if it takes too long to iterate the generator

        python -m qlib.workflow.task.manage -t <your task pool> query '{"_id": "615498be837d0053acbc5d58"}'

        Parameters
        ----------
        query: dict
            the dict of query
        decode: bool

        Returns
        -------
        dict: a task(document in collection) after decoding
        """
        query = query.copy()
        query = self._decode_query(query)
        for t in self.task_pool.find(query):
            yield self._decode_task(t)

    def re_query(self, _id) -> dict:
        """
        Use _id to query task.

        Args:
            _id (str): _id of a document

        Returns:
            dict: a task(document in collection) after decoding
        """
        t = self.task_pool.find_one({"_id": ObjectId(_id)})
        return self._decode_task(t)

    def commit_task_res(self, task, res, status=STATUS_DONE):
        """
        Commit the result to task['res'].

        Args:
            task ([type]): [description]
            res (object): the result you want to save
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE. Defaults to STATUS_DONE.
        """
        # A workaround to use the class attribute.
        if status is None:
            status = TaskManager.STATUS_DONE
        self.task_pool.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": status, "res": Binary(pickle.dumps(res, protocol=C.dump_protocol_version))}},
        )

    def return_task(self, task, status=STATUS_WAITING):
        """
        Return a task to status. Always using in error handling.

        Args:
            task ([type]): [description]
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE. Defaults to STATUS_WAITING.
        """
        if status is None:
            status = TaskManager.STATUS_WAITING
        update_dict = {"$set": {"status": status}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def remove(self, query={}):
        """
        Remove the task using query

        Parameters
        ----------
        query: dict
            the dict of query

        """
        query = query.copy()
        query = self._decode_query(query)
        self.task_pool.delete_many(query)

    def task_stat(self, query={}) -> dict:
        """
        Count the tasks in every status.

        Args:
            query (dict, optional): the query dict. Defaults to {}.

        Returns:
            dict
        """
        query = query.copy()
        query = self._decode_query(query)
        tasks = self.query(query=query, decode=False)
        status_stat = {}
        for t in tasks:
            status_stat[t["status"]] = status_stat.get(t["status"], 0) + 1
        return status_stat

    def reset_waiting(self, query={}):
        """
        Reset all running task into waiting status. Can be used when some running task exit unexpected.

        Args:
            query (dict, optional): the query dict. Defaults to {}.
        """
        query = query.copy()
        # default query
        if "status" not in query:
            query["status"] = self.STATUS_RUNNING
        return self.reset_status(query=query, status=self.STATUS_WAITING)

    def reset_status(self, query, status):
        query = query.copy()
        query = self._decode_query(query)
        print(self.task_pool.update_many(query, {"$set": {"status": status}}))

    def prioritize(self, task, priority: int):
        """
        Set priority for task

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
        return (
            task_stat.get(self.STATUS_WAITING, 0)
            + task_stat.get(self.STATUS_RUNNING, 0)
            + task_stat.get(self.STATUS_PART_DONE, 0)
        )

    def _get_total(self, task_stat):
        return sum(task_stat.values())

    def wait(self, query={}):
        """
        When multiprocessing, the main progress may fetch nothing from TaskManager because there are still some running tasks.
        So main progress should wait until all tasks are trained well by other progress or machines.

        Args:
            query (dict, optional): the query dict. Defaults to {}.
        """
        task_stat = self.task_stat(query)
        total = self._get_total(task_stat)
        last_undone_n = self._get_undone_n(task_stat)
        if last_undone_n == 0:
            return
        self.logger.warning(f"Waiting for {last_undone_n} undone tasks. Please make sure they are running.")
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


def run_task(
    task_func: Callable,
    task_pool: str,
    query: dict = {},
    force_release: bool = False,
    before_status: str = TaskManager.STATUS_WAITING,
    after_status: str = TaskManager.STATUS_DONE,
    **kwargs,
):
    """
    While the task pool is not empty (has WAITING tasks), use task_func to fetch and run tasks in task_pool

    After running this method, here are 4 situations (before_status -> after_status):

        STATUS_WAITING -> STATUS_DONE: use task["def"] as `task_func` param, it means that the task has not been started

        STATUS_WAITING -> STATUS_PART_DONE: use task["def"] as `task_func` param

        STATUS_PART_DONE -> STATUS_PART_DONE: use task["res"] as `task_func` param, it means that the task has been started but not completed

        STATUS_PART_DONE -> STATUS_DONE: use task["res"] as `task_func` param

    Parameters
    ----------
    task_func : Callable
        def (task_def, **kwargs) -> <res which will be committed>
            the function to run the task
    task_pool : str
        the name of the task pool (Collection in MongoDB)
    query: dict
        will use this dict to query task_pool when fetching task
    force_release : bool
        will the program force to release the resource
    before_status : str:
        the tasks in before_status will be fetched and trained. Can be STATUS_WAITING, STATUS_PART_DONE.
    after_status : str:
        the tasks after trained will become after_status. Can be STATUS_WAITING, STATUS_PART_DONE.
    kwargs
        the params for `task_func`
    """
    tm = TaskManager(task_pool)

    ever_run = False

    while True:
        with tm.safe_fetch_task(status=before_status, query=query) as task:
            if task is None:
                break
            get_module_logger("run_task").info(task["def"])
            # when fetching `WAITING` task, use task["def"] to train
            if before_status == TaskManager.STATUS_WAITING:
                param = task["def"]
            # when fetching `PART_DONE` task, use task["res"] to train because the middle result has been saved to task["res"]
            elif before_status == TaskManager.STATUS_PART_DONE:
                param = task["res"]
            else:
                raise ValueError("The fetched task must be `STATUS_WAITING` or `STATUS_PART_DONE`!")
            if force_release:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    res = executor.submit(task_func, param, **kwargs).result()
            else:
                res = task_func(param, **kwargs)
            tm.commit_task_res(task, res, status=after_status)
            ever_run = True

    return ever_run


if __name__ == "__main__":
    # This is for using it in cmd
    # E.g. : `python -m qlib.workflow.task.manage list`
    auto_init()
    fire.Fire(TaskManager)
