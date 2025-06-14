import collections
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, List, Optional, Deque

from factor_engine.core.dag import DAGNode, NodeStatus
from factor_engine.engine.cache import Cache
from factor_engine.engine.utils import generate_cache_key
from .context import ExecutionContext
from ..registry import op_registry


class Scheduler:
    """
    负责以并行或串行方式执行由 ExecutionPlanner 生成的 DAG。
    """
    def __init__(
        self,
        data_provider: Any, # Changed to Any to avoid circular dependency with DataProvider
        cache: Optional[Cache] = None,
        max_workers: int = 4,
        verbose: bool = False
    ):
        """
        Args:
            data_provider: 用于加载基础字段数据的 DataProvider 实例。
            cache: 用于存储和检索中间结果的缓存对象。
            max_workers: 线程池中的最大工作线程数。
                         如果设置为 1 或 0, 将以单线程同步模式执行，便于调试。
            verbose: 如果为 True, 将在执行期间打印详细的状态信息。
        """
        self._data_provider = data_provider
        self._cache = cache
        self._max_workers = max_workers
        self._verbose = verbose
        
        if self._verbose:
            print(f"[Scheduler] Initialized with cache: {type(cache)} ({cache})")
        
        # 如果 max_workers <= 1, 则切换到同步（单线程）模式
        self._synchronous = max_workers <= 1

        # --- 运行时状态 ---
        self._lock = threading.Lock()
        self._completion_event: Optional[threading.Event] = None
        self._node_results: Dict[str, Any] = {}
        self._root_node: Optional[DAGNode] = None
        self._context: Optional[ExecutionContext] = None
        self._execution_errors: List[Exception] = []
        self._executor: Optional[ThreadPoolExecutor] = None

    def execute(self, root_node: DAGNode, context: ExecutionContext, timeout: Optional[float] = None) -> Any:
        """
        执行整个 DAG 计算。这是一个阻塞操作，直到计算完成或失败。
        """
        if root_node.status == NodeStatus.CACHED:
            if self._verbose:
                print(f"[Scheduler] Root node '{root_node.expression}' was already cached. Returning result immediately.")
            return root_node.result_ref

        # 1. 初始化执行状态
        self._root_node = root_node
        self._context = context
        self._completion_event = threading.Event()
        self._node_results.clear()
        self._execution_errors.clear()

        all_nodes = self._collect_nodes(root_node)
        self._prepare_nodes_for_execution(all_nodes)
        
        # 根据执行模式调用不同的路径
        if self._synchronous:
            if self._verbose:
                print("[Scheduler] Running in synchronous (single-threaded) mode.")
            return self._execute_synchronously(all_nodes)
        else:
            if self._verbose:
                print(f"[Scheduler] Running in parallel mode with {self._max_workers} workers.")
            return self._execute_in_parallel(all_nodes, timeout)

    def _execute_synchronously(self, all_nodes: List[DAGNode]) -> Any:
        """以单线程、同步的方式执行 DAG。"""
        ready_queue = collections.deque(
            [node for node in all_nodes if node.in_degree == 0 and node.status != NodeStatus.CACHED]
        )

        # 处理缓存的节点 - 将它们的结果添加到结果字典中
        for node in all_nodes:
            if node.status == NodeStatus.CACHED:
                self._node_results[node.id] = node.result_ref
                if self._verbose:
                    print(f"[Scheduler] Using cached result for node: '{node.expression}'")

        # 如果根节点已经缓存，直接返回
        if self._root_node.status == NodeStatus.CACHED:
            return self._root_node.result_ref

        while ready_queue:
            node = ready_queue.popleft()
            
            try:
                result = self._execute_node(node)
                self._node_results[node.id] = result
                node.status = NodeStatus.COMPLETED
                self._write_to_cache(node, result)

                for parent in node.parents:
                    parent.in_degree -= 1
                    if parent.in_degree == 0:
                        ready_queue.append(parent)

            except Exception as e:
                node.status = NodeStatus.FAILED
                if self._verbose:
                    print(f"[Scheduler] Node '{node.expression}' failed with error: {e}")
                raise RuntimeError(f"Execution failed at node '{node.expression}'") from e

        if self._root_node.status != NodeStatus.COMPLETED:
            raise RuntimeError("Execution finished, but the root node was not completed. Check for cycles or logic errors in the DAG.")

        return self._node_results[self._root_node.id]

    def _execute_in_parallel(self, all_nodes: List[DAGNode], timeout: Optional[float]) -> Any:
        """使用线程池以并行方式执行 DAG。"""
        ready_queue = collections.deque(
            [node for node in all_nodes if node.in_degree == 0 and node.status != NodeStatus.CACHED]
        )

        # 处理缓存的节点 - 将它们的结果添加到结果字典中
        for node in all_nodes:
            if node.status == NodeStatus.CACHED:
                self._node_results[node.id] = node.result_ref
                if self._verbose:
                    print(f"[Scheduler] Using cached result for node: '{node.expression}'")

        # 如果根节点已经缓存，直接返回
        if self._root_node.status == NodeStatus.CACHED:
            return self._root_node.result_ref

        if not ready_queue and self._root_node.status != NodeStatus.COMPLETED:
             raise RuntimeError(f"No nodes are ready to run, but the root node '{self._root_node.expression}' is not completed. Check for cycles in the DAG.")

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            self._executor = executor
            for node in ready_queue:
                self._submit_node(node)

            completed_in_time = self._completion_event.wait(timeout)

            if not completed_in_time:
                self._executor = None # Clear reference
                raise TimeoutError(f"Execution timed out after {timeout} seconds. Active errors: {self._execution_errors}")

        self._executor = None # Clear reference

        if self._execution_errors:
            # Maybe raise a custom exception that holds all errors
            raise self._execution_errors[0]

        return self._node_results[self._root_node.id]

    def _submit_node(self, node: DAGNode):
        """提交一个节点到线程池执行。"""
        if self._verbose:
            print(f"[Scheduler] Submitting node: '{node.expression}'")
        
        future = self._executor.submit(self._execute_node, node)
        future.add_done_callback(functools.partial(self._task_done_callback, node))

    def _execute_node(self, node: DAGNode) -> Any:
        """在工作线程中实际执行单个节点的计算任务。"""
        if self._verbose:
            print(f"[Scheduler] Executing node: '{node.expression}'")

        if node.operator == 'literal':
            return node.args[0]
        
        if node.operator == 'load_data':
            field = node.args[0]
            return self._data_provider.load(
                field=field,
                start_date=self._context.start_date,
                end_date=self._context.end_date,
                stocks=self._context.stocks,
            )

        # 修复：正确处理参数，避免访问未准备好的结果
        args = []
        for arg in node.args:
            if isinstance(arg, DAGNode):
                # 如果是DAGNode，我们需要等待它的结果
                with self._lock:
                    if arg.id not in self._node_results:
                        # 这不应该发生，因为DAG的拓扑排序应该确保依赖项先完成
                        raise RuntimeError(f"Dependency node '{arg.expression}' result not available for '{node.expression}'")
                    args.append(self._node_results[arg.id])
            else:
                # 直接使用字面量或其他值
                args.append(arg)

        # 获取操作符并执行
        op = op_registry.get(node.operator, **node.kwargs)
        return op(*args)

    def _task_done_callback(self, node: DAGNode, future: Future):
        """任务完成后的回调函数。"""
        try:
            result = future.result()
            with self._lock:
                if self._verbose:
                    print(f"[Scheduler] _task_done_callback: cache is {type(self._cache)} ({self._cache})")
                
                if self._execution_errors: # An error has occurred, stop processing
                    return

                self._node_results[node.id] = result
                node.status = NodeStatus.COMPLETED
                self._write_to_cache(node, result)

                if self._verbose:
                    print(f"[Scheduler] Completed node: '{node.expression}'")

                # 检查并提交准备好的父节点
                for parent in node.parents:
                    parent.in_degree -= 1
                    if parent.in_degree == 0:
                        # 确保所有依赖的结果都已准备好
                        all_deps_ready = all(
                            dep.status == NodeStatus.COMPLETED or dep.status == NodeStatus.CACHED
                            for dep in parent.dependencies
                        )
                        if all_deps_ready:
                            self._submit_node(parent)
                        else:
                            if self._verbose:
                                print(f"[Scheduler] Parent node '{parent.expression}' dependencies not all ready yet")

                if self._root_node.status == NodeStatus.COMPLETED:
                    self._completion_event.set()

        except Exception as e:
            with self._lock:
                node.status = NodeStatus.FAILED
                self._execution_errors.append(e)
                if self._verbose:
                    print(f"[Scheduler] Node '{node.expression}' failed with error: {e}")
                # Trigger completion to stop waiting, execution has failed
                self._completion_event.set()

    def _write_to_cache(self, node: DAGNode, result: Any):
        if self._verbose:
            print(f"[Scheduler] _write_to_cache called for node: '{node.expression}', operator: {node.operator}")
            print(f"[Scheduler] self._cache is: {self._cache}")
            print(f"[Scheduler] bool(self._cache): {bool(self._cache)}")
            print(f"[Scheduler] node.operator != 'literal': {node.operator != 'literal'}")
        
        if self._cache is not None and node.operator != 'literal':
            cache_key = generate_cache_key(node.expression, self._context)
            if self._verbose:
                print(f"[Scheduler] Writing to cache with key: {cache_key}")
            self._cache.set(cache_key, result)
            if self._verbose:
                print(f"[Scheduler] Cache size after write: {len(self._cache)}")
        elif self._cache is None:
            if self._verbose:
                print(f"[Scheduler] No cache available")
        elif node.operator == 'literal':
            if self._verbose:
                print(f"[Scheduler] Skipping cache for literal node")
        else:
            if self._verbose:
                print(f"[Scheduler] Cache write conditions not met")

    @staticmethod
    def _collect_nodes(root_node: DAGNode) -> List[DAGNode]:
        """从根节点开始，通过广度优先搜索收集 DAG 中的所有节点。"""
        all_nodes = set()
        queue = collections.deque([root_node])
        visited = {root_node}
        
        while queue:
            node = queue.popleft()
            all_nodes.add(node)
            for dep in node.dependencies:
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
        return list(all_nodes)

    @staticmethod
    def _prepare_nodes_for_execution(nodes: List[DAGNode]):
        """重置所有节点的状态以准备一次新的执行。"""
        for node in nodes:
            if node.status == NodeStatus.CACHED:
                continue
            
            node.status = NodeStatus.PENDING
            node.update_in_degree() 