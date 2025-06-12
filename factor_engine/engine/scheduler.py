import collections
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Callable, List, Optional

from factor_engine.core.dag import DAGNode, NodeStatus
from factor_engine.data_layer.loader import DataProvider
from factor_engine.engine.cache import Cache
from factor_engine.engine.utils import generate_cache_key
from .context import ExecutionContext


class Scheduler:
    """
    负责高效、并行地执行由 ExecutionPlanner 生成的 DAG。

    该调度器采用基于入度(in-degree)的拓扑排序算法，并利用线程池
    来实现节点计算的并行化。执行过程由任务完成后的回调驱动，
    是事件驱动和异步的。
    """

    def __init__(self, data_provider: DataProvider, operators: Dict[str, Callable], cache: Optional[Cache] = None, max_workers: Optional[int] = None):
        """
        初始化调度器。

        Args:
            data_provider: 用于加载底层数据的数据提供者。
            operators: 一个字典，将操作符名称 (e.g., 'op_add') 映射到可调用的函数。
            cache: 用于存储计算结果的缓存对象。
            max_workers: 线程池中的最大工作线程数。默认为 None，由 ThreadPoolExecutor 决定。
        """
        self.data_provider = data_provider
        self.operators = operators
        self.cache = cache
        self.max_workers = max_workers
        
        # 运行时状态，每次执行时都会重置
        self._lock = threading.Lock()
        self._completion_event: Optional[threading.Event] = None
        self._node_results: Dict[str, Any] = {}
        self._root_node: Optional[DAGNode] = None
        self._context: Optional[ExecutionContext] = None
        self._execution_error: Optional[Exception] = None

    def execute(self, root_node: DAGNode, context: ExecutionContext) -> Any:
        """
        执行整个 DAG 计算。这是一个阻塞操作，直到计算完成或失败。

        Args:
            root_node: DAG 的根节点。
            context: 本次执行的上下文，包含时间范围和股票列表。

        Returns:
            根节点的最终计算结果。
        
        Raises:
            Exception: 如果在执行过程中有任何节点计算失败，则会抛出异常。
        """
        # 1. 初始化执行状态
        self._root_node = root_node
        self._context = context
        self._completion_event = threading.Event()
        self._node_results.clear()
        self._execution_error = None

        all_nodes = self._collect_nodes(root_node)
        self._prepare_nodes_for_execution(all_nodes)
        
        ready_queue = collections.deque(
            [node for node in all_nodes if node.in_degree == 0 and node.status != NodeStatus.CACHED]
        )

        # 2. 如果根节点已经被缓存，直接返回结果
        if root_node.status == NodeStatus.CACHED:
            return root_node.result_ref

        # 3. 如果没有可执行的节点，但根节点未完成，说明有问题
        if not ready_queue and root_node.status != NodeStatus.COMPLETED:
             raise RuntimeError("DAG 无效：没有入度为0的节点可开始执行。")

        # 4. 并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for node in ready_queue:
                self._submit_task(node, executor)
            
            # 5. 等待根节点计算完成或任何节点失败
            self._completion_event.wait()

        # 6. 如果在执行过程中发生错误，则抛出异常
        if self._execution_error:
            raise self._execution_error

        # 7. 返回结果
        return self._node_results.get(self._root_node.id)

    def _submit_task(self, node: DAGNode, executor: ThreadPoolExecutor):
        """提交一个任务到线程池执行。"""
        node.status = NodeStatus.RUNNING
        future = executor.submit(self._run_task, node)
        future.add_done_callback(
            functools.partial(self._task_done_callback, node, executor)
        )

    def _run_task(self, node: DAGNode) -> Any:
        """
        在工作线程中实际执行单个节点的计算任务。
        这是一个独立的单元，不应有任何副作用，只负责计算并返回值。
        """
        # 'literal' 节点是 planner 用来处理常数参数的内部节点，直接返回值
        if node.operator == 'literal':
             return node.args[0]

        # 准备计算所需的参数
        args = []
        for arg in node.args:
            if isinstance(arg, DAGNode):
                # 如果参数是另一个节点，从结果字典中获取其计算结果
                args.append(self._node_results[arg.id])
            else:
                # 否则，是字面量参数 (e.g., window size)
                args.append(arg)
        
        # 根据操作符执行计算
        if node.operator == 'load_data':
            field = node.args[0]
            return self.data_provider.load(
                field=field,
                start_date=self._context.start_date,
                end_date=self._context.end_date,
                stocks=self._context.stocks,
            )
        elif node.operator in self.operators:
            op_func = self.operators[node.operator]
            return op_func(*args, **node.kwargs)
        else:
            raise ValueError(f"调度器无法执行未知的操作符: '{node.operator}' for node '{node.expression}'")

    def _task_done_callback(self, node: DAGNode, executor: ThreadPoolExecutor, future: Future):
        """
        任务完成后的回调函数。
        - 保存结果
        - 处理异常
        - 触发依赖此节点的父节点的执行
        """
        try:
            result = future.result()
            self._node_results[node.id] = result
            node.result_ref = result  # 也将引用保存在节点中
            node.status = NodeStatus.COMPLETED
            
            # --- 缓存写入 ---
            if self.cache:
                # 'literal' 节点的结果是常数，不应被缓存
                if node.operator != 'literal':
                    cache_key = generate_cache_key(node.expression, self._context)
                    self.cache.set(cache_key, result)

            # 检查是否为根节点，如果是则发出完成信号
            if node.id == self._root_node.id:
                self._completion_event.set()
                return

            # 传播给父节点，在锁内进行以保证线程安全
            with self._lock:
                for parent in node.parents:
                    parent.in_degree -= 1
                    if parent.in_degree == 0 and parent.status == NodeStatus.PENDING:
                        parent.status = NodeStatus.READY
                        self._submit_task(parent, executor)

        except Exception as e:
            node.status = NodeStatus.FAILED
            # 捕获第一个发生的异常
            with self._lock:
                if not self._execution_error:
                    self._execution_error = e
            # 任何节点失败，都立即停止整个执行
            self._completion_event.set()

    @staticmethod
    def _collect_nodes(root_node: DAGNode) -> List[DAGNode]:
        """从根节点开始，通过广度优先搜索收集 DAG 中的所有节点。"""
        all_nodes = set()
        queue = collections.deque([root_node])
        visited = {root_node.id}
        
        while queue:
            node = queue.popleft()
            all_nodes.add(node)
            for dep in node.dependencies:
                if dep.id not in visited:
                    visited.add(dep.id)
                    queue.append(dep)
        return list(all_nodes)
    
    @staticmethod
    def _prepare_nodes_for_execution(nodes: List[DAGNode]):
        """重置所有节点的状态并计算其初始入度，为新的执行做准备。"""
        for node in nodes:
            # 如果节点已经被 planner 标记为已缓存，则跳过状态重置
            if node.status == NodeStatus.CACHED:
                continue
            
            # 只重置执行相关的状态
            node.status = NodeStatus.PENDING
            node.update_in_degree() 