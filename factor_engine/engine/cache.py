from abc import ABC, abstractmethod
from typing import Any, Optional, OrderedDict
from collections import OrderedDict

class Cache(ABC):
    """缓存系统的抽象基类接口。"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        根据键从缓存中获取值。

        Args:
            key: 缓存键。

        Returns:
            如果命中则返回缓存的值，否则返回 None。
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any):
        """
        将一个键值对存入缓存。

        Args:
            key: 缓存键。
            value: 要缓存的值。
        """
        pass

class InMemoryCache(Cache):
    """
    一个简单的基于内存的 LRU (Least Recently Used) 缓存实现。

    它使用 collections.OrderedDict 来跟踪项目的访问顺序。
    当缓存达到其最大容量时，最近最少使用的项目将被丢弃。
    """

    def __init__(self, max_size: int = 128):
        """
        初始化内存缓存。

        Args:
            max_size: 缓存可以存储的最大项目数。
        """
        if max_size <= 0:
            raise ValueError("max_size 必须为正数")
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """
        获取一个项目。如果命中，则将其移动到有序字典的末尾（标记为最近使用）。
        """
        if key not in self._cache:
            return None
        
        # 将被访问的项移到末尾，表示最近使用过
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any):
        """
        设置一个项目。如果缓存已满，则移除最近最少使用的项目。
        """
        if key in self._cache:
            # 如果键已存在，先删除旧的，再插入新的，以更新其位置
            del self._cache[key]
        
        self._cache[key] = value
        
        # 检查是否超出容量
        if len(self._cache) > self.max_size:
            # 弹出第一个插入的项 (最近最少使用的)
            self._cache.popitem(last=False)

    def clear(self):
        """清空整个缓存。"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache 