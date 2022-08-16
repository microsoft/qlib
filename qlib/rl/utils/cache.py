from __future__ import annotations

import collections


class LRUCache:
    def __init__(self, pool_size: int = 200) -> None:
        self.pool_size = pool_size
        self.contents: dict = {}
        self.keys: collections.deque = collections.deque()

    def put(self, key, item):
        if self.has(key):
            self.keys.remove(key)
        self.keys.append(key)
        self.contents[key] = item
        while len(self.contents) > self.pool_size:
            self.contents.pop(self.keys.popleft())

    def get(self, key):
        return self.contents[key]

    def has(self, key):
        return key in self.contents
