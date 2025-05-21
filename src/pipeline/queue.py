from threading import Lock

from src.pipeline.node.base import Node


class ThreadSafeDict:


    def __init__(self,nodes:list[tuple[str,Node]],lock:Lock):
        self.nodes = {}
        self.lock = lock
        with self.lock:
            for name,node in nodes:
                self.nodes[name] = node

    def push(self,node:Node):
        pass

    def pop(self):
        pass

    def get(self):
        pass

    def get_next_ready(self):
        pass

    def get_ready_cnt(self):
        pass

    def get_ready_wait_cnt(self) -> int:
        pass

    def update_status(self):
        pass




