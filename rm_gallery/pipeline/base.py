from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict, Any

from rm_gallery.data.base import BaseData
from rm_gallery.pipeline.node.base import Node


class RewardPipeline:

    nodes: list[tuple[str,Node]] = None

    def __init__(self,nodes: list[tuple[str,Node]]):
        self.nodes = nodes

    def run(self,data:BaseData):
        pass

    def add_node(self,block:tuple[str,Node]):
        pass

    def delete_node(self,block_name:str):
        pass

    @classmethod
    def from_dict(cls,datas:Dict[str,Any]):
        pass

    @classmethod
    def from_gallery(cls,path:str):
        return cls(nodes=None)
        pass

