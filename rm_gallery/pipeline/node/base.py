from typing import Any, Dict
from pydantic import BaseModel

class Node(BaseModel):

    def check_in(self, **kwargs) -> bool:
        pass

    def run(self, **kwargs) -> Any:
        pass

    def __or__(self, other):
        pass

    def __ror__(self, other):
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        pass

    @classmethod
    def from_gallery(cls, path):
        pass

    @classmethod
    def from_string(cls,data:str):
        pass
