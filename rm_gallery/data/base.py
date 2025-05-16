from typing import Sequence
from pydantic import BaseModel


class BaseData(BaseModel):
    pass

class BaseDataSet(BaseModel):
    datas: Sequence[BaseData]