
from enum import Enum
from typing import Any, List
from pydantic import BaseModel, Field
from src.data.data_schema import ContentList, EvaluationSample
from src.pipeline.node.base import Node
from src.utils.data import get_value_by_path


class VarType(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class Var(BaseModel):
    name: str = Field(default=...)
    path: str = Field(default=...)
    vtype: VarType = Field(default=VarType.OUTPUT)
    default: Any | None = Field(default=None)


class Rule(BaseModel):
    desc: str = Field(default=..., description="rule description")
    score: float | int | str = Field(default=..., description="scorer")


class BaseParser(Node):
    inputs: List[Var] = Field(default=[])
    name: str = Field(default=...)

    # 处理提取数据层
    def prepare_params(self, sample_input: EvaluationSample, sample_output: ContentList | List[ContentList]) -> dict:
        params = {}
        for var in self.inputs:
            if var.vtype == VarType.INPUT:
                params[var.name] = get_value_by_path(sample_input, var.path, var.default)
            elif var.vtype == VarType.OUTPUT:
                if isinstance(sample_output, list):
                    var_list = []
                    for output in sample_output:
                        var_list.append(get_value_by_path(output, var.path, var.default))
                    params[var.name] = var_list
                else:
                    params[var.name] = get_value_by_path(sample_output, var.path, var.default)
        return params

    def _run(self, **kwargs) -> Any:
        ...

    def run_pointwise(self, sample: EvaluationSample):
        # TODO parallel
        for output in sample.outputs:
            kwargs = self.prepare_params(sample_input=sample, sample_output=output)
            context = self._run(**kwargs)
            output.evaluation_contexts[self.name] = context
            
    def run_listwise(self, sample: EvaluationSample):
        kwargs = self.prepare_params(sample_input=sample, sample_output=sample.outputs)
        context = self._run(**kwargs)
        sample.evaluation_contexts[self.name] = context

    def run(self, **kwargs) -> Any:
        return None
