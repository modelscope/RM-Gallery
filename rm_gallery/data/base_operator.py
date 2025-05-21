from typing import List, Dict, Any, Optional, Callable, Union, TypeVar, Generic, Type
from abc import ABC, abstractmethod
from loguru import logger
import importlib

from base import BaseData

T = TypeVar('T', bound=BaseData)

class Operator(Generic[T], ABC):
    """
    Base class for all data processing operators
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process the entire dataset
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class OperatorFactory:
    """
    Factory class for creating operators from configuration
    """
    _operator_registry: Dict[str, Callable[[Dict[str, Any]], Operator]] = {}

    @classmethod
    def register(cls, operator_type: str) -> Callable:
        """
        Decorator for registering operator creation functions.
        
        Args:
            operator_type: The type identifier for the operator
            
        Returns:
            A decorator function that registers the operator creation function
        """
        def decorator(func: Callable[[Dict[str, Any]], Operator]) -> Callable:
            cls._operator_registry[operator_type] = func
            return func
        return decorator

    @classmethod
    def create_operator(cls, operator_config: Dict[str, Any]) -> Operator:
        """
        Create an operator from configuration
        """
        op_type = operator_config.get('type')
        name = operator_config.get('name', op_type)
        config = operator_config.get('config', {})
        
        # Check if there's a registered operator creator for this type or name
        if op_type in cls._operator_registry:
            return cls._operator_registry[op_type](operator_config)
        elif name in cls._operator_registry:
            return cls._operator_registry[name](operator_config)
        
        # Fall back to built-in operator types
        if op_type == 'filter':
            from data_processor import FilterOperator
            return FilterOperator(
                name=name,
                config=config
            )
        elif op_type == 'map':
            from data_processor import MapOperator
            return MapOperator(
                name=name,
                config=config
            )
        elif op_type == 'reward':
            from data_processor import RewardOperator
            return RewardOperator(
                name=name,
                config=config
            )
        elif op_type == 'data_juicer':
            # Convert name to CamelCase (e.g., perplexity_filter -> PerplexityFilter)
            operator_name = ''.join(word.capitalize() for word in name.split('_'))
            operator_module = importlib.import_module(f"data_juicer.ops.filter.{name.lower()}")
            operator_class = getattr(operator_module, operator_name)
            
            from data_processor import DataJuicerOperator
            return DataJuicerOperator(
                name=operator_name,
                juicer_op_class=operator_class,
                config=config
            )
        else:
            raise ValueError(f"Unknown operator type: {op_type}") 