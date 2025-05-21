from typing import List, Dict, Any, Optional, TypeVar
from loguru import logger
import importlib

from data_schema import EvaluationSample
from data_load_strategy import DataLoadStrategyRegistry
from base import BaseData
from base_operator import Operator, OperatorFactory
from ops import register_all_operators # 导入ops模块，这会自动注册所有算子

T = TypeVar('T', bound=BaseData)

class DataJuicerOperator(Operator[T]):
    """
    Adapter class to integrate data-juicer operators into our system
    """
    def __init__(self, 
                 name: str,
                 juicer_op_class: Any,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.juicer_op = juicer_op_class(**config) if config else juicer_op_class()

    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process the entire dataset using data-juicer operators
        """
        try:
            # Extract all texts and create a mapping to track which item they belong to
            all_texts = []
            text_to_item_indices = {}  # Map text to list of item indices
            
            for i, item in enumerate(items):
                # Process input texts
                if item.evaluation_sample.input:
                    for input_item in item.evaluation_sample.input:
                        if input_item.content:
                            all_texts.append(input_item.content)
                            if input_item.content not in text_to_item_indices:
                                text_to_item_indices[input_item.content] = []
                            text_to_item_indices[input_item.content].append(i)
                
                # Process output texts
                if item.evaluation_sample.outputs:
                    for output_item in item.evaluation_sample.outputs:
                        if output_item.content:
                            all_texts.append(output_item.content)
                            if output_item.content not in text_to_item_indices:
                                text_to_item_indices[output_item.content] = []
                            text_to_item_indices[output_item.content].append(i)

            if not all_texts:
                return items

            # Create a dataset-like structure for data-juicer
            sample = {
                'text': all_texts,
                '__dj__stats__': [{} for _ in range(len(all_texts))]
            }

            # Process the sample
            processed_sample = self.juicer_op.compute_stats_batched(sample)
            keep_indices = list(self.juicer_op.process_batched(processed_sample))

            # Track which items should be kept
            items_to_keep = set()
            for i, (text, keep) in enumerate(zip(all_texts, keep_indices)):
                if keep:
                    items_to_keep.update(text_to_item_indices[text])

            # Return only the items that should be kept
            return [items[i] for i in range(len(items)) if i in items_to_keep]

        except Exception as e:
            logger.error(f"Error in dataset-level processing with operator {self.name}: {str(e)}")
            return items


class FilterOperator(Operator[T]):
    """
    Filter operator for filtering items based on criteria
    """
    def __init__(self, 
                 name: str,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        logger.info(f"Initializing FilterOperator: {name} with config: {config}")

    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process the dataset using the configured filter
        """
        try:
            # Check if there's a registered operator creator for this name
            if self.name in OperatorFactory._operator_registry:
                # Create a new operator instance with the same config
                operator = OperatorFactory._operator_registry[self.name]({
                    'type': 'filter',
                    'name': self.name,
                    'config': self.config
                })
                # Use the registered operator's process_dataset method
                return operator.process_dataset(items)
            
            logger.warning(f"No registered operator found for name: {self.name}")
            return items
        except Exception as e:
            logger.error(f"Error in filter operation {self.name}: {str(e)}")
            return items


class MapOperator(Operator[T]):
    """
    Map operator for transforming items
    """
    def __init__(self, 
                 name: str,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    def process_dataset(self, items: List[T]) -> List[T]:
        try:
            # Check if there's a registered operator creator for this name
            if self.name in OperatorFactory._operator_registry:
                # Create a new operator instance with the same config
                operator = OperatorFactory._operator_registry[self.name]({
                    'type': 'map',
                    'name': self.name,
                    'config': self.config
                })
                # Use the registered operator's process_dataset method
                return operator.process_dataset(items)
            
            logger.warning(f"No registered operator found for name: {self.name}")
            return items
        except Exception as e:
            logger.error(f"Error in filter operation {self.name}: {str(e)}")
            return items


class RewardOperator(Operator[T]):
    """
    Reward operator for processing rewards
    """
    def __init__(self, 
                 name: str,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    def process_dataset(self, items: List[T]) -> List[T]:
        try:
            # Check if there's a registered operator creator for this name
            if self.name in OperatorFactory._operator_registry:
                # Create a new operator instance with the same config
                operator = OperatorFactory._operator_registry[self.name]({
                    'type': 'reward',
                    'name': self.name,
                    'config': self.config
                })
                # Use the registered operator's process_dataset method
                return operator.process_dataset(items)
            
            logger.warning(f"No registered operator found for name: {self.name}")
            return items
        except Exception as e:
            logger.error(f"Error in filter operation {self.name}: {str(e)}")
            return items


class DataPipeline:
    """
    Main data processing pipeline that orchestrates data loading and processing
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.operators: List[Operator] = []
        
    def add_operator(self, operator: Operator) -> 'DataPipeline':
        """
        Add an operator to the pipeline
        """
        self.operators.append(operator)
        return self
        
    def process(self, items: List[T]) -> List[T]:
        """
        Process all items through the pipeline
        """
        processed_items = items.copy()
        
        # First apply dataset-level processing
        for operator in self.operators:
            try:
                processed_items = operator.process_dataset(processed_items)
            except Exception as e:
                logger.error(f"Error in dataset-level processing with operator {operator}: {str(e)}")
                continue
        
        return processed_items
        
    def run(self, items: List[T]) -> List[T]:
        """
        Run the complete pipeline: process data
        """
        return self.process(items)

    def __str__(self) -> str:
        return f"DataPipeline({self.name}, operators={[str(op) for op in self.operators]})"
