"""
Data Processor Module - Unified data processing functionality
"""

from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.process.ops.base import BaseOperator
from rm_gallery.core.data.schema import BaseDataSet, DataSample


class DataProcess(BaseDataModule):
    """Data process module - process data"""

    operators: List[BaseOperator] = Field(
        default_factory=list, description="operators list"
    )

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        operators: Optional[List[BaseOperator]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            module_type=DataModuleType.PROCESS,
            name=name,
            config=config,
            operators=operators or [],
            metadata=metadata,
            **kwargs,
        )

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample]], **kwargs
    ) -> BaseDataSet:
        """Process data through the operator pipeline"""
        try:
            data_samples = self._prepare_data(input_data)
            processed_data = data_samples

            # Preserve original dataset metadata if available
            original_metadata = {}
            if isinstance(input_data, BaseDataSet):
                original_metadata = input_data.metadata or {}

            logger.info(
                f"Processing {len(data_samples)} items with {len(self.operators)} operators"
            )

            # Apply operators sequentially
            for i, operator in enumerate(self.operators):
                try:
                    logger.info(
                        f"Applying operator {i + 1}/{len(self.operators)}: {operator.name}"
                    )
                    processed_data = operator.process_dataset(processed_data)
                    logger.info(
                        f"Operator {operator.name} completed: {len(processed_data)} items remaining"
                    )
                except Exception as e:
                    logger.error(f"Error in operator {operator.name}: {str(e)}")
                    continue

            # Merge original metadata with processing metadata
            combined_metadata = original_metadata.copy()
            combined_metadata.update(
                {
                    "original_count": len(data_samples),
                    "processed_count": len(processed_data),
                    "operators_applied": [op.name for op in self.operators],
                }
            )

            # Create output dataset with preserved metadata
            output_dataset = BaseDataSet(
                name=f"{self.name}_processed",
                metadata=combined_metadata,
                datas=processed_data,
            )

            logger.info(
                f"Processing completed: {len(data_samples)} -> {len(processed_data)} items"
            )
            return output_dataset

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise e

    def _prepare_data(
        self, input_data: Union[BaseDataSet, List[DataSample]]
    ) -> List[DataSample]:
        """Prepare data for processing"""
        if isinstance(input_data, BaseDataSet):
            return list(input_data.datas)
        return input_data

    def get_operators_info(self) -> List[Dict[str, Any]]:
        """Get information about all operators"""
        return [
            {"name": op.name, "type": op.__class__.__name__, "config": op.config}
            for op in self.operators
        ]


def create_process_module(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    operators: Optional[List[BaseOperator]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataProcess:
    """Create data process module factory function"""
    return DataProcess(name=name, config=config, operators=operators, metadata=metadata)
