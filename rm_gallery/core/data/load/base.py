"""
Data Load Module - load data from various data sources
"""
import fnmatch
import json
import random
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import BaseDataSet, DataSample


@dataclass(frozen=True)
class StrategyKey:
    """
    Immutable key for strategy registration with wildcard support
    """

    data_type: str
    data_source: str

    def matches(self, other: "StrategyKey") -> bool:
        """
        Check if this key matches another key with wildcard support
        """
        return fnmatch.fnmatch(other.data_type, self.data_type) and fnmatch.fnmatch(
            other.data_source, self.data_source
        )


class DataLoad(BaseDataModule):
    """
    Unified Data Load Module - serves as both the main data loading module and base class for loading strategies
    """

    load_strategy_type: str = Field(
        default="local", description="data load strategy type (local or remote)"
    )
    data_source: str = Field(default="*", description="data source")

    def __init__(
        self,
        name: str,
        load_strategy_type: str = "local",
        data_source: str = "*",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize data load module

        Args:
            name: module name
            load_strategy_type: load strategy type
            data_source: data source
            config: load config
            metadata: metadata for the module
        """
        super().__init__(
            module_type=DataModuleType.LOAD,
            name=name,
            load_strategy_type=load_strategy_type,
            data_source=data_source,
            config=config or {},
            metadata=metadata,
            **kwargs,
        )
        self.validate_config(config or {})

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary
        Override this method in subclasses to add specific validation rules
        """
        pass

    def load_data(self, **kwargs) -> List[DataSample]:
        """
        Load data from the source and return a list of DataSample objects
        Default implementation uses strategy pattern for different data sources
        """
        # If this is a strategy instance (subclass), call the abstract method
        if self.__class__ != DataLoad:
            return self._load_data_impl(**kwargs)

        # Otherwise, use strategy pattern to find appropriate loader
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            data_type=self.load_strategy_type, data_source=self.data_source
        )

        if not strategy_class:
            error_msg = f"No suitable data load strategy found for type: {self.load_strategy_type}, source: {self.data_source}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize and use strategy to load data
        strategy_config = self.config.copy()
        strategy = strategy_class(
            name=f"{self.name}_strategy", config=strategy_config, metadata=self.metadata
        )
        return strategy.load_data(**kwargs)

    def _load_data_impl(self, **kwargs) -> List[DataSample]:
        """
        Abstract method for strategy implementations to override
        """
        raise NotImplementedError("Subclasses must implement _load_data_impl method")

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs
    ) -> BaseDataSet:
        """Load data and return as BaseDataSet"""
        try:
            # Load data using strategy
            loaded_items = self.load_data(**kwargs)

            # Convert loaded items to DataSample objects if needed
            data_samples = []
            for item in loaded_items:
                data_samples.append(item)

            # Apply limit (if specified)
            if (
                "limit" in self.config
                and self.config["limit"] is not None
                and self.config["limit"] > 0
            ):
                limit = min(int(self.config["limit"]), len(data_samples))
                data_samples = random.sample(data_samples, limit)
                logger.info(
                    f"Applied limit of {limit}, final count: {len(data_samples)}"
                )

            # Create output dataset
            dataset_name = self.name
            if dataset_name.endswith("-loader"):
                dataset_name = dataset_name[:-7]

            output_dataset = BaseDataSet(
                name=dataset_name,
                metadata={
                    "source": self.data_source,
                    "strategy_type": self.load_strategy_type,
                    "config": self.config,
                },
                datas=data_samples,
            )
            logger.info(
                f"Successfully loaded {len(data_samples)} items from {self.data_source}"
            )

            return output_dataset
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


class DataLoadStrategyRegistry:
    """
    Registry for data load strategies with wildcard matching
    """

    _strategies: Dict[StrategyKey, Type[DataLoad]] = {}

    @classmethod
    def get_strategy_class(
        cls, data_type: str, data_source: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Type[DataLoad]]:
        """
        Retrieve the most specific matching strategy
        """
        logger.info(
            f"Getting strategy class for data_type: {data_type}, data_source: {data_source}"
        )

        # Default to wildcard if not provided
        data_type = data_type or "*"
        data_source = data_source or "*"

        # Create the lookup key
        lookup_key = StrategyKey(data_type=data_type, data_source=data_source)

        # First, check for exact match
        exact_match = cls._strategies.get(lookup_key)
        if exact_match:
            return exact_match

        # Find all matching wildcard strategies
        matching_strategies = []
        for registered_key, strategy in cls._strategies.items():
            if registered_key.matches(lookup_key):
                matching_strategies.append((registered_key, strategy))

        # Sort matching strategies by specificity
        if matching_strategies:

            def specificity_score(key: StrategyKey) -> int:
                return sum(
                    1 for part in [key.data_type, key.data_source] if part == "*"
                )

            matching_strategies.sort(key=lambda x: specificity_score(x[0]))
            found = matching_strategies[0][1]
            logger.info(f"Found matching strategy: {found}")
            return found

        logger.warning(
            f"No matching strategy found for data_type: {data_type}, data_source: {data_source}"
        )
        return None

    @classmethod
    def register(cls, data_type: str, data_source: str):
        """
        Decorator for registering data load strategies
        """

        def decorator(strategy_class: Type[DataLoad]):
            key = StrategyKey(data_type=data_type, data_source=data_source)
            cls._strategies[key] = strategy_class
            return strategy_class

        return decorator


class FileDataLoadStrategy(DataLoad):
    """
    File-based data loading strategy for JSON, JSONL, and Parquet files
    """

    def validate_config(self, config: Dict[str, Any]) -> None:
        if "path" not in config:
            raise ValueError("File data strategy requires 'path' in config")
        if not isinstance(config["path"], str):
            raise ValueError("'path' must be a string")

        path = Path(config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Could not find file '{path}'")

        ext = path.suffix.lower()
        if ext not in [".json", ".jsonl", ".parquet"]:
            raise ValueError(
                f"Unsupported file format: {ext}. Supported formats: .json, .jsonl, .parquet"
            )

    def _load_data_impl(self, **kwargs) -> List[DataSample]:
        path = Path(self.config["path"])
        ext = path.suffix.lower()

        try:
            if ext == ".json":
                return self._load_json(path)
            elif ext == ".jsonl":
                return self._load_jsonl(path)
            elif ext == ".parquet":
                return self._load_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {path}: {str(e)}")

    def _load_json(self, path: Path) -> List[DataSample]:
        """Load data from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [self._convert_to_data_sample(item) for item in data]
        elif isinstance(data, dict):
            return [self._convert_to_data_sample(data)]
        else:
            raise ValueError("Invalid JSON format: expected list or dict")

    def _load_jsonl(self, path: Path) -> List[DataSample]:
        """Load data from JSONL file"""
        data_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    data_list.append(self._convert_to_data_sample(data))
        return data_list

    def _load_parquet(self, path: Path) -> List[DataSample]:
        """Load data from Parquet file"""
        try:
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError("Please install pandas package: pip install pandas")

        data_list = []
        for _, row in df.iterrows():
            try:
                # Convert row to dict and handle any non-serializable types
                data_dict = {}
                for k, v in row.items():
                    if hasattr(v, "item"):
                        try:
                            data_dict[k] = v.item()
                        except (ValueError, AttributeError):
                            # if array type, convert to list and handle nested structures
                            if hasattr(v, "tolist"):
                                data_dict[k] = v.tolist()
                            else:
                                data_dict[k] = v
                    elif hasattr(v, "tolist"):
                        # Handle numpy arrays
                        data_dict[k] = v.tolist()
                    else:
                        data_dict[k] = v

                # ensure data dict contains necessary fields
                if "prompt" not in data_dict:
                    logger.warning(f"Row missing 'prompt' field, skipping: {data_dict}")
                    continue

                # convert data to DataSample object
                data_sample = self._convert_to_data_sample(data_dict)
                if data_sample is not None:
                    data_list.append(data_sample)
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue

        return data_list

    @abstractmethod
    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert raw data dictionary to DataSample format"""
        pass


def create_load_module(
    name: str,
    load_strategy_type: str = "local",
    data_source: str = "*",
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataLoad:
    """Create data load module factory function"""
    return DataLoad(
        name=name,
        load_strategy_type=load_strategy_type,
        data_source=data_source,
        config=config,
        metadata=metadata,
    )
