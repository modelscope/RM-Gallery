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
    dimension: str = "*"

    def matches(self, other: "StrategyKey") -> bool:
        """
        Check if this key matches another key with wildcard support
        """
        return (
            fnmatch.fnmatch(other.data_type, self.data_type)
            and fnmatch.fnmatch(other.data_source, self.data_source)
            and fnmatch.fnmatch(other.dimension, self.dimension)
        )


class DataLoadStrategy(BaseDataModule):
    """
    Abstract class for data load strategy
    """

    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any], **kwargs):
        # Provide required fields for BaseDataModule
        super().__init__(
            module_type=DataModuleType.LOAD,
            name=kwargs.get("name", f"{self.__class__.__name__}"),
            config=config,
            **kwargs,
        )
        self.validate_config(config)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary
        Override this method in subclasses to add specific validation rules
        """
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> List[DataSample]:
        """
        Load data from the source and return a list of DataSample objects
        """
        pass

    def run(self, **kwargs) -> List[DataSample]:
        """
        Run method implementation for DataLoadStrategy
        """
        return self.load_data(**kwargs)


class DataLoadStrategyRegistry:
    """
    Registry for data load strategies with wildcard matching
    """

    _strategies: Dict[StrategyKey, Type[DataLoadStrategy]] = {}

    @classmethod
    def get_strategy_class(
        cls, data_type: str, data_source: str, dimension: str
    ) -> Optional[Type[DataLoadStrategy]]:
        """
        Retrieve the most specific matching strategy
        """
        logger.info(
            f"Getting strategy class for data_type: {data_type}, data_source: {data_source}, dimension: {dimension}"
        )

        # Default to wildcard if not provided
        data_type = data_type or "*"
        data_source = data_source or "*"

        # Create the lookup key
        lookup_key = StrategyKey(
            data_type=data_type, data_source=data_source, dimension=dimension
        )

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
            f"No matching strategy found for data_type: {data_type}, data_source: {data_source}, dimension: {dimension}"
        )
        return None

    @classmethod
    def register(cls, data_type: str, data_source: str, dimension: str):
        """
        Decorator for registering data load strategies
        """

        def decorator(strategy_class: Type[DataLoadStrategy]):
            key = StrategyKey(
                data_type=data_type, data_source=data_source, dimension=dimension
            )
            cls._strategies[key] = strategy_class
            return strategy_class

        return decorator


class FileDataLoadStrategy(DataLoadStrategy):
    """
    Base strategy for loading data from files (JSON, JSONL, Parquet)
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

    def load_data(self, **kwargs) -> List[DataSample]:
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


class DataLoad(BaseDataModule):
    """Data Load Module - load data from various data sources"""

    load_strategy_type: str = Field(
        default="local", description="data load strategy type (local or remote)"
    )
    data_source: str = Field(default="*", description="data source")
    dimension: str = Field(default="*", description="data dimension")
    load_config: Dict[str, Any] = Field(default_factory=dict, description="load config")

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        load_strategy_type: str = "local",
        data_source: str = "*",
        dimension: str = "*",
        load_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        initialize data load module

        Args:
            name: module name
            config: module config
            load_strategy_type: load strategy type
            data_source: data source
            dimension: data dimension
            load_config: load config
        """
        super().__init__(
            module_type=DataModuleType.LOAD,
            name=name,
            config=config,
            load_strategy_type=load_strategy_type,
            data_source=data_source,
            dimension=dimension,
            load_config=load_config or {},
            **kwargs,
        )

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs
    ) -> BaseDataSet:
        """load data"""
        try:
            # Get appropriate data loading strategy
            strategy_class = DataLoadStrategyRegistry.get_strategy_class(
                data_type=self.load_strategy_type,
                data_source=self.data_source,
                dimension=self.dimension,
            )

            if not strategy_class:
                error_msg = f"No suitable data load strategy found for type: {self.load_strategy_type}, source: {self.data_source}, dimension: {self.dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Initialize and use strategy to load data
            # Add dimension to the config passed to strategy
            strategy_config = self.load_config.copy()
            strategy_config["dimension"] = self.dimension
            strategy = strategy_class(strategy_config)
            try:
                loaded_items = strategy.load_data()
            except Exception as load_error:
                logger.error(f"Error in strategy.load_data(): {str(load_error)}")
                raise load_error

            # Convert loaded items to DataSample objects if needed
            data_samples = []

            for item in loaded_items:
                data_samples.append(item)

            # Apply limit (if specified)
            if (
                "limit" in self.load_config
                and self.load_config["limit"] is not None
                and self.load_config["limit"] > 0
            ):
                limit = min(int(self.load_config["limit"]), len(data_samples))
                data_samples = random.sample(data_samples, limit)
                logger.info(
                    f"Applied limit of {limit}, final count: {len(data_samples)}"
                )

            # Create output dataset
            output_dataset = BaseDataSet(
                name=f"loaded_dataset_{self.data_source}",
                metadata={
                    "source": self.data_source,
                    "strategy_type": self.load_strategy_type,
                    "dimension": self.dimension,
                    "load_config": self.load_config,
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


def create_load_module(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    load_strategy_type: str = "local",
    data_source: str = "*",
    dimension: str = "*",
    load_config: Optional[Dict[str, Any]] = None,
) -> DataLoad:
    """create data load module factory function"""
    return DataLoad(
        name=name,
        config=config,
        load_strategy_type=load_strategy_type,
        data_source=data_source,
        dimension=dimension,
        load_config=load_config,
    )
