"""
Data Export Module - export data to various formats with train/test split support
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import BaseDataSet, DataSample


class DataExport(BaseDataModule):
    """Data export module - export data to various formats"""

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            module_type=DataModuleType.EXPORT,
            name=name,
            config=config,
            metadata=metadata,
            **kwargs,
        )

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs
    ) -> BaseDataSet:
        """Run data export pipeline"""
        try:
            if input_data is None:
                logger.warning("No input data provided for export")
                return BaseDataSet(name="empty_export", datas=[])

            # Convert to BaseDataSet if needed
            if isinstance(input_data, list):
                dataset = BaseDataSet(name=self.name, datas=input_data)
            else:
                dataset = input_data

            # Get export configuration
            export_config = self.config or {}
            output_dir = Path(export_config.get("output_dir", "./exports"))
            formats = export_config.get("formats", ["json"])
            split_ratio = export_config.get(
                "split_ratio", None
            )  # e.g., {"train": 0.8, "test": 0.2}
            filename_prefix = self.name

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Split dataset if requested
            if split_ratio:
                train_data, test_data = self._split_dataset(dataset.datas, split_ratio)
                datasets_to_export = {
                    "train": BaseDataSet(
                        name=f"{dataset.name}_train",
                        datas=train_data,
                        metadata=dataset.metadata,
                    ),
                    "test": BaseDataSet(
                        name=f"{dataset.name}_test",
                        datas=test_data,
                        metadata=dataset.metadata,
                    ),
                }
            else:
                datasets_to_export = {"full": dataset}

            # Export in requested formats
            for split_name, split_dataset in datasets_to_export.items():
                for format_type in formats:
                    self._export_format(
                        split_dataset,
                        output_dir,
                        format_type,
                        filename_prefix,
                        split_name,
                    )

            logger.info(
                f"Successfully exported {len(dataset.datas)} samples to {output_dir}"
            )
            return dataset

        except Exception as e:
            logger.error(f"Error during data export: {str(e)}")
            raise

    def _split_dataset(
        self, data_samples: List[DataSample], split_ratio: Dict[str, float]
    ) -> Tuple[List[DataSample], List[DataSample]]:
        """Split dataset into train/test sets"""
        if not split_ratio or "train" not in split_ratio:
            raise ValueError("Split ratio must contain 'train' key")

        train_ratio = split_ratio["train"]
        if not 0 < train_ratio < 1:
            raise ValueError("Train ratio must be between 0 and 1")

        # Shuffle data for random split
        shuffled_data = data_samples.copy()
        random.seed(42)  # For reproducible results
        random.shuffle(shuffled_data)

        # Calculate split point
        train_size = int(len(shuffled_data) * train_ratio)

        train_data = shuffled_data[:train_size]
        test_data = shuffled_data[train_size:]

        logger.info(
            f"Split dataset: {len(train_data)} training samples, {len(test_data)} test samples"
        )
        return train_data, test_data

    def _export_format(
        self,
        dataset: BaseDataSet,
        output_dir: Path,
        format_type: str,
        filename_prefix: str,
        split_name: str,
    ):
        """Export dataset in specified format"""
        if split_name == "full":
            filename = f"{filename_prefix}.{format_type}"
        else:
            filename = f"{filename_prefix}_{split_name}.{format_type}"

        filepath = output_dir / filename

        if format_type.lower() == "json":
            self._export_json(dataset, filepath)
        elif format_type.lower() == "jsonl":
            self._export_jsonl(dataset, filepath)
        elif format_type.lower() == "parquet":
            self._export_parquet(dataset, filepath)
        else:
            logger.warning(f"Unsupported format: {format_type}")

    def _export_json(self, dataset: BaseDataSet, filepath: Path):
        """Export dataset to JSON format"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    dataset.to_dict(), f, ensure_ascii=False, indent=2, default=str
                )
            logger.info(f"Exported to JSON: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export JSON to {filepath}: {str(e)}")
            raise

    def _export_jsonl(self, dataset: BaseDataSet, filepath: Path):
        """Export dataset to JSONL format (one JSON object per line)"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for sample in dataset.datas:
                    json.dump(sample.dict(), f, ensure_ascii=False, default=str)
                    f.write("\n")
            logger.info(f"Exported to JSONL: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export JSONL to {filepath}: {str(e)}")
            raise

    def _export_parquet(self, dataset: BaseDataSet, filepath: Path):
        """Export dataset to Parquet format"""
        try:
            # Convert data samples to flat dictionary format
            records = []
            for sample in dataset.datas:
                record = {
                    "unique_id": sample.unique_id,
                    "input": json.dumps(
                        [msg.dict() for msg in sample.input], default=str
                    ),
                    "output": json.dumps(
                        [out.dict() for out in sample.output], default=str
                    ),
                    "task_category": sample.task_category,
                    "source": sample.source,
                    "created_at": sample.created_at,
                    "metadata": json.dumps(sample.metadata, default=str)
                    if sample.metadata
                    else None,
                }
                records.append(record)

            # Create DataFrame and save as Parquet
            df = pd.DataFrame(records)
            df.to_parquet(filepath, index=False, engine="pyarrow")
            logger.info(f"Exported to Parquet: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export Parquet to {filepath}: {str(e)}")
            raise


def create_export_module(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataExport:
    """Factory function to create a data export module"""
    return DataExport(name=name, config=config, metadata=metadata)
