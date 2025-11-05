from typing import List

from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field, model_validator

from rm_gallery.core.utils import get_value_by_mapping


class DataSample(BaseModel):
    """Data sample containing shared data and individual samples."""

    data: dict = Field(default_factory=dict, description="Shared data for all samples")
    samples: List[dict] = Field(
        default_factory=list,
        description="List of individual samples to evaluate",
    )


class DataSampleMapping(BaseModel):
    """Mapping for transforming data samples."""

    data_mapping: dict | None = Field(
        default=None, description="mapping between data variables and values"
    )
    sample_mapping: dict | None = Field(
        default=None, description="mapping between sample variables and values"
    )

    def __call__(self, data_sample: DataSample, *args, **kwrgs) -> DataSample:
        # Apply data mapping to the main data dictionary
        if self.data_mapping:
            data = get_value_by_mapping(data_sample.data, self.data_mapping)
        else:
            data = data_sample.data.copy()

        # Apply sample mapping to each sample in samples list
        if self.sample_mapping:
            samples = [
                get_value_by_mapping(sample, self.sample_mapping)
                for sample in data_sample.samples
            ]
        else:
            samples = [sample.copy() for sample in data_sample.samples]

        return DataSample(data=data, samples=samples)


class EvaluationDataset(BaseModel):
    """Dataset for evaluation with schema validation."""

    data_sample_schema: dict = Field(
        default_factory=dict, description="Schema for validating data samples"
    )
    data_samples: List[dict | DataSample] = Field(
        default_factory=list, description="Data samples to evaluate"
    )

    @model_validator(mode="before")
    def validate_schema(cls, values):
        """Validate schema and data samples.

        Args:
            values: Dictionary of values to validate

        Returns:
            Validated values

        Raises:
            ValueError: If schema validation fails
        """
        data_sample_schema = values.get("data_sample_schema", {})
        data_samples = values.get("data_samples", [])

        # Validate that data_sample_schema is provided
        if not data_sample_schema:
            raise ValueError("data_sample_schema is required")

        # Validate that all data samples conform to data_sample_schema
        for i, sample in enumerate(data_samples):
            try:
                # If it's already a DataSample, convert to dict for validation
                if isinstance(sample, DataSample):
                    # For DataSample objects, we validate the 'data' part against the schema
                    sample_dict = sample.data
                else:
                    # For dict objects, validate directly
                    sample_dict = sample

                validate(instance=sample_dict, schema=data_sample_schema)
            except ValidationError as e:
                raise ValueError(
                    f"Data sample at index {i} does not conform to data_sample_schema: {str(e)}"
                )
            except Exception as e:
                raise ValueError(f"Error validating data sample at index {i}: {str(e)}")

        return values
