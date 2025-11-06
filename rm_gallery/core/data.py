from typing import List

from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field

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


def validate_data_samples(
    data_samples: List[dict | DataSample], schema: dict | None = None
):
    # Validate that data_sample_schema is provided
    if not schema:
        raise ValueError("the schema of data sample is required")

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

            validate(instance=sample_dict, schema=schema)
        except ValidationError as e:
            raise ValueError(
                f"Data sample at index {i} does not conform to schema: {str(e)}"
            )
        except Exception as e:
            raise ValueError(f"Error validating data sample at index {i}: {str(e)}")

    return [
        DataSample(**data_sample) if isinstance(data_sample, dict) else data_sample
        for data_sample in data_samples
    ]
