from typing import Any, Dict, List, Optional

from rm_gallery.core.data.process.ops.base import BaseOperator, OperatorFactory
from rm_gallery.core.data.schema import DataSample


@OperatorFactory.register("text_length_filter")
class TextLengthFilter(BaseOperator):
    """
    Filter texts based on their length.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the text length filter.

        Args:
            name: Name of the operator
            min_length: Minimum text length required (inclusive)
            max_length: Maximum text length allowed (inclusive)
            config: Additional configuration parameters
        """
        super().__init__(name=name, config=config)

    def process_dataset(self, items: List[DataSample]) -> List[DataSample]:
        """
        Filter items based on text length.

        Args:
            items: List of data items to process

        Returns:
            Filtered list of items
        """
        filtered_items = []
        for item in items:
            # get all input and output texts
            texts = []

            # process input from history
            if item.input:
                for input_item in item.input:
                    if input_item.content:
                        texts.append(input_item.content)

            # process output from answers
            if item.output:
                for output_item in item.output:
                    if (
                        hasattr(output_item, "answer")
                        and output_item.answer
                        and output_item.answer.content
                    ):
                        texts.append(output_item.answer.content)

            # calculate total length
            total_length = sum(len(text) for text in texts)

            if (
                self.config.get("min_length", 10)
                <= total_length
                <= self.config.get("max_length", 1000)
            ):
                filtered_items.append(item)
            else:
                pass
                # logger.debug(f"Filtered out item with total length {total_length}")
        return filtered_items


def create_text_length_filter(operator_config: Dict[str, Any]) -> BaseOperator:
    """
    Create a text length filter operator from configuration.

    Args:
        operator_config: Configuration dictionary containing:
            - name: Name of the operator
            - config: Configuration dictionary containing:
                - min_length: Minimum text length (optional)
                - max_length: Maximum text length (optional)

    Returns:
        TextLengthFilter instance
    """
    name = operator_config.get("name", "text_length_filter")
    config = operator_config.get("config", {})

    return TextLengthFilter(name=name, config=config)
