from typing import Any, Dict, List, Optional

from loguru import logger

from rm_gallery.core.data.process.ops.base import BaseOperator, OperatorFactory
from rm_gallery.core.data.schema import DataSample


@OperatorFactory.register("conversation_turn_filter")
class ConversationTurnFilter(BaseOperator):
    """
    Filter conversations based on the number of turns in the input.
    A turn is defined as a single message in the conversation.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the conversation turn filter.

        Args:
            name: Name of the operator
            min_turns: Minimum number of turns required (inclusive)
            max_turns: Maximum number of turns allowed (inclusive)
            config: Additional configuration parameters
        """
        super().__init__(name=name, config=config)

    def process_dataset(self, items: List[DataSample]) -> List[DataSample]:
        """
        Filter conversations based on the number of turns.

        Args:
            items: List of DataSample items to process

        Returns:
            List of DataSample items that meet the turn count criteria
        """
        try:
            filtered_items = []
            for item in items:
                # Count the number of user turns in the input
                num_turns = (
                    sum(1 for input_item in item.input if input_item.role == "user")
                    if item.input
                    else 0
                )

                # Check if the number of turns is within the specified range
                if (
                    self.config.get("min_turns", 1)
                    <= num_turns
                    <= self.config.get("max_turns", 100)
                ):
                    filtered_items.append(item)
                else:
                    pass
                    # logger.debug(f"Filtered out conversation with {num_turns} user turns "
                    #            f"(min: {self.min_turns}, max: {self.max_turns})")

            return filtered_items
        except Exception as e:
            logger.error(f"Error in conversation turn filtering: {str(e)}")
            return items


def create_conversation_turn_filter(operator_config: Dict[str, Any]) -> BaseOperator:
    """
    Create a conversation turn filter operator from configuration.

    Args:
        operator_config: Configuration dictionary containing:
            - name: Name of the operator
            - config: Configuration dictionary containing:
                - min_turns: Minimum number of turns (default: 1)
                - max_turns: Maximum number of turns (default: 100)

    Returns:
        A configured ConversationTurnFilter operator
    """
    name = operator_config.get("name", "conversation_turn_filter")
    config = operator_config.get("config", {})

    return ConversationTurnFilter(name=name, config=config)
