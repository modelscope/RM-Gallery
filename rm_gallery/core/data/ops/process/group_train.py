import random
from typing import Any, Dict, List, Optional

from loguru import logger

from rm_gallery.core.data.process import BaseOperator, OperatorFactory
from rm_gallery.core.data.schema import DataSample


@OperatorFactory.register("group_train")
class GroupTrain(BaseOperator):
    """
    Filter and group data items into different sets (e.g., train/test).
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the group filter.

        Args:
            name: Name of the operator
            train_ratio: Ratio of data to be used for training (default: 0.8)
            test_ratio: Ratio of data to be used for testing (default: 0.2)
            seed: Random seed for reproducibility (default: 42)
            config: Additional configuration parameters
        """
        super().__init__(
            name=name,
            config=config,
        )

        # Validate ratios
        if (
            abs(
                self.config.get("train_ratio", 0.8)
                + self.config.get("test_ratio", 0.2)
                - 1.0
            )
            > 1e-6
        ):
            logger.warning(
                f"Train ratio ({self.config.get('train_ratio', 0.8)}) + test ratio ({self.config.get('test_ratio', 0.2)}) != 1.0"
            )

        # Set random seed
        random.seed(self.config.get("seed", 42))

    def process_dataset(self, items: List[DataSample]) -> List[DataSample]:
        """
        Split items into train and test sets.

        Args:
            items: List of data items to process

        Returns:
            List of data items with group information added
        """
        try:
            # Shuffle items
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)

            # Calculate split indices
            n_items = len(shuffled_items)
            n_train = int(n_items * self.config.get("train_ratio", 0.8))

            # Split items
            train_items = shuffled_items[:n_train]
            test_items = shuffled_items[n_train:]

            # Add group information to items
            for item in train_items:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["group"] = "train"

            for item in test_items:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["group"] = "test"

            logger.info(
                f"Split {n_items} items into {len(train_items)} train and {len(test_items)} test items"
            )

            # Return all items
            return shuffled_items

        except Exception as e:
            logger.error(f"Error in group filtering: {str(e)}")
            return items


def create_group_train(operator_config: Dict[str, Any]) -> BaseOperator:
    """
    Create a group filter operator from configuration.

    Args:
        operator_config: Configuration dictionary containing:
            - name: Name of the operator
            - config: Configuration dictionary containing:
                - train_ratio: Ratio of data for training (optional)
                - test_ratio: Ratio of data for testing (optional)
                - seed: Random seed (optional)

    Returns:
        GroupFilter instance
    """
    name = operator_config.get("name", "group_filter")
    config = operator_config.get("config", {})

    return GroupTrain(
        name=name,
        config=config,
    )
