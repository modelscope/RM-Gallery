"""
Metric Registry

Metric registry for managing and discovering all available evaluation metrics.
Uses singleton pattern to ensure global uniqueness.
"""

from typing import Dict, List, Optional, Type

from loguru import logger

from rm_gallery.core.metrics.base import BaseMetric


class MetricRegistry:
    """
    Metric Registry (Singleton Pattern)

    Manages all registered evaluation metrics, provides registration, query and listing functionality.

    Example:
        >>> from rm_gallery.core.metrics import metric_registry, register_metric
        >>>
        >>> @register_metric("my_metric")
        ... class MyMetric(BaseMetric):
        ...     name: str = "my_metric"
        ...     def compute(self, input_data):
        ...         return MetricResult(name=self.name, score=1.0)
        >>>
        >>> # Get registered metric
        >>> MetricClass = metric_registry.get("my_metric")
        >>> metric = MetricClass()
    """

    _instance: Optional["MetricRegistry"] = None
    _metrics: Dict[str, Type[BaseMetric]] = {}

    def __new__(cls) -> "MetricRegistry":
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = {}
        return cls._instance

    def register(
        self, name: str, metric_class: Type[BaseMetric], override: bool = False
    ) -> None:
        """
        Register metric class

        Args:
            name: Metric name (unique identifier)
            metric_class: Metric class (must inherit from BaseMetric)
            override: Whether to allow overriding existing metrics

        Raises:
            ValueError: When metric already exists and override=False
            TypeError: When metric_class is not a subclass of BaseMetric

        Example:
            >>> registry = MetricRegistry()
            >>> registry.register("bleu", BLEUMetric)
        """
        # Type check
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"{metric_class.__name__} must be a subclass of BaseMetric")

        # Check if already exists
        if name in self._metrics and not override:
            logger.warning(
                f"Metric '{name}' is already registered. "
                f"Use override=True to replace it, or choose a different name."
            )
            return

        self._metrics[name] = metric_class
        logger.info(f"Registered metric: {name} ({metric_class.__name__})")

    def unregister(self, name: str) -> bool:
        """
        Unregister metric

        Args:
            name: Metric name

        Returns:
            bool: Whether successfully unregistered

        Example:
            >>> registry.unregister("bleu")
            True
        """
        if name in self._metrics:
            del self._metrics[name]
            logger.info(f"Unregistered metric: {name}")
            return True
        else:
            logger.warning(f"Metric '{name}' not found in registry")
            return False

    def get(self, name: str) -> Optional[Type[BaseMetric]]:
        """
        Get metric class

        Args:
            name: Metric name

        Returns:
            Optional[Type[BaseMetric]]: Metric class, returns None if not found

        Example:
            >>> MetricClass = registry.get("bleu")
            >>> if MetricClass:
            ...     metric = MetricClass()
        """
        metric_class = self._metrics.get(name)
        if metric_class is None:
            logger.warning(f"Metric '{name}' not found in registry")
        return metric_class

    def get_instance(self, name: str, **kwargs) -> Optional[BaseMetric]:
        """
        Get metric instance

        Args:
            name: Metric name
            **kwargs: Parameters passed to metric constructor

        Returns:
            Optional[BaseMetric]: Metric instance

        Example:
            >>> metric = registry.get_instance("bleu", max_ngram_order=4)
        """
        metric_class = self.get(name)
        if metric_class is None:
            return None

        try:
            return metric_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating instance of metric '{name}': {e}")
            return None

    def list_metrics(self) -> List[str]:
        """
        List all registered metric names

        Returns:
            List[str]: List of metric names

        Example:
            >>> registry.list_metrics()
            ['bleu', 'rouge', 'meteor', 'fuzzy_match', 'cosine']
        """
        return sorted(list(self._metrics.keys()))

    def list_metrics_by_category(self) -> Dict[str, List[str]]:
        """
        List metrics by category

        Returns:
            Dict[str, List[str]]: Mapping from category to list of metric names

        Example:
            >>> registry.list_metrics_by_category()
            {
                'nlp_metrics': ['bleu', 'rouge', 'meteor'],
                'text_similarity': ['fuzzy_match', 'cosine'],
                'string_check': ['exact_match', 'substring']
            }
        """
        categories: Dict[str, List[str]] = {}

        for name, metric_class in self._metrics.items():
            # Infer category from module path
            module_path = metric_class.__module__
            if "nlp_metrics" in module_path:
                category = "nlp_metrics"
            elif "text_similarity" in module_path:
                category = "text_similarity"
            elif "string_check" in module_path:
                category = "string_check"
            else:
                category = "other"

            if category not in categories:
                categories[category] = []
            categories[category].append(name)

        return categories

    def is_registered(self, name: str) -> bool:
        """
        Check if metric is registered

        Args:
            name: Metric name

        Returns:
            bool: Whether registered
        """
        return name in self._metrics

    def clear(self) -> None:
        """
        Clear registry (mainly for testing)

        Warning:
            This operation removes all registered metrics, use with caution.
        """
        count = len(self._metrics)
        self._metrics.clear()
        logger.warning(f"Cleared metric registry ({count} metrics removed)")

    def __len__(self) -> int:
        """Return number of registered metrics"""
        return len(self._metrics)

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator"""
        return name in self._metrics

    def __repr__(self) -> str:
        """String representation"""
        return f"MetricRegistry({len(self._metrics)} metrics registered)"


# Global registry instance
metric_registry = MetricRegistry()


def register_metric(name: str, override: bool = False):
    """
    Decorator: Register metric class

    This is the most concise way to register metrics.

    Args:
        name: Metric name
        override: Whether to allow overriding existing metrics

    Example:
        >>> @register_metric("bleu")
        ... class BLEUMetric(BaseMetric):
        ...     name: str = "bleu"
        ...
        ...     def compute(self, input_data: ComparisonInput) -> MetricResult:
        ...         # Implement BLEU computation
        ...         return MetricResult(name=self.name, score=0.8)
    """

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        metric_registry.register(name, cls, override=override)
        return cls

    return decorator


def get_metric(name: str, **kwargs) -> Optional[BaseMetric]:
    """
    Helper function: Get metric instance

    Args:
        name: Metric name
        **kwargs: Parameters passed to metric constructor

    Returns:
        Optional[BaseMetric]: Metric instance

    Example:
        >>> bleu = get_metric("bleu", max_ngram_order=4)
        >>> if bleu:
        ...     result = bleu.compute(input_data)
    """
    return metric_registry.get_instance(name, **kwargs)


def list_available_metrics() -> List[str]:
    """
    Helper function: List all available metrics

    Returns:
        List[str]: List of metric names

    Example:
        >>> metrics = list_available_metrics()
        >>> print(f"Available metrics: {', '.join(metrics)}")
    """
    return metric_registry.list_metrics()
