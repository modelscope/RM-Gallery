from typing import Callable, Dict, Optional, Union

from loguru import logger

from rm_gallery.core.grader import Grader


class GraderRegistry:
    """Registry for managing grader functions in v2 with hierarchical structure support."""

    _graders: Dict[str, Union[Grader | Callable, Dict]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        grader: Grader | Callable | None = None,
        namespace: str | None = None,
    ) -> Union[None, Callable]:
        """Register a grader function with a given name, optionally under a namespace.
        Can be used as a decorator or a direct function call.

        Args:
            name: The name to register the grader under
            grader: The grader function to register (if used as direct function call)
            namespace: Optional namespace to group graders (e.g., "math", "code")

        Returns:
            None if used as direct function call, decorator function if used as decorator
        """
        # If grader is provided, register it directly
        if grader is not None:
            cls._register_grader(name, grader, namespace)
            return None

        # If grader is not provided, return a decorator
        def decorator(grader: Grader | Callable) -> Grader | Callable:
            cls._register_grader(name, grader, namespace)
            return grader

        return decorator

    @classmethod
    def _register_grader(
        cls, name: str, grader: Grader | Callable, namespace: Optional[str] = None
    ) -> None:
        """Internal method to register a grader function.

        Args:
            name: The name to register the grader under
            grader: The grader function to register
            namespace: Optional namespace to group graders (e.g., "math", "code")
        """
        if not isinstance(grader, Grader):
            raise TypeError(
                f"grader must be an instance of Evaluationgrader, got {type(grader)}"
            )

        full_name = f"{namespace}.{name}" if namespace else name

        # Handle namespace creation
        if namespace:
            if namespace not in cls._graders:
                cls._graders[namespace] = {}
            elif not isinstance(cls._graders[namespace], dict):
                raise ValueError(
                    f"Namespace '{namespace}' conflicts with an existing grader name"
                )

            namespace_dict = cls._graders[namespace]
            if not isinstance(namespace_dict, dict):
                raise ValueError(f"Namespace '{namespace}' is not a valid namespace")

            if name in namespace_dict:
                logger.warning(
                    f"grader '{full_name}' is already registered. Overwriting."
                )

            grader.name = full_name
            namespace_dict[name] = grader
        else:
            if name in cls._graders and isinstance(cls._graders[name], dict):
                raise ValueError(
                    f"grader name '{name}' conflicts with an existing namespace"
                )

            if name in cls._graders:
                logger.warning(f"grader '{name}' is already registered. Overwriting.")

            grader.name = name
            cls._graders[name] = grader

        logger.info(f"Registered grader '{full_name}'")

    @classmethod
    def get(cls, name: str) -> Optional[Grader]:
        """Get a registered grader function by name (supports dot notation for namespaces).

        Args:
            name: The name of the grader function to get (e.g., "math.accuracy" or "general")

        Returns:
            The registered grader function, or None if not found
        """
        if "." in name:
            # Handle namespaced graders
            namespace, sub_name = name.split(".", 1)
            if namespace in cls._graders and isinstance(cls._graders[namespace], dict):
                namespace_dict = cls._graders[namespace]
                if isinstance(namespace_dict, dict) and sub_name in namespace_dict:
                    return namespace_dict[sub_name]
            return None
        else:
            # Handle direct graders
            grader = cls._graders.get(name)
            if isinstance(grader, Grader):
                return grader
            return None

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a registered grader function by name (supports dot notation for namespaces).

        Args:
            name: The name of the grader function to remove (e.g., "math.accuracy" or "general")

        Returns:
            True if the grader was removed, False if it wasn't registered
        """
        if "." in name:
            # Handle namespaced graders
            namespace, sub_name = name.split(".", 1)
            if namespace in cls._graders and isinstance(cls._graders[namespace], dict):
                namespace_dict = cls._graders[namespace]
                if isinstance(namespace_dict, dict) and sub_name in namespace_dict:
                    del namespace_dict[sub_name]
                    # Clean up empty namespace
                    if not namespace_dict:
                        del cls._graders[namespace]
                    logger.info(f"Removed grader '{name}'")
                    return True
            return False
        else:
            # Handle direct graders
            if name in cls._graders and isinstance(cls._graders[name], Grader):
                del cls._graders[name]
                logger.info(f"Removed grader '{name}'")
                return True
            return False

    @classmethod
    def list_graders(cls, namespace: Optional[str] = None) -> Dict[str, str]:
        """List registered graders, optionally filtered by namespace.

        Args:
            namespace: Optional namespace to filter by

        Returns:
            A dictionary mapping grader names to their types
        """
        result = {}

        if namespace:
            # List graders in a specific namespace
            if namespace in cls._graders and isinstance(cls._graders[namespace], dict):
                namespace_dict = cls._graders[namespace]
                if isinstance(namespace_dict, dict):
                    for name, grader in namespace_dict.items():
                        if isinstance(grader, Grader):
                            result[f"{namespace}.{name}"] = type(grader).__name__
            return result
        else:
            # List all graders
            for key, value in cls._graders.items():
                if isinstance(value, Grader):
                    # Direct grader
                    result[key] = type(value).__name__
                elif isinstance(value, dict):
                    # Namespace
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, Grader):
                            result[f"{key}.{sub_key}"] = type(sub_value).__name__
            return result

    @classmethod
    def list_namespaces(cls) -> list:
        """List all available namespaces.

        Returns:
            A list of namespace names
        """
        return [key for key, value in cls._graders.items() if isinstance(value, dict)]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered graders."""
        cls._graders.clear()
        logger.info("Cleared all registered graders")


GR = GraderRegistry()
