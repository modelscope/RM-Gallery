from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type


class BaseAnnotationTemplate(ABC):
    """Base class for annotation templates"""

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def label_config(self) -> str:
        """Return the Label Studio XML configuration"""
        pass

    @abstractmethod
    def process_annotations(self, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process annotation data and return structured results

        Args:
            annotation_data: Generic annotation data with ratings, choices, text_areas

        Returns:
            Processed data structured for specific evaluation needs
        """
        pass

    def validate_annotation_data(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Validate annotation data structure (optional to override)

        Args:
            annotation_data: Annotation data to validate

        Returns:
            True if valid, False otherwise
        """
        return True


class AnnotationTemplateRegistry:
    """Registry for managing annotation templates"""

    _templates: Dict[str, BaseAnnotationTemplate] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Type[BaseAnnotationTemplate]], Type[BaseAnnotationTemplate]]:
        """Register a template as a decorator"""

        def decorator(
            template_class: Type[BaseAnnotationTemplate],
        ) -> Type[BaseAnnotationTemplate]:
            # Create an instance of the template with the given name
            template_instance = template_class(name)
            cls._templates[name] = template_instance
            return template_class

        return decorator

    @classmethod
    def get_template(cls, name: str) -> Optional[BaseAnnotationTemplate]:
        """Get a template by name"""
        return cls._templates.get(name)

    @classmethod
    def get_label_config(cls, template_name: str) -> Optional[str]:
        """Get label config from template"""
        template = cls.get_template(template_name)
        return template.label_config if template else None

    @classmethod
    def list_templates(cls) -> list[str]:
        """List all registered template names"""
        return list(cls._templates.keys())
