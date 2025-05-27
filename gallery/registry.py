from typing import Dict, Type, Optional, List

from src.rm.module import BaseRewardModule


class GalleryRegistry:
    """评估器注册表，用于管理不同评分体系的评估器"""
    _registry: Dict[str, Dict[str, Type[BaseRewardModule]]] = {}
    
    @classmethod
    def register(cls, system: str, eval_type: str, evaluator_class: Type[BaseRewardModule]):
        """
        register a reward module class
        """
        if system not in cls._registry:
            cls._registry[system] = {}
        cls._registry[system][eval_type] = evaluator_class
    
    @classmethod
    def get_module(cls, system: str, eval_type: str) -> Optional[Type[BaseRewardModule]]:
        """get a reward module class"""
        return cls._registry.get(system, {}).get(eval_type)
