"""
Data module initialization - centralized imports for data processing components.
Provides standardized access to data operations, loaders, and strategies.
"""
from rm_gallery.core.data.load.chat_message import ChatMessageConverter
from rm_gallery.core.data.load.huggingface import GenericConverter

# Multimodal support (new)
from rm_gallery.core.data.multimodal_content import ImageContent, MultimodalContent
from rm_gallery.core.data.multimodal_message import (
    MultimodalChatMessage,
    convert_to_multimodal,
)
from rm_gallery.core.data.process.ops.filter.conversation_turn_filter import (
    ConversationTurnFilter,
)
from rm_gallery.core.data.process.ops.filter.text_length_filter import TextLengthFilter

# Core data schemas
from rm_gallery.core.data.schema import (
    BaseDataSet,
    DataOutput,
    DataSample,
    Reward,
    Step,
)

OPERATORS = {
    "conversation_turn_filter": ConversationTurnFilter,
    "text_length_filter": TextLengthFilter,
}

LOAD_STRATEGIES = {
    "chat_message": ChatMessageConverter,
    "huggingface": GenericConverter,
}

__all__ = [
    # Operators and strategies
    "OPERATORS",
    "LOAD_STRATEGIES",
    "ChatMessageConverter",
    "GenericConverter",
    "ConversationTurnFilter",
    "TextLengthFilter",
    # Core data schemas
    "BaseDataSet",
    "DataOutput",
    "DataSample",
    "Reward",
    "Step",
    # Multimodal support
    "ImageContent",
    "MultimodalContent",
    "MultimodalChatMessage",
    "convert_to_multimodal",
]
