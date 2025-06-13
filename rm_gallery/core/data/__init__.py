from rm_gallery.core.data.load.chat_message import ChatMessageDataLoadStrategy
from rm_gallery.core.data.load.huggingface import HuggingfaceDataLoadStrategy
from rm_gallery.core.data.process.ops.filter.conversation_turn_filter import (
    ConversationTurnFilter,
)
from rm_gallery.core.data.process.ops.filter.text_length_filter import TextLengthFilter

OPERATORS = {
    "conversation_turn_filter": ConversationTurnFilter,
    "text_length_filter": TextLengthFilter,
}

LOAD_STRATEGIES = {
    "chatmessage": ChatMessageDataLoadStrategy,
    "huggingface": HuggingfaceDataLoadStrategy,
}
