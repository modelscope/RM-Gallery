from ..process import OperatorFactory
from .load.local_chatmessage import ChatMessageDataLoadStrategy
from .load.local_normal import NormalDataLoadStrategy
from .load.local_prmbench import PRMDataLoadStrategy
from .load.local_rewardbench import RewardBenchDataLoadStrategy
from .load.remote_huggingface import HuggingfaceDataLoadStrategy
from .process.conversation_turn_filter import ConversationTurnFilter
from .process.group_train import GroupTrain
from .process.text_length_filter import TextLengthFilter

OPERATORS = {
    "conversation_turn_filter": ConversationTurnFilter,
    "rm_text_length_filter": TextLengthFilter,
    "group_train": GroupTrain,
}

LOAD_STRATEGIES = {
    "rewardbench": RewardBenchDataLoadStrategy,
    "chatmessage": ChatMessageDataLoadStrategy,
    "prmbench": PRMDataLoadStrategy,
    "huggingface": HuggingfaceDataLoadStrategy,
    "normal": NormalDataLoadStrategy,
}
