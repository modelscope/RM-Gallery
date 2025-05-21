from base_operator import OperatorFactory, Operator
from .conversation_turn_filter import ConversationTurnFilter, create_conversation_turn_filter

# Define all operators
OPERATORS = {
    'conversation_turn_filter': create_conversation_turn_filter,
}

def register_all_operators():
    """Register all operators with the factory"""
    for op_name, op_creator in OPERATORS.items():
        OperatorFactory.register(op_name)(op_creator)

# Register operators when the module is imported
register_all_operators() 