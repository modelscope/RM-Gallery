from typing import Any


def get_value_by_path(data, path: str, default: Any | None = None):
    keys = path.split(".")
    current_value = data
    try:
        for key in keys:
            current_value = getattr(current_value, key)
    except (KeyError, TypeError):
        if default is not None:
            return default
        else:
            raise
    return current_value