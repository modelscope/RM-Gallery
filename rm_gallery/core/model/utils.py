import json

from json_repair import repair_json


def _json_loads_with_repair(
    json_str: str,
) -> dict | list | str | float | int | bool | None:
    """The given json_str maybe incomplete, e.g. '{"key', so we need to
    repair and load it into a Python object.
    """
    repaired = json_str
    try:
        repaired = repair_json(json_str)
    except Exception:
        pass

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to decode JSON string `{json_str}` after repairing it "
            f"into `{repaired}`. Error: {e}",
        ) from e
