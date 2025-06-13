from time import time
from typing import Any

from loguru import logger


class Retry(object):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def __enter__(self) -> "Retry":
        return self

    def __call__(self, func, *args: Any, **kwrgs: Any) -> Any:
        for i in range(self.max_retries):
            try:
                return func(*args, **kwrgs)
            except Exception as e:
                if i < self.max_retries - 1:
                    logger.warning(
                        f"Attempt {i + 1} failed. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed.")
                    raise e

    def __exit__(self, **kwargs):
        logger.info("Retry manager exited: {kwargs}")
