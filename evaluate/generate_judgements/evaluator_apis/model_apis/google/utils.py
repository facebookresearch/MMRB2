"""Google API utility functions."""

import threading
from time import sleep
from typing import Any


def _call_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Call a function with a timeout."""
    result_holder = {"has_result": False, "result": None}
    error_holder = {"error": None}

    def runner():
        try:
            result_holder["result"] = func(*args, **kwargs)
            result_holder["has_result"] = True
        except Exception as e:
            error_holder["error"] = e

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    if error_holder["error"] is not None:
        raise error_holder["error"]

    return result_holder["result"]


def generate_content_with_retry(
    client, max_retries: int = 8, timeout_seconds: int = 600, **kwargs
) -> Any:
    """Generate content with retry logic and timeout.

    Args:
        client: The Gemini client instance.
        max_retries: Maximum number of retry attempts.
        timeout_seconds: Timeout for each attempt in seconds.
        **kwargs: Arguments to pass to generate_content.

    Returns:
        The API response.

    Raises:
        ValueError: If all retries are exhausted.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return _call_with_timeout(
                client.models.generate_content, timeout_seconds, **kwargs
            )
        except Exception as e:
            last_exception = e
            print(f"Google API Error in attempt {attempt + 1}: {e}")
            # Exponential backoff
            sleep(2**attempt)
    raise ValueError(f"Failed after {max_retries} attempts: {str(last_exception)}")
