class DeprecatedError(Exception):
    """Simple custom class to raise an error for something that is now deprecated."""

    def __init__(self, msg: str):
        self.base = "DeprecationError"
        self.message = msg

    def __str__(self):
        return f"{self.base}: {self.message}"
