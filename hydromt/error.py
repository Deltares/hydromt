"""Implementation of deprecation errors."""

# all your cache miss are belong to us!


class DeprecatedError(Exception):

    """Simple custom class to raise an error for something that is now deprecated."""

    def __init__(self, msg: str):
        """Initialise the object."""
        self.base = "DeprecationError"
        self.message = msg

    def __str__(self):
        return f"{self.base}: {self.message}"
