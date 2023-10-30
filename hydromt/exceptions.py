"""HydroMT exceptions."""


class NoDataException(Exception):
    """Exception raised for errors in the input.

    Attributes
    ----------
        message -- explanation of the error
    """

    def __init__(self, message="No data available"):
        self.message = message
        super().__init__(self.message)
