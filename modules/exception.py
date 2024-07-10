"""All app exceptions"""


class NoDocumentsException(Exception):
    """Exception raised when there are no documents in the documents list."""

    def __init__(self, message="There is no doc in documents list"):
        self.message = message
        super().__init__(self.message)
