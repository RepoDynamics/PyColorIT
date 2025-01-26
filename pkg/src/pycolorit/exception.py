class PyColorITException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
        return


class PyColorITParseError(PyColorITException):
    def __init__(self, value: str, message: str):
        self.value = value
        super().__init__(f"Failed to parse color value '{value}': {message}")
        return