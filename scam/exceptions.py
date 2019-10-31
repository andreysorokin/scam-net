class InvalidState(ValueError):
    def __init__(self, message, *args):
        self.message = message # without this you may get DeprecationWarning
        super(InvalidState, self).__init__(message, *args)