class InvalidArgumentError(ValueError):
    pass


class InvalidOperationError(ValueError):
    pass


class LinkOverridingError(ValueError):
    pass


__all__ = [
    "InvalidArgumentError",
    "InvalidOperationError",
    "LinkOverridingError",
]
