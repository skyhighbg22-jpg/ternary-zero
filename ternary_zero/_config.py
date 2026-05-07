_grad_enabled = True


def is_grad_enabled():
    return _grad_enabled


def _set_grad_enabled(enabled: bool):
    global _grad_enabled
    _grad_enabled = enabled


class enable_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = True

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev


class no_grad:
    def __init__(self):
        pass

    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev
