__all__ = ["runtime_status", "vectorize_icon_crop"]


def vectorize_icon_crop(*args, **kwargs):
    from .vectorize import vectorize_icon_crop as _vectorize_icon_crop

    return _vectorize_icon_crop(*args, **kwargs)


def runtime_status():
    from .vectorize import runtime_status as _runtime_status

    return _runtime_status()
