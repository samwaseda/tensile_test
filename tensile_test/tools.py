import inspect
from functools import wraps


# Sentinel wrapper
class ExplicitDefault:
    def __init__(self, default, msg=None):
        self.default = default
        self.msg = msg


def use_default(default, msg=None):
    return ExplicitDefault(default, msg)


# Decorator
def with_explicit_defaults(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Track updated arguments
        new_args = list(args)
        new_kwargs = dict(kwargs)

        for name, param in sig.parameters.items():
            default_val = param.default

            if not isinstance(default_val, ExplicitDefault):
                continue  # Only handle sentinel-wrapped defaults

            # Determine if argument was passed
            was_explicit = (
                name in bound.arguments
                and name in kwargs
                or (
                    param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]
                    and list(sig.parameters).index(name) < len(args)
                )
            )

            if not was_explicit:
                # Not passed: Replace sentinel with real default
                if default_val.msg:
                    warnings.warn(default_val.msg)
                else:
                    warnings.warn(
                        f"'{name}' not provided, using default: {default_val.default}"
                    )
                if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]:
                    idx = list(sig.parameters).index(name)
                    if idx < len(new_args):
                        new_args[idx] = default_val.default
                    else:
                        new_kwargs[name] = default_val.default
                else:
                    new_kwargs[name] = default_val.default

        return func(*new_args, **new_kwargs)

    return wrapper


