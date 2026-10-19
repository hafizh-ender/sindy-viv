"""Compatibility-only callable objects for loading older wake CustomLibrary pickles."""

from typing import Any, Callable, Sequence


def _full_input_lambda(count: int, operation: Callable) -> Callable:
    """Rebuild the old full-arity callables when an existing model is unpickled.

    Parameters:
        count (int): Number of positional arguments exposed by the legacy callable.
        operation (Callable): Operation evaluated by the reconstructed legacy callable.
    """
    match count:
        case 4:
            return lambda a, b, c, d: operation((a, b, c, d))
        case 5:
            return lambda a, b, c, d, e: operation((a, b, c, d, e))
        case 8:
            return lambda a, b, c, d, e, f, g, h: operation((a, b, c, d, e, f, g, h))
        case 10:
            return lambda a, b, c, d, e, f, g, h, i, j: operation((a, b, c, d, e, f, g, h, i, j))
        case _:
            raise ValueError(f"Unsupported legacy wake layout with {count} columns.")


class _WakeFunction:
    """Restore old serialized monomial callables; new fitting never constructs these."""

    def __init__(self, input_names: Sequence[str], powers: Sequence[int], naming: bool = False) -> None:
        """Reconstruct the callable from the original pickle constructor arguments.

        Parameters:
            input_names (Sequence[str]): Ordered state and measured-control names expected by this library.
            powers (Sequence[int]): Exponent vector identifying the legacy monomial.
            naming (bool): Whether the legacy callable generates feature names instead of numeric values. Default: False
        """
        self.input_names, self.powers, self.naming = tuple(input_names), tuple(powers), naming
        self.function = _full_input_lambda(len(input_names), self._apply)

    @property
    def __code__(self) -> Any:
        """Expose arity to legacy CustomLibrary feature dispatch."""
        return self.function.__code__

    def __call__(self, *values: Any) -> Any:
        """Evaluate the legacy callable or feature-name formatter.

        Parameters:
            values (object): Positional feature arrays or symbolic names passed to the legacy callable. Default: no extra arguments
        """
        return self.function(*values)

    def __reduce__(self) -> tuple:
        """Serialize parameters rather than an unpicklable closure."""
        return type(self), (self.input_names, self.powers, self.naming)

    def _apply(self, values: Sequence[Any]) -> Any:
        """Preserve the historical monomial values and feature-name order.

        Parameters:
            values (Sequence[Any]): Feature arrays or symbolic names used to evaluate the stored monomial.
        """
        if self.naming:
            defaults = tuple(f"x{i}" for i in range(len(values)))
            if tuple(values) not in (self.input_names, defaults):
                raise ValueError(f"Wake library expects state/control order {self.input_names}.")
            return " ".join(name if power == 1 else f"{name}^{power}" for name, power in zip(values, self.powers) if power)
        result = None
        for value, power in zip(values, self.powers):
            if power:
                factor = value if power == 1 else value**power
                result = factor if result is None else result * factor
        return result
