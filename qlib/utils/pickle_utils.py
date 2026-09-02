# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Secure pickle utilities to prevent arbitrary code execution through deserialization.

This module provides a secure alternative to pickle.load() and pickle.loads()
that restricts deserialization to a whitelist of safe classes.
"""

import io
import pickle
from typing import Any, BinaryIO, Set, Tuple

# Whitelist of safe classes that are allowed to be unpickled
# These are common data types used in qlib that should be safe to deserialize
SAFE_PICKLE_CLASSES: Set[Tuple[str, str]] = {
    # python builtins
    ("builtins", "slice"),
    ("builtins", "range"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "bytes"),
    ("builtins", "str"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("builtins", "complex"),
    ("builtins", "type"),
    ("builtins", "property"),
    # common utility classes
    ("datetime", "datetime"),
    ("datetime", "date"),
    ("datetime", "time"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("decimal", "Decimal"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("collections", "Counter"),
    ("collections", "namedtuple"),
    ("enum", "Enum"),
    ("pathlib", "Path"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("qlib.data.dataset.handler", "DataHandler"),
    ("qlib.data.dataset.handler", "DataHandlerLP"),
    ("qlib.data.dataset.loader", "StaticDataLoader"),

    # NumPy reconstruction primitives. Keep this list explicit: trusting the
    # whole numpy namespace would also expose functions such as numpy.load.
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),

    # Pandas reconstruction primitives used by Series/DataFrame pickles.
    # These entries are deliberately exact. I/O helpers such as
    # pandas.read_pickle must never be added here.
    ("pandas.core.series", "Series"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.internals.blocks", "new_block"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
    ("pandas.core.indexes.datetimes", "DatetimeIndex"),
    ("pandas.core.indexes.timedeltas", "TimedeltaIndex"),
    ("pandas.core.indexes.period", "PeriodIndex"),
    ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
    ("pandas._libs.tslibs.timestamps", "Timestamp"),
    ("pandas._libs.tslibs.timedeltas", "Timedelta"),
    ("pandas._libs.tslibs.period", "Period"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas.core.arrays.timedeltas", "TimedeltaArray"),
    ("pandas.core.arrays.period", "PeriodArray"),
    ("pandas.core.arrays.categorical", "Categorical"),
    ("pandas.core.dtypes.dtypes", "CategoricalDtype"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Custom unpickler that only allows safe classes to be deserialized.

    This prevents arbitrary code execution through malicious pickle files by
    restricting deserialization to a whitelist of safe classes.

    Example:
        >>> with open("data.pkl", "rb") as f:
        ...     data = RestrictedUnpickler(f).load()
    """

    def find_class(self, module: str, name: str):
        """Override find_class to restrict allowed classes.

        Args:
            module: Module name of the class
            name: Class name

        Returns:
            The class object if it's in the whitelist

        Raises:
            pickle.UnpicklingError: If the class is not in the whitelist
        """
        if (module, name) in SAFE_PICKLE_CLASSES:
            return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Forbidden class: {module}.{name}. "
            f"Only whitelisted classes are allowed for security reasons. "
            f"This is to prevent arbitrary code execution through pickle deserialization."
        )


def restricted_pickle_load(file: BinaryIO) -> Any:
    """Safely load a pickle file with restricted classes.

    This is a drop-in replacement for pickle.load() that prevents
    arbitrary code execution by only allowing whitelisted classes.

    Args:
        file: An opened file object in binary mode

    Returns:
        The unpickled Python object

    Raises:
        pickle.UnpicklingError: If the pickle contains forbidden classes

    Example:
        >>> with open("data.pkl", "rb") as f:
        ...     data = restricted_pickle_load(f)
    """
    return RestrictedUnpickler(file).load()


def restricted_pickle_loads(data: bytes) -> Any:
    """Safely load a pickle from bytes with restricted classes.

    This is a drop-in replacement for pickle.loads() that prevents
    arbitrary code execution by only allowing whitelisted classes.

    Args:
        data: Bytes object containing pickled data

    Returns:
        The unpickled Python object

    Raises:
        pickle.UnpicklingError: If the pickle contains forbidden classes

    Example:
        >>> data = b'\\x80\\x04\\x95...'
        >>> obj = restricted_pickle_loads(data)
    """
    file_like = io.BytesIO(data)
    return RestrictedUnpickler(file_like).load()


def add_safe_class(module: str, name: str) -> None:
    """Add a class to the whitelist of safe classes for unpickling.

    Use this function to extend the whitelist if your code needs to deserialize
    additional classes. However, be very careful when adding classes, as this
    could potentially introduce security vulnerabilities.

    Args:
        module: Module name of the class (e.g., 'my_package.my_module')
        name: Class name (e.g., 'MyClass')

    Warning:
        Only add classes that you fully control and trust. Adding arbitrary
        classes from external packages could introduce security risks.

    Example:
        >>> add_safe_class('my_package.models', 'CustomModel')
    """
    SAFE_PICKLE_CLASSES.add((module, name))


def get_safe_classes() -> Set[Tuple[str, str]]:
    """Get a copy of the current whitelist of safe classes.

    Returns:
        A set of (module, name) tuples representing allowed classes
    """
    return SAFE_PICKLE_CLASSES.copy()
