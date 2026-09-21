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

ARTIFACT_MIGRATION_URL = "https://qlib.readthedocs.io/en/latest/start/artifact_migration.html"


def validate_trusted(trusted: bool) -> bool:
    if not isinstance(trusted, bool):
        raise TypeError(f"`trusted` must be a bool. Migration guide: {ARTIFACT_MIGRATION_URL}")
    return trusted


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
    # Protocol 5 uses _frombuffer instead of _reconstruct for numeric arrays.
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.ma.core", "_mareconstruct"),
    # NumPy 1.x and 2.x pickle this class under different module paths.
    ("numpy.ma.core", "MaskedArray"),
    ("numpy.ma", "MaskedArray"),
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
    ("pandas.core.indexes.interval", "_new_IntervalIndex"),
    ("pandas.core.indexes.interval", "IntervalIndex"),
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
    ("pandas.core.dtypes.dtypes", "PeriodDtype"),
    ("pandas.core.dtypes.dtypes", "IntervalDtype"),
    ("pandas.core.dtypes.dtypes", "SparseDtype"),
    ("pandas.core.arrays.sparse.dtype", "SparseDtype"),
    ("pandas.core.arrays.interval", "IntervalArray"),
    ("pandas._libs.interval", "__pyx_unpickle_IntervalMixin"),
    ("pandas.core.arrays.sparse.array", "SparseArray"),
    ("pandas._libs.sparse", "IntIndex"),
    ("pandas._libs.sparse", "BlockIndex"),
    ("pandas.core.dtypes.dtypes", "DatetimeTZDtype"),
    ("pandas._libs.tslibs.nattype", "__nat_unpickle"),
    ("pandas._libs.missing", "NA"),
    # DatetimeIndex/PeriodIndex retain their frequency and timezone metadata.
    ("pandas._libs.tslibs.offsets", "Day"),
    ("pandas._libs.tslibs.offsets", "BusinessDay"),
    ("pandas._libs.tslibs.offsets", "Week"),
    ("pandas._libs.tslibs.offsets", "MonthBegin"),
    ("pandas._libs.tslibs.offsets", "MonthEnd"),
    ("pandas._libs.tslibs.offsets", "BusinessMonthBegin"),
    ("pandas._libs.tslibs.offsets", "BusinessMonthEnd"),
    ("pandas._libs.tslibs.offsets", "QuarterBegin"),
    ("pandas._libs.tslibs.offsets", "QuarterEnd"),
    ("pandas._libs.tslibs.offsets", "YearBegin"),
    ("pandas._libs.tslibs.offsets", "YearEnd"),
    ("pandas._libs.tslibs.offsets", "Hour"),
    ("pandas._libs.tslibs.offsets", "Minute"),
    ("pandas._libs.tslibs.offsets", "Second"),
    ("pandas._libs.tslibs.offsets", "Milli"),
    ("pandas._libs.tslibs.offsets", "Micro"),
    ("pandas._libs.tslibs.offsets", "Nano"),
    ("pytz", "_UTC"),
    ("pytz", "_p"),
    # Nullable arrays serialize their masks and dtype objects as well as data.
    ("pandas.core.arrays.integer", "IntegerArray"),
    ("pandas.core.arrays.integer", "Int8Dtype"),
    ("pandas.core.arrays.integer", "Int16Dtype"),
    ("pandas.core.arrays.integer", "Int32Dtype"),
    ("pandas.core.arrays.integer", "Int64Dtype"),
    ("pandas.core.arrays.integer", "UInt8Dtype"),
    ("pandas.core.arrays.integer", "UInt16Dtype"),
    ("pandas.core.arrays.integer", "UInt32Dtype"),
    ("pandas.core.arrays.integer", "UInt64Dtype"),
    ("pandas.core.arrays.floating", "FloatingArray"),
    ("pandas.core.arrays.floating", "Float32Dtype"),
    ("pandas.core.arrays.floating", "Float64Dtype"),
    ("pandas.core.arrays.boolean", "BooleanArray"),
    ("pandas.core.arrays.boolean", "BooleanDtype"),
    ("pandas.core.arrays.string_", "StringArray"),
    ("pandas.core.arrays.string_", "StringDtype"),
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
            f"This is to prevent arbitrary code execution through pickle deserialization. "
            f"Migration guide: {ARTIFACT_MIGRATION_URL}"
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
