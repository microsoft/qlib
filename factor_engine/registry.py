# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .operators import Operator

class OperatorRegistry:
    """
    A central registry for managing and providing instances of operators.
    This pattern allows for decoupling operator definitions from their usage
    and facilitates easy extension with custom operators.
    """
    def __init__(self):
        """Initializes the registry."""
        self._registry: Dict[str, Type[Operator]] = {}

    def register(self, operator_cls: Type[Operator]):
        """
        Registers an operator class with the registry.
        The operator's `name` attribute is used as the key.

        Args:
            operator_cls: The operator class to register (must be a subclass of Operator).
        
        Raises:
            ValueError: If an operator with the same name is already registered.
            TypeError: If the class to be registered is not a subclass of Operator.
        """
        # We need to import here to avoid circular dependency
        from .operators import Operator
        if not issubclass(operator_cls, Operator):
            raise TypeError(f"Only subclasses of Operator can be registered, but got {operator_cls}")

        name = operator_cls().name # Instantiate to get the default name
        if name in self._registry:
            raise ValueError(f"Operator '{name}' is already registered.")
        
        self._registry[name] = operator_cls

    def get(self, name: str, **kwargs) -> Operator:
        """
        Retrieves an initialized operator instance from the registry.

        Args:
            name (str): The name of the operator to retrieve.
            **kwargs: Initialization arguments for the operator (e.g., `window`).

        Returns:
            Operator: An instance of the requested operator.
            
        Raises:
            ValueError: If the operator is not found in the registry.
        """
        if name not in self._registry:
            raise ValueError(f"Operator '{name}' is not registered.")
        
        operator_cls = self._registry[name]
        return operator_cls(**kwargs)

# A global instance of the registry for convenient access
op_registry = OperatorRegistry()
