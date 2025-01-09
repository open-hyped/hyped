"""Module for managing lightweight session-based contextual data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable
from uuid import UUID, uuid4


@dataclass
class ValidationSession:
    """A lightweight session context manager for managing key-value contextual data.

    This class provides a simple way to store, retrieve, and clear context during
    a specific scope of execution. Contexts are stored as key-value pairs in an
    internal dictionary, which is cleared automatically when the top-level session ends.
    """

    _session_id: None | UUID = field(default=None, init=False)
    """A session ID unique to each runtime of a session."""

    _contexts: dict[Hashable, Any] = field(default_factory=dict, init=False)
    """A dictionary to store the contextual data."""

    _ref_count: int = field(default=0, init=False)
    """A reference counter to track nested with statements."""

    @property
    def session_id(self) -> UUID:
        """The unique session ID, generated on demand if not already initialized.

        Returns:
            UUID: A unique identifier for the session.
        """
        return self._session_id

    def set_context(self, key: Hashable, value: Any) -> None:
        """Set a context value associated with a specific key.

        Args:
            key (Hashable): The key for the context entry.
            value (Any): The value to associate with the key.
        """
        self._contexts[key] = value

    def get_context(self, key: Hashable) -> Any:
        """Retrieve the context value associated with a specific key.

        Args:
            key (Hashable): The key for the context entry to retrieve.

        Returns:
            Any: The value associated with the key, or None if the key is not found.
        """
        return self._contexts.get(key)

    def clear_context(self, key: None | Hashable = None) -> None:
        """Clear one or all contexts.

        Args:
            key (None | Hashable): The key of the context entry to clear. If None,
                all contexts are cleared. Defaults to None.
        """
        if key is None:
            self._contexts.clear()
        else:
            self._contexts.pop(key, None)

    def __enter__(self) -> ValidationSession:
        """Enter the context management block, returning the instance itself.

        Returns:
            ValidationSession: The instance of this class.
        """
        if self._ref_count == 0 and self._session_id is None:
            self._session_id = uuid4()
        self._ref_count += 1
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the context management block.

        Cleares all stored contexts only if it is the top-level session.

        Args:
            exc_type (type): The exception type, if any.
            exc_value (Exception): The exception value, if any.
            traceback (traceback): The traceback object, if any.
        """
        self._ref_count -= 1
        if self._ref_count == 0:
            self._session_id = None
            self.clear_context()
