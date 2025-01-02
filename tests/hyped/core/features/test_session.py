from uuid import UUID

from hyped.core.features.session import ValidationSession


class TestValidationSession:
    """Test suite for the ValidationSession class."""

    def test_set_and_get_context(self):
        """Test setting and retrieving context values."""
        session = ValidationSession()
        with session:
            session.set_context("key1", "value1")
            session.set_context("key2", "value2")
            assert session.get_context("key1") == "value1"
            assert session.get_context("key2") == "value2"
            assert session.get_context("missing_key") is None

    def test_clear_specific_context(self):
        """Test clearing a specific context key."""
        session = ValidationSession()
        with session:
            session.set_context("key", "value")
            session.clear_context("key")
            assert session.get_context("key") is None

    def test_clear_all_context(self):
        """Test clearing all contexts."""
        session = ValidationSession()
        with session:
            session.set_context("key1", "value1")
            session.set_context("key2", "value2")
            session.clear_context()
            assert session.get_context("key1") is None
            assert session.get_context("key2") is None

    def test_session_id_generation(self):
        """Test that session ID is generated on first use."""
        session = ValidationSession()
        assert session.session_id is None
        with session:
            assert isinstance(session.session_id, UUID)
        assert session.session_id is None

    def test_context_cleared_on_exit(self):
        """Test that context is cleared automatically when session exits."""
        session = ValidationSession()
        with session:
            session.set_context("key", "value")
        assert session.get_context("key") is None

    def test_nested_contexts(self):
        """Test that context is only cleared after the top-level session exits."""
        session = ValidationSession()
        with session:
            session.set_context("key1", "value1")
            with session:
                session.set_context("key2", "value2")
                assert session.get_context("key1") == "value1"
                assert session.get_context("key2") == "value2"
            # After the inner block, the context should remain intact
            assert session.get_context("key1") == "value1"
            assert session.get_context("key2") == "value2"
        # After the top-level block, the context should be cleared
        assert session.get_context("key1") is None
        assert session.get_context("key2") is None
