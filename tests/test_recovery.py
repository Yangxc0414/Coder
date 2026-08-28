"""Tests for RecoveryStrategy."""

import pytest

from coder_agent.recovery import ErrorType, RecoveryResult, RecoveryStrategy


class TestRecoveryStrategy:
    @pytest.fixture
    def recovery(self):
        return RecoveryStrategy()

    def test_classify_format_error(self, recovery):
        assert recovery._classify(ValueError("JSON parse error")) == ErrorType.FORMAT_ERROR
        assert recovery._classify(Exception('Invalid JSON')) == ErrorType.FORMAT_ERROR

    def test_classify_api_connection(self, recovery):
        assert recovery._classify(Exception("Connection error")) == ErrorType.API_CONNECTION
        assert recovery._classify(Exception("EOF occurred")) == ErrorType.API_CONNECTION

    def test_classify_api_rate_limit(self, recovery):
        assert recovery._classify(Exception("rate limit exceeded")) == ErrorType.API_RATE_LIMIT
        assert recovery._classify(Exception("429 Too Many Requests")) == ErrorType.API_RATE_LIMIT

    def test_classify_policy_denial(self, recovery):
        assert recovery._classify(Exception("policy denied")) == ErrorType.POLICY_DENIAL

    def test_classify_loop_detected(self, recovery):
        assert recovery._classify(Exception("loop detected")) == ErrorType.LOOP_DETECTED

    def test_classify_max_steps(self, recovery):
        assert recovery._classify(Exception("max steps reached")) == ErrorType.MAX_STEPS

    def test_classify_unexpected(self, recovery):
        assert recovery._classify(Exception("something weird")) == ErrorType.UNEXPECTED

    def test_handle_format_error_recovers(self, recovery):
        recovery._error_counts[ErrorType.FORMAT_ERROR] = 1
        result = recovery.handle(ValueError("bad json"), None)
        assert result.recovered is True
        assert result.action == "format_error_retry"
        assert "format error" in result.message.lower()

    def test_handle_format_error_max_retries(self, recovery):
        recovery._error_counts[ErrorType.FORMAT_ERROR] = 3
        result = recovery.handle(ValueError("bad json"), None)
        assert result.recovered is False
        assert result.action == "max_format_errors"

    def test_handle_api_connection_retries(self, recovery):
        recovery._error_counts[ErrorType.API_CONNECTION] = 1
        result = recovery.handle(Exception("connection refused"), None)
        assert result.recovered is True
        assert result.action == "api_connection_retry"

    def test_handle_api_connection_max_retries(self, recovery):
        recovery._error_counts[ErrorType.API_CONNECTION] = 3
        result = recovery.handle(Exception("connection refused"), None)
        assert result.recovered is False
        assert result.action == "max_api_retries"

    def test_handle_policy_denial_guidance(self, recovery):
        result = recovery.handle(Exception("policy denied"), None)
        assert result.recovered is True
        assert result.action == "policy_denial_guidance"
        assert "safer alternative" in result.message.lower()

    def test_handle_loop_detected_guidance(self, recovery):
        result = recovery.handle(Exception("loop detected"), None)
        assert result.recovered is True
        assert result.action == "loop_detection_guidance"
        assert "different approach" in result.message.lower()

    def test_handle_max_steps_terminates(self, recovery):
        result = recovery.handle(Exception("max steps reached"), None)
        assert result.recovered is False
        assert result.action == "max_steps_reached"

    def test_handle_unexpected_terminates(self, recovery):
        result = recovery.handle(Exception("unknown error"), None)
        assert result.recovered is False
        assert result.action == "unexpected_error"

    def test_reset_clears_counts(self, recovery):
        recovery._error_counts[ErrorType.FORMAT_ERROR] = 2
        recovery.reset()
        assert recovery._error_counts == {}

    def test_get_stats(self, recovery):
        recovery._error_counts[ErrorType.FORMAT_ERROR] = 1
        stats = recovery.get_stats()
        assert stats[ErrorType.FORMAT_ERROR] == 1
