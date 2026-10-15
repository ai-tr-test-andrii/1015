"""
Tests for the Log Forging (CWE-117) remediation in the /log-action endpoint.

The fix replaces string concatenation in logger.info() with parameterized
logging (logger.info("...", action)) so that unsanitized user input is never
concatenated into the log message string before it reaches the logging sink.
"""

import logging
import pytest

# Import the Flask application under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sast_critical_high import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestLogActionEndpoint:
    """Tests for the /log-action route."""

    def test_normal_action_returns_logged_response(self, client):
        """A normal action value should produce a 200 response with 'Logged'."""
        response = client.get("/log-action?action=login")
        assert response.status_code == 200
        assert response.data == b"Logged"

    def test_empty_action_returns_logged_response(self, client):
        """Missing action parameter should still work (defaults to empty string)."""
        response = client.get("/log-action")
        assert response.status_code == 200
        assert response.data == b"Logged"

    def test_no_string_concatenation_in_log_message(self, caplog):
        """
        Verify the fix: the log record message must be the format string only
        — *not* the pre-interpolated string with user input embedded in it.

        Python's parameterized logging stores the format string in LogRecord.msg
        and the args in LogRecord.args.  String concatenation or %-formatting at
        the call site would have already collapsed the two into a single string
        before the record is created, losing the separation.
        """
        with app.test_client() as client:
            with caplog.at_level(logging.INFO):
                client.get("/log-action?action=login")

        # Find the relevant log record
        records = [r for r in caplog.records if "performed action" in r.getMessage()]
        assert records, "Expected at least one 'performed action' log record"

        record = records[0]
        # With parameterized logging the raw .msg should be the format string,
        # not the fully-rendered message.
        assert record.msg == "User performed action: %s", (
            "logger.info() must use a %%s format string, not string concatenation. "
            f"Got msg={record.msg!r}"
        )
        # The user-supplied value should be passed as an argument, not baked in.
        assert record.args == ("login",), (
            f"Expected args=('login',) but got {record.args!r}"
        )

    def test_log_injection_payload_does_not_forge_new_log_line(self, caplog):
        """
        An attacker who supplies a newline in the action parameter must NOT be
        able to inject a new, forged log entry.

        With parameterized logging the logging framework handles the substitution;
        the injected newline stays inside the single LogRecord rather than
        creating a second record visible via caplog.records.
        """
        forged_suffix = "\nINFO User performed action: admin_delete_all"
        with app.test_client() as client:
            with caplog.at_level(logging.INFO):
                client.get(f"/log-action?action=harmless{forged_suffix}")

        # There must be exactly one record whose message template contains our sentinel.
        action_records = [
            r for r in caplog.records if "performed action" in r.getMessage()
        ]
        assert len(action_records) == 1, (
            "Log forging detected: expected exactly 1 'performed action' record, "
            f"got {len(action_records)}"
        )

    def test_log_injection_cr_payload_single_record(self, caplog):
        """
        A carriage-return injection payload must not forge a second log record.
        """
        cr_payload = "\rFAKE_LOG_ENTRY: privilege_escalation"
        with app.test_client() as client:
            with caplog.at_level(logging.INFO):
                client.get(f"/log-action?action=x{cr_payload}")

        action_records = [
            r for r in caplog.records if "performed action" in r.getMessage()
        ]
        assert len(action_records) == 1, (
            "CR injection produced more than one log record; log forging is still possible."
        )

    def test_special_characters_do_not_raise_exception(self, client):
        """
        Special characters (%, newlines, null bytes expressed as their escape)
        must not cause an unhandled exception in the endpoint.
        """
        payloads = [
            "action%20with%20spaces",
            "100%25done",          # literal % after URL decode = '100%done'
            "line1\nline2",
            "tab\there",
            "\x00nullbyte",        # NUL expressed as hex escape, not a raw byte
        ]
        for payload in payloads:
            response = client.get(f"/log-action", query_string={"action": payload})
            assert response.status_code == 200, (
                f"Endpoint raised an error for payload {payload!r}"
            )
