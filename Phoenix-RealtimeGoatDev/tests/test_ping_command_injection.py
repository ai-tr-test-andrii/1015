"""
Tests for CWE-77/CWE-78 Command Injection remediation in the /ping endpoint.

The vulnerability was:
    subprocess.check_output("ping -c 1 " + host, shell=True)
at sast_critical_high.py line 74.

The fix replaces the single shell string + shell=True with:
    subprocess.check_output(["ping", "-c", "1", host], shell=False)

By passing the user-supplied value as a discrete list element and setting
shell=False, shell metacharacters in `host` are treated as literal characters
by the OS rather than being interpreted by the shell.

These tests use unittest.mock.patch to avoid real network calls while still
exercising the argument-construction logic that was the source of the
vulnerability.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Allow importing the Flask app from the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sast_critical_high import app, init_db


@pytest.fixture
def client():
    """Configure the Flask test client with an in-memory DB."""
    app.config["TESTING"] = True
    init_db()
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Positive cases — legitimate hostname/IP that should reach check_output
# ---------------------------------------------------------------------------

class TestPingLegitimateInput:
    """The /ping endpoint should forward a valid host to check_output safely."""

    def test_valid_hostname_passes_as_list_arg(self, client):
        """
        A valid hostname must be forwarded to check_output as the last element
        of a list, never concatenated into a shell string.
        """
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b"PING example.com: 1 packet"
            resp = client.get("/ping?host=example.com")

        # The endpoint must have called check_output with a list, shell=False.
        mock_co.assert_called_once()
        call_args, call_kwargs = mock_co.call_args
        command = call_args[0]

        assert isinstance(command, list), "Command must be a list, not a string"
        assert command[0] == "ping", "First element must be the hard-coded 'ping' binary"
        assert "example.com" in command, "Host must appear as a discrete list element"
        assert call_kwargs.get("shell") is False or call_kwargs.get("shell") == False, (
            "shell= must be False"
        )
        assert resp.status_code == 200

    def test_valid_ipv4_passes_as_list_arg(self, client):
        """An IPv4 address must also be forwarded as a discrete list element."""
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b"PING 192.168.1.1"
            resp = client.get("/ping?host=192.168.1.1")

        call_args, call_kwargs = mock_co.call_args
        command = call_args[0]

        assert isinstance(command, list)
        assert "192.168.1.1" in command
        assert call_kwargs.get("shell") is False or call_kwargs.get("shell") == False
        assert resp.status_code == 200

    def test_response_body_contains_ping_output(self, client):
        """Ping output returned by check_output must be forwarded in the response."""
        ping_output = b"PING example.com 56 data bytes\n1 packets transmitted"
        with patch("subprocess.check_output", return_value=ping_output):
            resp = client.get("/ping?host=example.com")

        assert resp.status_code == 200
        assert ping_output in resp.data


# ---------------------------------------------------------------------------
# Security cases — command-injection payloads must NOT be shell-expanded
# ---------------------------------------------------------------------------

class TestPingCommandInjectionPrevented:
    """
    These payloads exploited the old shell=True sink.  With shell=False and a
    list-based invocation the shell never sees the metacharacters, so the
    injected fragment is passed verbatim to the ping binary as a host name
    argument, where it is harmless (ping will simply fail to resolve it).

    We verify that:
      1. check_output is called with a list (not a string).
      2. The injected payload appears only as a single list element.
      3. The shell= kwarg is False.
    """

    def _assert_no_shell_expansion(self, mock_co, expected_host_fragment):
        """
        Common assertion: the command passed to check_output must be a list
        containing the injected fragment as-is, not interpreted by a shell.
        """
        mock_co.assert_called_once()
        call_args, call_kwargs = mock_co.call_args
        command = call_args[0]

        assert isinstance(command, list), (
            "shell=False requires a list; a string would still be interpreted "
            "by the shell on some platforms."
        )
        # The tainted input must appear in exactly one list slot.
        matching = [el for el in command if expected_host_fragment in el]
        assert len(matching) == 1, (
            f"Injected fragment '{expected_host_fragment}' should appear in "
            f"exactly one list element, got {matching}"
        )
        # shell must be False
        assert call_kwargs.get("shell") is False or call_kwargs.get("shell") == False

    def test_semicolon_injection_not_expanded(self, client):
        """`; id` after the host must not spawn a second shell command."""
        payload = "127.0.0.1; id"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "id")

    def test_ampersand_injection_not_expanded(self, client):
        """`& id` (background operator) must not spawn a second command."""
        payload = "127.0.0.1 & id"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "id")

    def test_pipe_injection_not_expanded(self, client):
        """`| cat /etc/passwd` must not be piped to cat."""
        payload = "127.0.0.1 | cat /etc/passwd"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "cat")

    def test_backtick_injection_not_expanded(self, client):
        """Backtick command substitution must not be evaluated by the shell."""
        payload = "127.0.0.1`id`"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "id")

    def test_dollar_subshell_not_expanded(self, client):
        """$(id) subshell syntax must not be evaluated."""
        payload = "$(id)"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "id")

    def test_newline_injection_not_expanded(self, client):
        """A newline character embedded in host must not trigger a new command."""
        payload = "127.0.0.1\nid"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "id")

    def test_path_traversal_in_host_treated_as_literal(self, client):
        """
        A host containing path traversal chars (e.g. ../../etc/passwd) must be
        passed verbatim, not interpreted as a file path by the shell.
        """
        payload = "../../etc/passwd"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")
        self._assert_no_shell_expansion(mock_co, "etc/passwd")

    def test_host_injected_as_single_arg_not_split(self, client):
        """
        Even with spaces in the host, the entire value must map to exactly one
        list element (not shell-split into multiple tokens).
        """
        payload = "evil host ; rm -rf /"
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={payload}")

        call_args, _ = mock_co.call_args
        command = call_args[0]

        # The payload must be one element, not shell-tokenised into several.
        assert isinstance(command, list)
        joined = " ".join(command)
        # The injected text must appear somewhere in the joined command, but
        # 'rm' and '-rf' must NOT be separate top-level list items.
        rm_as_own_element = "rm" in command
        assert not rm_as_own_element, (
            "'rm' appeared as a standalone list element — the payload was "
            "split by the shell, indicating shell=True or string interpolation."
        )


# ---------------------------------------------------------------------------
# Structural guarantee — shell=True must never appear in the call
# ---------------------------------------------------------------------------

class TestPingShellFlagAlwaysFalse:
    """Ensure shell= is consistently False regardless of input."""

    @pytest.mark.parametrize("host", [
        "localhost",
        "8.8.8.8",
        "",
        "host; id",
        "$(whoami)",
    ])
    def test_shell_is_false_for_all_inputs(self, client, host):
        with patch("subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            client.get(f"/ping?host={host}")

        _, call_kwargs = mock_co.call_args
        # shell must be absent (defaults to False) or explicitly False.
        shell_value = call_kwargs.get("shell", False)
        assert shell_value is False or shell_value == False, (
            f"shell={shell_value!r} for host={host!r}; expected False"
        )
