"""
Tests for the command injection fix in the /whois endpoint (CWE-77).

The vulnerability was: os.popen("whois " + target) — string-concatenating
user input into a shell command string passed to os.popen, which invokes
a shell and therefore interprets shell metacharacters.

The fix is: subprocess.run(["whois", target], shell=False) — user input is
passed as a discrete argument to the subprocess, never interpreted by a shell.
"""
import subprocess
import unittest
from unittest.mock import patch, MagicMock

import sys
import os

# Allow importing the application module from its directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sast_critical_high as app_module


class TestWhoisCommandInjectionFix(unittest.TestCase):
    """Validate that the /whois endpoint is not vulnerable to command injection."""

    def setUp(self):
        """Configure the Flask test client."""
        app_module.app.config["TESTING"] = True
        app_module.init_db()
        self.client = app_module.app.test_client()

    # ------------------------------------------------------------------
    # Structural / code-level tests
    # ------------------------------------------------------------------

    def test_whois_function_does_not_use_os_popen(self):
        """The whois() function must not call os.popen (the vulnerable sink)."""
        import inspect
        source = inspect.getsource(app_module.whois)
        self.assertNotIn("os.popen", source,
                         "os.popen is a shell-expanding sink and must not be used")

    def test_whois_function_does_not_use_os_system(self):
        """The whois() function must not call os.system."""
        import inspect
        source = inspect.getsource(app_module.whois)
        self.assertNotIn("os.system", source,
                         "os.system is a shell-expanding sink and must not be used")

    def test_whois_function_uses_subprocess_run_with_shell_false(self):
        """The whois() function must use subprocess.run with shell=False."""
        import inspect
        source = inspect.getsource(app_module.whois)
        # subprocess.run must be present
        self.assertIn("subprocess.run", source,
                      "subprocess.run must be the execution API")
        # shell=False must be explicit (SAST-recognisable pattern)
        self.assertIn("shell=False", source,
                      "shell=False must be explicitly set to prevent shell expansion")

    def test_whois_passes_target_as_list_element(self):
        """target must be a separate list element, not concatenated into the command string."""
        import inspect
        source = inspect.getsource(app_module.whois)
        # Detect string concatenation of target into the command (the vulnerable pattern)
        # Patterns like: "whois " + target  OR  f"whois {target}"
        import re
        concat_pattern = re.compile(
            r'["\']whois\s+["\'].*\+\s*target'   # "whois " + target
            r'|f["\']whois\s+\{target\}'           # f"whois {target}"
        )
        self.assertIsNone(
            concat_pattern.search(source),
            "target must NOT be concatenated into a shell command string",
        )

    # ------------------------------------------------------------------
    # Behavioural tests (subprocess.run is mocked to avoid network calls)
    # ------------------------------------------------------------------

    def _make_completed_process(self, stdout="", returncode=0):
        mock_proc = MagicMock(spec=subprocess.CompletedProcess)
        mock_proc.stdout = stdout
        mock_proc.returncode = returncode
        return mock_proc

    @patch("sast_critical_high.subprocess.run")
    def test_normal_domain_lookup(self, mock_run):
        """A benign domain name produces a successful response."""
        mock_run.return_value = self._make_completed_process(
            stdout="Domain: example.com\nRegistrar: IANA\n"
        )
        response = self.client.get("/whois?target=example.com")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"example.com", response.data)

    @patch("sast_critical_high.subprocess.run")
    def test_subprocess_called_with_list_not_string(self, mock_run):
        """subprocess.run must receive a list with 'whois' as first element."""
        mock_run.return_value = self._make_completed_process(stdout="some whois output")
        self.client.get("/whois?target=example.com")

        mock_run.assert_called_once()
        call_args, call_kwargs = mock_run.call_args

        # First positional argument must be a list, not a string
        cmd = call_args[0]
        self.assertIsInstance(cmd, list,
                              "Command must be a list to prevent shell injection")
        self.assertEqual(cmd[0], "whois",
                         "First list element must be the hard-coded command 'whois'")
        self.assertEqual(cmd[1], "example.com",
                         "Second list element must be the verbatim user target")

    @patch("sast_critical_high.subprocess.run")
    def test_shell_kwarg_is_false(self, mock_run):
        """subprocess.run must be invoked with shell=False."""
        mock_run.return_value = self._make_completed_process(stdout="output")
        self.client.get("/whois?target=example.com")

        _, call_kwargs = mock_run.call_args
        self.assertFalse(
            call_kwargs.get("shell", True),
            "shell must be explicitly False to prevent shell metacharacter expansion",
        )

    @patch("sast_critical_high.subprocess.run")
    def test_semicolon_injection_payload_not_executed_as_shell(self, mock_run):
        """Payload with semicolon is passed as a literal argument, not executed by shell."""
        malicious_target = "example.com; cat /etc/passwd"
        mock_run.return_value = self._make_completed_process(stdout="whois output only")

        response = self.client.get(f"/whois?target={malicious_target}")
        self.assertEqual(response.status_code, 200)

        # subprocess.run must receive the full malicious string as a single argument,
        # proving no shell is interpreting the semicolon.
        call_args, _ = mock_run.call_args
        cmd = call_args[0]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[1], malicious_target,
                         "The payload must be forwarded verbatim as a single arg, not split by shell")

    @patch("sast_critical_high.subprocess.run")
    def test_pipe_injection_payload_not_executed_as_shell(self, mock_run):
        """Payload with pipe character is passed as a literal argument."""
        malicious_target = "example.com | id"
        mock_run.return_value = self._make_completed_process(stdout="whois output only")

        self.client.get(f"/whois?target={malicious_target}")

        call_args, _ = mock_run.call_args
        cmd = call_args[0]
        self.assertEqual(cmd[1], malicious_target)

    @patch("sast_critical_high.subprocess.run")
    def test_backtick_injection_payload_not_executed_as_shell(self, mock_run):
        """Payload with backtick command substitution is passed as a literal argument."""
        malicious_target = "example.com`id`"
        mock_run.return_value = self._make_completed_process(stdout="whois output only")

        self.client.get(f"/whois?target={malicious_target}")

        call_args, _ = mock_run.call_args
        cmd = call_args[0]
        self.assertEqual(cmd[1], malicious_target)

    @patch("sast_critical_high.subprocess.run")
    def test_dollar_substitution_payload_not_executed_as_shell(self, mock_run):
        """Payload with $() command substitution is passed as a literal argument."""
        malicious_target = "example.com$(cat /etc/shadow)"
        mock_run.return_value = self._make_completed_process(stdout="whois output only")

        self.client.get(f"/whois?target={malicious_target}")

        call_args, _ = mock_run.call_args
        cmd = call_args[0]
        self.assertEqual(cmd[1], malicious_target)

    @patch("sast_critical_high.subprocess.run")
    def test_ampersand_injection_payload_not_executed_as_shell(self, mock_run):
        """Payload with && operator is passed as a literal argument."""
        malicious_target = "example.com && rm -rf /"
        mock_run.return_value = self._make_completed_process(stdout="whois output only")

        self.client.get(f"/whois?target={malicious_target}")

        call_args, _ = mock_run.call_args
        cmd = call_args[0]
        self.assertEqual(cmd[1], malicious_target)

    @patch("sast_critical_high.subprocess.run")
    def test_empty_target_handled(self, mock_run):
        """Empty target parameter is forwarded safely."""
        mock_run.return_value = self._make_completed_process(stdout="")
        response = self.client.get("/whois?target=")
        self.assertEqual(response.status_code, 200)
        call_args, _ = mock_run.call_args
        cmd = call_args[0]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "whois")

    @patch("sast_critical_high.subprocess.run")
    def test_response_contains_subprocess_stdout(self, mock_run):
        """Response body comes from subprocess stdout, not from shell command expansion."""
        expected_output = "Registrar: Example Inc.\nExpiry: 2030-01-01\n"
        mock_run.return_value = self._make_completed_process(stdout=expected_output)

        response = self.client.get("/whois?target=example.com")
        self.assertEqual(response.data.decode(), expected_output)


if __name__ == "__main__":
    unittest.main()
