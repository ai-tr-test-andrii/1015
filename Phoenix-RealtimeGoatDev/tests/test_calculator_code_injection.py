"""
Tests for CWE-94 Code Injection remediation in the /calc endpoint.

The vulnerability was: eval(user_input) at sast_critical_high.py line 106.
The fix replaces eval() with ast.literal_eval(), which only parses Python
literals and raises ValueError/TypeError for anything else.
"""

import pytest
import sys
import os

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
# Positive cases — safe literal values that should still work
# ---------------------------------------------------------------------------

class TestCalculatorLegitimateInput:
    """The /calc endpoint should parse Python literals successfully."""

    def test_integer_literal(self, client):
        resp = client.get("/calc?expr=42")
        assert resp.status_code == 200
        assert b"42" in resp.data

    def test_float_literal(self, client):
        resp = client.get("/calc?expr=3.14")
        assert resp.status_code == 200
        assert b"3.14" in resp.data

    def test_negative_integer(self, client):
        resp = client.get("/calc?expr=-7")
        assert resp.status_code == 200
        assert b"-7" in resp.data

    def test_boolean_true(self, client):
        resp = client.get("/calc?expr=True")
        assert resp.status_code == 200
        assert b"True" in resp.data

    def test_none_literal(self, client):
        resp = client.get("/calc?expr=None")
        assert resp.status_code == 200
        assert b"None" in resp.data

    def test_string_literal(self, client):
        resp = client.get("/calc?expr='hello'")
        assert resp.status_code == 200
        assert b"hello" in resp.data

    def test_list_literal(self, client):
        resp = client.get("/calc?expr=[1,2,3]")
        assert resp.status_code == 200
        assert b"[1, 2, 3]" in resp.data

    def test_default_when_no_expr(self, client):
        """Default value '0' must still return successfully."""
        resp = client.get("/calc")
        assert resp.status_code == 200
        assert b"0" in resp.data


# ---------------------------------------------------------------------------
# Security cases — code-injection payloads MUST be rejected
# ---------------------------------------------------------------------------

class TestCalculatorCodeInjectionBlocked:
    """
    These payloads would execute arbitrary code under the old eval() sink.
    With ast.literal_eval() they must be rejected with HTTP 400.
    """

    def test_os_system_call(self, client):
        """Attempt to call os.system() via eval."""
        resp = client.get("/calc?expr=__import__('os').system('id')")
        assert resp.status_code == 400

    def test_open_file_read(self, client):
        """Attempt to read /etc/passwd via eval."""
        resp = client.get("/calc?expr=open('/etc/passwd').read()")
        assert resp.status_code == 400

    def test_subprocess_call(self, client):
        """Attempt to spawn a subprocess via eval."""
        resp = client.get("/calc?expr=__import__('subprocess').check_output(['id'])")
        assert resp.status_code == 400

    def test_exec_builtin(self, client):
        """Attempt to embed exec() inside eval."""
        resp = client.get("/calc?expr=exec('import os; os.system(\"id\")')")
        assert resp.status_code == 400

    def test_lambda_expression(self, client):
        """Lambda is not a literal — must be rejected."""
        resp = client.get("/calc?expr=lambda: None")
        assert resp.status_code == 400

    def test_arithmetic_expression(self, client):
        """
        '1+1' is an arithmetic expression, not a pure literal.
        ast.literal_eval rejects it; the endpoint should return 400.
        This confirms the secure baseline: only plain literals are accepted.
        """
        resp = client.get("/calc?expr=1%2B1")
        assert resp.status_code == 400

    def test_dunder_class_traversal(self, client):
        """Attempt class-hierarchy traversal to get builtins."""
        resp = client.get("/calc?expr=().__class__.__bases__[0].__subclasses__()")
        assert resp.status_code == 400

    def test_empty_string(self, client):
        """Empty expression is not a valid literal — should return 400."""
        resp = client.get("/calc?expr=")
        assert resp.status_code == 400

    def test_multiline_injection(self, client):
        """Multi-statement string via newline injection."""
        payload = "1\nimport os"
        resp = client.get(f"/calc?expr={payload}")
        assert resp.status_code == 400

    def test_globals_access(self, client):
        """Attempt to access globals() via eval."""
        resp = client.get("/calc?expr=globals()")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Response content guard — ensure no error detail leaks sensitive info
# ---------------------------------------------------------------------------

class TestCalculatorErrorResponse:
    """Error responses must not expose internal state or stack traces."""

    def test_400_body_is_safe(self, client):
        resp = client.get("/calc?expr=__import__('os').getcwd()")
        assert resp.status_code == 400
        # The response body should be the controlled error message, not a traceback.
        body = resp.data.decode("utf-8", errors="replace")
        assert "Traceback" not in body
        assert "__import__" not in body
