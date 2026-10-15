"""
Tests for CWE-94 Code Injection fix in the /calc endpoint.

The vulnerability was: user-supplied `expr` query parameter was passed
directly to eval(), allowing arbitrary code execution.

The fix replaces eval() with a safe AST-based evaluator (_safe_eval) that
only permits numeric constants and the basic arithmetic operators.
"""
import ast
import operator
import pytest

# ---------------------------------------------------------------------------
# Import the helper directly so we can unit-test it without running the full
# Flask application (avoids needing a database, etc.)
# ---------------------------------------------------------------------------
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sast_critical_high import app, _safe_eval, _SAFE_OPS


# ---------------------------------------------------------------------------
# Flask test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ===========================================================================
# Unit tests for _safe_eval (the safe AST evaluator helper)
# ===========================================================================

class TestSafeEvalHelper:
    """Direct unit tests of the _safe_eval helper function."""

    def _parse(self, expr):
        return ast.parse(expr, mode="eval")

    # --- Legitimate arithmetic ---

    def test_integer_literal(self):
        assert _safe_eval(self._parse("42")) == 42

    def test_float_literal(self):
        assert _safe_eval(self._parse("3.14")) == pytest.approx(3.14)

    def test_addition(self):
        assert _safe_eval(self._parse("2 + 3")) == 5

    def test_subtraction(self):
        assert _safe_eval(self._parse("10 - 4")) == 6

    def test_multiplication(self):
        assert _safe_eval(self._parse("6 * 7")) == 42

    def test_division(self):
        assert _safe_eval(self._parse("10 / 4")) == pytest.approx(2.5)

    def test_floor_division(self):
        assert _safe_eval(self._parse("10 // 3")) == 3

    def test_modulo(self):
        assert _safe_eval(self._parse("10 % 3")) == 1

    def test_power(self):
        assert _safe_eval(self._parse("2 ** 10")) == 1024

    def test_unary_negation(self):
        assert _safe_eval(self._parse("-5")) == -5

    def test_unary_positive(self):
        assert _safe_eval(self._parse("+5")) == 5

    def test_nested_expression(self):
        assert _safe_eval(self._parse("(2 + 3) * 4")) == 20

    def test_complex_nested_expression(self):
        assert _safe_eval(self._parse("100 / (2 + 3) - 1")) == pytest.approx(19.0)

    # --- Blocked constructs — must raise ValueError ---

    def test_blocks_string_literal(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(self._parse('"hello"'))

    def test_blocks_function_call(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("__import__('os')", mode="eval"))

    def test_blocks_attribute_access(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("os.getcwd()", mode="eval"))

    def test_blocks_name_reference(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("x", mode="eval"))

    def test_blocks_list_literal(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("[1, 2, 3]", mode="eval"))

    def test_blocks_dict_literal(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("{}", mode="eval"))

    def test_blocks_bool_literal(self):
        # Booleans are subclasses of int in Python, but we only allow int/float.
        # True == 1 as an int — the evaluator should reject the Bool constant type.
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("True", mode="eval"))

    def test_blocks_lambda(self):
        with pytest.raises((ValueError, SyntaxError, TypeError)):
            _safe_eval(ast.parse("lambda: None", mode="eval"))

    def test_blocks_comparison(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("1 == 1", mode="eval"))

    def test_blocks_boolean_op(self):
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(ast.parse("True or False", mode="eval"))


# ===========================================================================
# Integration tests via the Flask /calc endpoint
# ===========================================================================

class TestCalculatorEndpointLegitimate:
    """Verify that valid arithmetic still works correctly."""

    def test_simple_addition(self, client):
        rv = client.get("/calc?expr=2%2B3")
        assert rv.status_code == 200
        assert b"5" in rv.data

    def test_simple_subtraction(self, client):
        rv = client.get("/calc?expr=10-4")
        assert rv.status_code == 200
        assert b"6" in rv.data

    def test_multiplication(self, client):
        rv = client.get("/calc?expr=6*7")
        assert rv.status_code == 200
        assert b"42" in rv.data

    def test_division(self, client):
        rv = client.get("/calc?expr=10%2F4")
        assert rv.status_code == 200
        assert b"2.5" in rv.data

    def test_power(self, client):
        rv = client.get("/calc?expr=2**10")
        assert rv.status_code == 200
        assert b"1024" in rv.data

    def test_default_value(self, client):
        """No expr parameter → default "0" → result 0."""
        rv = client.get("/calc")
        assert rv.status_code == 200
        assert b"0" in rv.data

    def test_nested_parentheses(self, client):
        rv = client.get("/calc?expr=(2%2B3)*4")
        assert rv.status_code == 200
        assert b"20" in rv.data


class TestCalculatorEndpointCodeInjection:
    """
    Verify that code injection payloads are REJECTED and do NOT execute.

    Each test confirms the endpoint returns HTTP 400 (not 200) and does
    not return output that would indicate successful code execution.
    """

    def test_os_system_import(self, client):
        """Classic code injection: __import__('os').system('id')"""
        rv = client.get("/calc?expr=__import__(%27os%27).system(%27id%27)")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_exec_payload(self, client):
        """exec() call must be blocked."""
        rv = client.get("/calc?expr=exec(%27import+os%27)")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_open_file_payload(self, client):
        """Attempt to read /etc/passwd via open()."""
        rv = client.get("/calc?expr=open(%27%2Fetc%2Fpasswd%27).read()")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_getattr_payload(self, client):
        """getattr abuse for attribute access."""
        rv = client.get("/calc?expr=getattr(__builtins__,%27eval%27)(1%2B1)")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_string_literal_payload(self, client):
        """A plain string should be rejected (not a numeric expression)."""
        rv = client.get("/calc?expr=%22hello%22")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_boolean_expression_payload(self, client):
        """Boolean expressions should be rejected."""
        rv = client.get("/calc?expr=True")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_list_payload(self, client):
        """List literals must be rejected."""
        rv = client.get("/calc?expr=%5B1%2C2%2C3%5D")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_name_reference_payload(self, client):
        """Bare name references (e.g. variable names) must be rejected."""
        rv = client.get("/calc?expr=x")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_comparison_operator_payload(self, client):
        """Comparison operators should be rejected (not in allowed ops)."""
        rv = client.get("/calc?expr=1==1")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_empty_expression(self, client):
        """An empty string is not a valid expression."""
        rv = client.get("/calc?expr=")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_semicolon_chaining(self, client):
        """Semicolon chaining (statement separator) must be rejected."""
        rv = client.get("/calc?expr=1%3B__import__(%27os%27)")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_dunder_attack(self, client):
        """Access to __class__ or similar dunder attributes must be rejected."""
        rv = client.get("/calc?expr=(1).__class__.__bases__[0].__subclasses__()")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_format_string_attack(self, client):
        """f-string injection style — must be rejected."""
        rv = client.get("/calc?expr=f%22{__import__('os').getcwd()}%22")
        assert rv.status_code == 400
        assert b"Invalid expression" in rv.data

    def test_no_code_execution_side_effect(self, client):
        """
        Verify that the payload does not execute by checking the response
        does not contain typical shell-command output markers.
        """
        import tempfile, os
        sentinel = os.path.join(tempfile.gettempdir(), "calc_injection_sentinel.txt")
        # Remove sentinel if it exists from a previous run
        if os.path.exists(sentinel):
            os.remove(sentinel)

        # Try to create the file via injection — if eval were still present this would work
        payload = f"open('{sentinel}','w').write('pwned')"
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        rv = client.get(f"/calc?expr={encoded}")

        assert rv.status_code == 400, "Injection payload should be rejected with 400"
        assert not os.path.exists(sentinel), (
            "Injection payload created sentinel file — eval() was NOT properly removed!"
        )
