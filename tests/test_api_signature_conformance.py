"""Guard against MCP tools passing arguments that do not line up with polygon_api.

Every tool funnels its call through ``_call_polygon(polygon.<method>, ...)``.  This
module parses ``polygon_mcp/server.py`` and checks each of those call sites against
the real ``Polygon`` signature, so a renamed or reordered API parameter fails here
instead of at runtime against Polygon.
"""

import ast
import inspect
import os
import unittest

from polygon_api import Polygon

SERVER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "polygon_mcp",
    "server.py",
)

# Local variable names that intentionally differ from the API parameter they fill.
# Keyed by API parameter name.
ALLOWED_ALIASES = {
    "problem_info": {"info"},
    "problem_statement": {"statement"},
    "file": {"file_value", "updated"},
    "type": {"file_type", "requested_type"},
    "resource_advanced_properties": {"adv"},
    "test_verdict": {"verdict"},
    "test_input": {"input_value"},
    "source": {"updated"},
    "points_policy": {"points_policy_enum"},
    "feedback_policy": {"feedback_policy_enum"},
}


class CallSite:
    def __init__(self, lineno, method, positional, keywords):
        self.lineno = lineno
        self.method = method
        self.positional = positional
        self.keywords = keywords

    def __str__(self) -> str:
        return f"server.py:{self.lineno} polygon.{self.method}"


def _collect_call_sites():
    with open(SERVER_PATH, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=SERVER_PATH)

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "_call_polygon"):
            continue
        target = node.args[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
            raise AssertionError(
                f"server.py:{node.lineno}: _call_polygon target is not polygon.<method>"
            )
        sites.append(
            CallSite(
                node.lineno,
                target.attr,
                list(node.args[1:]),
                [(kw.arg, kw.value) for kw in node.keywords],
            )
        )
    return sites


def _api_parameters(method):
    fn = getattr(Polygon, method, None)
    if fn is None:
        return None
    return [
        name for name in inspect.signature(fn).parameters if name != "self"
    ]


class ApiSignatureConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sites = _collect_call_sites()

    def test_call_sites_were_found(self) -> None:
        """Sanity check: the parser must actually see the tool call sites."""
        self.assertGreater(len(self.sites), 40)

    def test_every_called_method_exists_on_polygon(self) -> None:
        for site in self.sites:
            with self.subTest(site=str(site)):
                self.assertTrue(
                    hasattr(Polygon, site.method),
                    f"{site}: no such method on polygon_api.Polygon",
                )

    def test_arguments_bind_to_api_signature(self) -> None:
        """Arity and keyword names must satisfy the real signature."""
        for site in self.sites:
            with self.subTest(site=str(site)):
                fn = getattr(Polygon, site.method)
                signature = inspect.signature(fn)
                try:
                    signature.bind(
                        None,
                        *["<arg>"] * len(site.positional),
                        **{name: "<arg>" for name, _ in site.keywords},
                    )
                except TypeError as exc:
                    self.fail(f"{site}: {exc}")

    def test_named_arguments_land_on_matching_parameter(self) -> None:
        """A bare identifier must fill the API parameter of the same name.

        Catches silently swapped or shifted positional arguments, which the bind
        check above cannot see.
        """
        for site in self.sites:
            params = _api_parameters(site.method)
            if params is None:
                continue
            for index, node in enumerate(site.positional):
                if not isinstance(node, ast.Name):
                    continue  # expression, not a pass-through variable
                if index >= len(params):
                    continue  # reported by the bind test
                expected = params[index]
                with self.subTest(site=str(site), param=expected):
                    self.assertIn(
                        node.id,
                        {expected} | ALLOWED_ALIASES.get(expected, set()),
                        f"{site}: positional #{index} passes '{node.id}' "
                        f"but the API expects '{expected}' there",
                    )
            for name, node in site.keywords:
                if not isinstance(node, ast.Name):
                    continue
                with self.subTest(site=str(site), param=name):
                    self.assertIn(
                        node.id,
                        {name} | ALLOWED_ALIASES.get(name, set()),
                        f"{site}: keyword '{name}' is filled from '{node.id}'",
                    )

    def test_installed_client_has_the_1_2_surface(self) -> None:
        """The pin below is a pre-release; fail loudly when the old client is installed."""
        self.assertTrue(
            hasattr(Polygon, "problem_cautions"),
            "polygon_api is older than 1.2.0a1: no Polygon.problem_cautions",
        )
        self.assertIn(
            "pin",
            inspect.signature(Polygon.problem_info).parameters,
            "polygon_api is older than 1.2.0a1: problem_info takes no pin",
        )


if __name__ == "__main__":
    unittest.main()
