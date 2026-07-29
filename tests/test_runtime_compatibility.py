"""Smoke-test the real modelling packages in isolated Python processes.

The protocol suites intentionally replace selected NeKo and PhysiCell modules
with lightweight test doubles. These checks run in child interpreters so those
doubles cannot hide an incompatible installed package.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_in_clean_interpreter(source: str) -> None:
    """Execute a runtime contract check without pytest's import stubs."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Runtime compatibility subprocess failed.\n"
            f"stdout:\n{completed.stdout or '(empty)'}\n"
            f"stderr:\n{completed.stderr or '(empty)'}",
            pytrace=False,
        )


def test_neko_runtime_exposes_server_and_history_contracts() -> None:
    _run_in_clean_interpreter(
        """
        from neko._outputs.exports import Exports
        from neko.core.network import Network
        from neko.core.tools import is_connected
        from neko.inputs import Universe, signor
        import pandas as pd

        assert Exports is not None
        assert Universe is not None
        assert callable(signor)
        assert callable(is_connected)

        required_history_api = {
            "checkout",
            "compare_states",
            "history_graph",
            "history_html",
            "list_states",
            "redo",
            "set_max_history",
            "undo",
        }
        missing = sorted(
            method
            for method in required_history_api
            if not hasattr(Network, method)
        )
        assert not missing, f"NeKo Network is missing history methods: {missing}"

        resources = pd.DataFrame(
            [
                {
                    "source": "P04637",
                    "target": "P38398",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "form_complex": False,
                }
            ]
        )
        network = Network(resources=resources)
        states = network.list_states()
        assert isinstance(states, list)
        assert network.history_graph() is not None
        assert isinstance(network.history_html(), str)
        """
    )


def test_maboss_runtime_exposes_plotting_contract_and_compiled_engine() -> None:
    _run_in_clean_interpreter(
        """
        import inspect

        import cmaboss
        import maboss
        from maboss.results.baseresult import BaseResult

        assert cmaboss is not None
        assert maboss is not None
        parameters = inspect.signature(BaseResult.plot_trajectory).parameters
        assert "until" in parameters
        assert "axes" in parameters
        """
    )


def test_physicell_settings_runtime_exposes_server_contract() -> None:
    _run_in_clean_interpreter(
        """
        from physicell_config import PhysiCellConfig
        from physicell_config.config.embedded_defaults import (
            get_default_parameters,
        )
        from physicell_config.config.embedded_signals_behaviors import (
            get_behavior_by_name,
            get_expanded_behaviors,
            get_expanded_signals,
            get_signal_by_name,
            get_signals_behaviors,
            update_signals_behaviors_context_from_config,
        )

        config = PhysiCellConfig()
        assert callable(config.generate_xml)
        assert callable(config.load_xml)
        assert callable(get_default_parameters)
        assert callable(get_signals_behaviors)
        assert callable(get_signal_by_name)
        assert callable(get_behavior_by_name)
        assert callable(update_signals_behaviors_context_from_config)
        assert callable(get_expanded_signals)
        assert callable(get_expanded_behaviors)
        """
    )


def test_markdown_and_http_runtime_dependencies_are_installed() -> None:
    _run_in_clean_interpreter(
        """
        import pandas as pd
        import requests
        import tabulate

        assert requests is not None
        assert tabulate is not None
        markdown = pd.DataFrame([{"node": "TP53"}]).to_markdown(index=False)
        assert "TP53" in markdown
        """
    )
