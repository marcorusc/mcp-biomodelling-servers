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
        from neko.core.strategy_options import ConnectionStrategyMigrationWarning
        from neko.core.tools import is_connected
        from neko.inputs import Universe, signor
        import pandas as pd
        import warnings
        from pathlib import Path
        from tempfile import TemporaryDirectory

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

        export_network = Network(
            initial_nodes=["P04637", "P38398"],
            resources=resources,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConnectionStrategyMigrationWarning)
            export_network.complete_connection(
                maxlen=1,
                path_policy="one_shortest",
                reuse_policy="discovered_paths",
            )
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "runtime"
            Exports(export_network).export_bnet(str(prefix))
            generated_bnets = list(Path(temporary_directory).glob("*.bnet"))
            assert len(generated_bnets) == 1
            assert generated_bnets[0].name.startswith("runtime")
        """
    )


def test_maboss_runtime_exposes_plotting_contract_and_compiled_engine() -> None:
    _run_in_clean_interpreter(
        """
        import inspect

        import cmaboss
        import maboss
        from maboss.results.baseresult import BaseResult
        from pathlib import Path
        from tempfile import TemporaryDirectory

        assert cmaboss is not None
        assert maboss is not None
        parameters = inspect.signature(BaseResult.plot_trajectory).parameters
        assert "until" in parameters
        assert "axes" in parameters

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bnet_path = root / "runtime.bnet"
            bnd_path = root / "runtime.bnd"
            cfg_path = root / "runtime.cfg"
            bnet_path.write_text("A, B\\nB, A\\n", encoding="utf-8")
            maboss.bnet_to_bnd_and_cfg(
                str(bnet_path),
                str(bnd_path),
                str(cfg_path),
            )
            assert bnd_path.is_file() and bnd_path.stat().st_size > 0
            assert cfg_path.is_file() and cfg_path.stat().st_size > 0
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
        from copy import deepcopy

        config = PhysiCellConfig()
        assert callable(config.generate_xml)
        assert callable(config.load_xml)
        config.substrates.add_substrate("oxygen")
        config.cell_types.add_cell_type("tumour")
        config.cell_types.set_volume_parameters(
            "tumour",
            total=3000,
            nuclear=700,
            fluid_fraction=0.6,
        )
        config.cell_types.set_motility(
            "tumour",
            speed=2,
            persistence_time=9,
            enabled=False,
        )
        config.cell_types.set_death_rate("tumour", "apoptosis", 0.001)
        config.cell_types.set_death_rate("tumour", "necrosis", 0.002)
        config.cell_types.add_secretion(
            "tumour",
            "oxygen",
            secretion_rate=2,
            secretion_target=3,
            uptake_rate=4,
            net_export_rate=5,
        )
        patch_candidate = deepcopy(config)
        patch_candidate.cell_types.set_volume_parameters(
            "tumour",
            total=None,
            nuclear=800,
            fluid_fraction=None,
        )
        patch_candidate.cell_types.set_motility(
            "tumour",
            speed=3,
            persistence_time=None,
            enabled=None,
        )
        patched_phenotype = patch_candidate.cell_types.get_cell_types()[
            "tumour"
        ]["phenotype"]
        assert patched_phenotype["volume"]["total"] == 3000
        assert patched_phenotype["volume"]["nuclear"] == 800
        assert patched_phenotype["volume"]["fluid_fraction"] == 0.6
        assert patched_phenotype["motility"]["speed"] == 3
        assert patched_phenotype["motility"]["persistence_time"] == 9
        assert patched_phenotype["motility"]["enabled"] is False
        assert patched_phenotype["death"]["apoptosis"]["default_rate"] == 0.001
        assert patched_phenotype["death"]["necrosis"]["default_rate"] == 0.002
        assert patched_phenotype["secretion"]["oxygen"] == {
            "secretion_rate": 2,
            "secretion_target": 3,
            "uptake_rate": 4,
            "net_export_rate": 5,
        }
        from PhysiCell import server as physicell_server

        session_id = physicell_server.session_manager.create_session()
        session = physicell_server.session_manager.get_session(session_id)
        assert session is not None
        session.config = deepcopy(config)
        physicell_server.configure_cell_parameters(
            cell_type="tumour",
            volume_total=None,
            volume_nuclear=None,
            fluid_fraction=None,
            motility_speed=None,
            persistence_time=None,
            motility_enabled=None,
            apoptosis_rate=0.003,
            necrosis_rate=None,
            session_id=session_id,
        )
        server_phenotype = session.config.cell_types.get_cell_types()[
            "tumour"
        ]["phenotype"]
        assert server_phenotype["volume"]["total"] == 3000
        assert server_phenotype["motility"]["enabled"] is False
        assert (
            server_phenotype["death"]["apoptosis"]["default_rate"]
            == 0.003
        )
        assert (
            server_phenotype["death"]["necrosis"]["default_rate"]
            == 0.002
        )
        candidate = config.copy()
        candidate.physiboss.add_intracellular_model(
            cell_type_name="tumour",
            model_type="maboss",
            bnd_filename="/tmp/runtime.bnd",
            cfg_filename="/tmp/runtime.cfg",
        )
        original_cell = config.cell_types.get_cell_types()["tumour"]
        candidate_cell = candidate.cell_types.get_cell_types()["tumour"]
        assert "intracellular" not in original_cell["phenotype"]
        intracellular = candidate_cell["phenotype"]["intracellular"]
        assert intracellular["type"] == "maboss"
        assert intracellular["bnd_filename"] == "/tmp/runtime.bnd"
        assert intracellular["cfg_filename"] == "/tmp/runtime.cfg"
        candidate.physiboss.set_intracellular_settings(
            cell_type_name="tumour",
            intracellular_dt=6,
            time_stochasticity=2,
            scaling=1.5,
            start_time=4,
            inheritance_global=True,
        )
        candidate.physiboss.set_intracellular_settings(
            cell_type_name="tumour",
            intracellular_dt=12,
            time_stochasticity=None,
            scaling=None,
            start_time=None,
            inheritance_global=None,
        )
        settings = candidate.cell_types.get_cell_types()["tumour"][
            "phenotype"
        ]["intracellular"]["settings"]
        assert settings == {
            "intracellular_dt": 12,
            "time_stochasticity": 2,
            "scaling": 1.5,
            "start_time": 4,
            "inheritance": {"global": True},
        }
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
