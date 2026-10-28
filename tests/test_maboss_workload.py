"""Heavy default workloads are flagged before run_simulation can time out."""

import pytest

from MaBoSS.services.formatting import cfg_parameters, workload_warning
from tests.test_maboss_mcp_errors import _call_tool, _run


def test_threshold_counts_nodes_time_samples_and_threads() -> None:
    heavy = {"sample_count": 10000, "max_time": 1000, "thread_count": 1}
    assert "heavy workload (55 nodes" in workload_warning(heavy, 55)
    assert 'update_maboss_parameters(parameters={"sample_count"' in workload_warning(heavy, 55)
    assert workload_warning(heavy, 3) == ""
    assert workload_warning({**heavy, "thread_count": 8}, 55) == ""
    assert workload_warning({"sample_count": "bad"}, 55) == ""


def test_cfg_parameters_reads_top_level_assignments() -> None:
    text = "time_tick = 0.5;\nmax_time = 1000;\nsample_count = 10000;\nA.istate = 1;\n"
    assert cfg_parameters(text) == {"time_tick": "0.5", "max_time": "1000", "sample_count": "10000"}


def _bnet(count: int) -> str:
    nodes = [f"N{i}" for i in range(count)]
    rules = [f"{node}, {nodes[i - 1]}" for i, node in enumerate(nodes)]
    return "targets, factors\n" + "\n".join(rules) + "\n"


@pytest.mark.parametrize(("count", "warned"), [(60, True), (3, False)])
def test_conversion_and_build_warn_only_for_heavy_defaults(
    monkeypatch: pytest.MonkeyPatch, count: int, warned: bool,
) -> None:
    monkeypatch.setenv("PATH", "")
    created = _run(_call_tool("create_session", {}))
    session_id = created.structured_content["result"].split()[2]
    converted = _run(_call_tool("bnet_to_bnd_and_cfg", {
        "session_id": session_id, "bnet_text": _bnet(count),
    }))
    assert converted.is_error is False, converted.content[0].text
    assert ("heavy workload" in converted.content[0].text) is warned
    built = _run(_call_tool("build_simulation", {"session_id": session_id}))
    assert built.is_error is False, built.content[0].text
    assert ("heavy workload" in built.content[0].text) is warned
