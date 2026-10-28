"""Presentation helpers shared by MaBoSS tools and resources."""

import pandas as pd


def clean_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise a DataFrame for safe Markdown rendering.

    Converts all cells to strings, collapses whitespace, removes ``nan``
    literals, and drops entirely empty rows and columns.
    """
    df_str = df.map(
        lambda val: " ".join(str(val).split()),
        na_action="ignore",
    ).fillna("")
    df_str = df_str.replace("nan", "", regex=False)
    df_str = df_str.dropna(axis=1, how="all")
    df_str = df_str.loc[:, (df_str != "").any(axis=0)]
    df_str = df_str.dropna(axis=0, how="all")
    return df_str.loc[(df_str != "").any(axis=1), :]


# Rough engine work (nodes x simulated time x trajectories / threads) above
# which a run usually outlasts a client's tool timeout (some hosts allow 60 s).
HEAVY_WORKLOAD = 10**8


def workload_warning(parameters, node_count: int) -> str:
    """Return a warning line for a heavy simulation workload, or an empty string."""
    try:
        samples = float(parameters["sample_count"])
        max_time = float(parameters["max_time"])
        threads = max(1, int(float(parameters.get("thread_count", 1))))
    except (KeyError, TypeError, ValueError):
        return ""
    if node_count * max_time * samples / threads <= HEAVY_WORKLOAD:
        return ""
    return (
        f"WARNING: heavy workload ({node_count} nodes, max_time={max_time:g}, "
        f"sample_count={samples:g}, thread_count={threads}). run_simulation may "
        "outlast the client's tool timeout (60 s in some hosts). For exploration, "
        "reduce it first, e.g. update_maboss_parameters(parameters="
        '{"sample_count": 1000, "max_time": 100, "thread_count": 4}).'
    )


def cfg_parameters(cfg_text: str) -> dict[str, str]:
    """Read top-level ``name = value;`` simulation parameters from CFG text."""
    import re

    return dict(re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*([^;]+?)\s*;", cfg_text, flags=re.M))
