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
