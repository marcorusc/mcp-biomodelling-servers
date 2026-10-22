"""MCP servers for biological modelling."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcp-biomodelling-servers")
except PackageNotFoundError:
    # Keep source-checkout execution aligned with the project metadata.
    __version__ = "2.5.1"

__all__ = ["__version__"]
