# MCP Bio‑Modelling MCP Servers

This repository centralizes **Model Context Protocol (MCP)** servers that wrap Python‑based *mechanistic / systems biology* modelling tools. Each subfolder contains a `server.py` entrypoint plus a README describing the specific tool interface.

Current servers (see their own READMEs & upstream docs):

| Tool | Folder | Upstream Documentation |
|------|--------|------------------------|
| MaBoSS | `MaBoSS/` | https://github.com/colomoto/pyMaBoSS |
| NeKo | `NeKo/` | https://github.com/sysbio-curie/Neko |
| PhysiCell (settings wrapper) | `PhysiCell/` | https://github.com/marcorusc/PhysiCell_Settings |

All servers are Python processes speaking MCP over stdio.

---
## MCP Background
The Model Context Protocol standardizes how external tools expose *tools* and *resources* to AI assistants / IDEs. Spec & introduction: https://modelcontextprotocol.io/docs/getting-started/intro

Each `server.py` advertises modelling actions (e.g. run simulations, manage sessions) to any MCP‑aware client (e.g. VS Code with GitHub Copilot Chat MCP support).

---
## Repository Layout
```
MaBoSS/    # MaBoSS MCP server (Boolean / stochastic models)
NeKo/      # NeKo MCP server
PhysiCell/ # PhysiCell settings / sessions MCP server
README.md
```
Consult the README within each tool folder for: purpose, required Python packages, and any model/data file expectations. Installation instructions for the modelling tools themselves live *there* (or in the upstream project links above) — they are intentionally not duplicated here.

---
## Environment Assumption
All tools are Python‑based. Create (and manage) a single Conda environment that contains the dependencies for MaBoSS, NeKo, and PhysiCell. The exact creation commands are up to you (not prescribed here). Once created, note the absolute path to its Python interpreter (e.g. `/home/you/miniforge3/envs/mcp_modelling/bin/python`).

---
## Configure in VS Code (GitHub Copilot Chat / MCP)
1. Clone this repo somewhere stable (no spaces in path recommended).
2. Open VS Code.
3. Ensure the Copilot Chat (or other MCP‑capable) extension is installed and updated.
4. Press `Ctrl + Shift + P` and search for the command that opens the MCP configuration JSON (e.g. “MCP: Open Configuration” or locate `mcp.json`). On Linux it typically lives at: `~/.config/Code/User/mcp.json`.
5. Add entries pointing to each `server.py` using the Conda environment’s Python.

Example (adapt paths to your system; based on the working setup):

```jsonc
{
  "servers": {
    "maboss": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/MaBoSS/server.py"
      ],
      "env": {
        "PATH": "/home/you/miniforge3/envs/mcp_modelling/bin:${Path}",
        "CONDA_PREFIX": "/home/you/miniforge3/envs/mcp_modelling"
      }
    },
    "neko": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/NeKo/server.py"
      ]
    },
    "physicell": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/PhysiCell/server.py"
      ]
    }
  },
  "inputs": [
    {
      "id": "memory_file_path",
      "type": "promptString",
      "description": "Path to the memory storage file",
      "password": false
    }
  ]
}
```

Notes:
- Replace `/home/you/...` and `/absolute/path/to/...` with your actual directories.
- Keep all three servers referencing the *same* Conda interpreter to share installed libraries.
- Add further environment variables (e.g. data directories) per tool README if required.

After saving, reload / restart VS Code so the MCP client reconnects.

Activation / usage guidance in VS Code: https://code.visualstudio.com/docs/copilot/chat/mcp-servers

You should then see the servers’ tools listed in the Copilot Chat “/tools” (or similar) UI. Invoke them by name with required parameters.

---
## Adding Another Server
1. Create a new folder with `server.py` and a README describing the underlying modelling tool and dependencies.
2. Follow existing server structure for registering MCP tools.
3. Update your `mcp.json` with a new block (use the same Conda Python path).
4. Document any additional env vars in that folder README.

---
## License
Project is MIT (see existing LICENSE file). Underlying tools retain their own licenses — consult upstream repositories.

---
## Quick Reference
| Action | What to Do |
|--------|-----------|
| Get tool install steps | Open the tool’s subfolder README or upstream link |
| Ensure deps present | Install into your chosen Conda env (user‑defined) |
| Configure MCP | Edit `~/.config/Code/User/mcp.json` as above |
| Reload servers | Reload VS Code window |
| Learn MCP | Spec: modelcontextprotocol.io; VS Code guide link above |

---
Happy modelling!
