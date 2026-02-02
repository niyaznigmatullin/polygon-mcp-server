# Polygon MCP Server

MCP server for Polygon API (Codeforces Polygon).

## Requirements

- Python 3.10+
- `polygon_api==1.1.0a1`

## Install

### From PyPI repository

```bash
pip install polygon-mcp-server
```

With uv:

```bash
uv pip install polygon-mcp-server
```

### From the sources

Install the package (adds the `polygon-mcp` CLI):

```bash
pip install .
```

With uv:

```bash
uv pip install .
```

Or install the CLI tool into uv's tool environment:

```bash
uv tool install .
```

## Run

Set credentials:

- `POLYGON_API_KEY`
- `POLYGON_API_SECRET`
- Optional: `POLYGON_API_URL`
- Optional: `POLYGON_MCP_CONFIG` to load stored credentials

Then start:

```bash
polygon-mcp
```

## Logging

Logs are written to `~/.local/state/polygon-mcp/polygon-mcp.log` (or
`$XDG_STATE_HOME/polygon-mcp/polygon-mcp.log` if set). Override with
`POLYGON_MCP_LOG_FILE`.

## File output safety

Tools that write to disk only allow paths within:

- the current project directory
- `/tmp`
- any extra roots in `POLYGON_MCP_OUTPUT_ROOTS` (colon-separated)
