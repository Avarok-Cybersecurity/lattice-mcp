# Lattice-MCP Setup

## 1. Configuration

```bash
mkdir -p ~/.lattice-mcp
cp config.example.toml ~/.lattice-mcp/config.toml
# Edit ~/.lattice-mcp/config.toml — set your embedding endpoint and API key env var
```

Ensure the API key environment variable is set (default: `OPENAI_API_KEY`).

## 2. Build

```bash
cargo build --release
```

Binary at: `target/release/lattice-mcp`

## 3. Register MCP Server

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "lattice-memory": {
      "command": "/Users/nologik/avarok/lattice-mcp/target/release/lattice-mcp",
      "args": ["--config", "~/.lattice-mcp/config.toml"]
    }
  }
}
```

## 4. Skill

The skill at `~/.claude/skills/lattice-memory/skill.md` instructs Claude to proactively use the memory system during conversations.

## 5. Verify

After restarting Claude Code, run `/mcp` to confirm `lattice-memory` appears with 11 tools.

## 6. Remote Mode (Optional)

To connect to a running LatticeDB instance instead of embedded mode:

```toml
[database]
mode = "remote"
remote_url = "http://192.168.68.61:6334"
```
