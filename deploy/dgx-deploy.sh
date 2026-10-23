#!/usr/bin/env bash
#
# dgx-deploy.sh — idempotent build & install of lattice-mcp on the DGX from git.
#
# Clones (or fast-forwards) lattice-mcp and its path-dependency lattice-db as
# SIBLINGS so Cargo's `../lattice-db` path deps resolve, builds the release
# binary, installs it, re-asserts the `claude` ACLs, and smoke-tests the new
# binary against the live config.
#
# Run ON the DGX. Prerequisites for whichever user runs this:
#   - git, cargo/rustc, setfacl
#   - GitHub *read* auth for the private Avarok repos (token or deploy key)
#   - the lattice config already present at $CONFIG (deployed out of band)
#
# Re-run any time to update: it fast-forwards both repos to their remote tip
# and rebuilds incrementally (target/ is intentionally preserved).

set -euo pipefail

# ---- authoritative configuration (SSOT; edit here to change) ----------------
SRC_BASE="/workspace/avarok"
MCP_REPO="https://github.com/Avarok-Cybersecurity/lattice-mcp.git"
DB_REPO="https://github.com/Avarok-Cybersecurity/lattice-db.git"
MCP_BRANCH="master"
DB_BRANCH="acid"                       # lattice-mcp's path dep tracks this branch
INSTALL_DIR="/workspace/.lattice-mcp"
BIN_DST="${INSTALL_DIR}/bin/lattice-mcp"
CONFIG="${INSTALL_DIR}/config.toml"
RUNTIME_USER="claude"                  # user the MCP server runs as
# -----------------------------------------------------------------------------

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy:WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[deploy:FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- fail fast on prerequisites ---------------------------------------------
command -v git     >/dev/null 2>&1 || die "git not found"
command -v setfacl >/dev/null 2>&1 || die "setfacl not found (install the 'acl' package)"
CARGO="$(command -v cargo 2>/dev/null || true)"
[ -n "$CARGO" ] || { [ -x /usr/local/bin/cargo ] && CARGO=/usr/local/bin/cargo; }
[ -n "$CARGO" ] || die "cargo not found in PATH or /usr/local/bin"
log "cargo: $CARGO ($("$CARGO" --version))"

# ---- clone-or-fast-forward, pinned to the expected branch -------------------
sync_repo() {
  local url="$1" dir="$2" branch="$3"
  if [ -d "$dir/.git" ]; then
    log "updating $(basename "$dir") -> origin/$branch"
    git -C "$dir" fetch --quiet origin "$branch"
    git -C "$dir" checkout --quiet "$branch"
    git -C "$dir" reset --hard --quiet "origin/$branch"   # deploy mirror: discard local edits
  else
    log "cloning $(basename "$dir") ($branch)"
    git clone --quiet -b "$branch" "$url" "$dir"
  fi
  log "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD) ($branch)"
}

mkdir -p "$SRC_BASE"
sync_repo "$MCP_REPO" "$SRC_BASE/lattice-mcp" "$MCP_BRANCH"
sync_repo "$DB_REPO"  "$SRC_BASE/lattice-db"  "$DB_BRANCH"

# path-dep sanity: lattice-mcp/Cargo.toml expects ../lattice-db/crates/{core,storage}
for c in lattice-core lattice-storage; do
  [ -f "$SRC_BASE/lattice-db/crates/$c/Cargo.toml" ] \
    || die "missing $SRC_BASE/lattice-db/crates/$c — siblings not laid out correctly"
done

# ---- build ------------------------------------------------------------------
log "building lattice-mcp (release; LTO=fat — slow on a cold cache)"
( cd "$SRC_BASE/lattice-mcp" && "$CARGO" build --release )
BUILT="$SRC_BASE/lattice-mcp/target/release/lattice-mcp"
[ -x "$BUILT" ] || die "build did not produce $BUILT"

# ---- install ----------------------------------------------------------------
install -D -m 755 "$BUILT" "$BIN_DST"
log "installed -> $BIN_DST ($(du -h "$BIN_DST" | cut -f1))"

# ---- re-assert runtime-user access ------------------------------------------
# Explicit ACL works when run by the dir owner; otherwise the persistent
# access ACL set at first deploy still stands. Either way we verify below.
if setfacl -R   -m "u:${RUNTIME_USER}:rwx" "$INSTALL_DIR" 2>/dev/null \
&& setfacl -R -d -m "u:${RUNTIME_USER}:rwx" "$INSTALL_DIR" 2>/dev/null; then
  log "ACLs re-asserted for '$RUNTIME_USER' on $INSTALL_DIR"
else
  warn "could not setfacl (not '$INSTALL_DIR' owner?) — relying on existing ACL"
fi
getfacl -p "$INSTALL_DIR" 2>/dev/null | grep -q "user:${RUNTIME_USER}:rwx" \
  || die "'$RUNTIME_USER' lacks rwx on $INSTALL_DIR (needed to create the embedded-DB data dir)"

# ---- verify: boot new binary vs live config, no stray /workspace data dir ----
[ -f "$CONFIG" ] || die "expected config at $CONFIG — deploy it before running this"
SMOKE="$(mktemp -d)"; trap 'rm -rf "$SMOKE"' EXIT
set +e
HOME="$SMOKE" timeout 12 "$BIN_DST" --config "$CONFIG" </dev/null >"$SMOKE/out" 2>"$SMOKE/err"
set -e
if grep -q "Memory system initialized, starting MCP server" "$SMOKE/err"; then
  log "smoke OK: config parsed, embedded DB initialized, MCP server reached"
else
  echo "---- smoke stderr ----" >&2; cat "$SMOKE/err" >&2
  die "smoke test failed — new binary did not boot cleanly with $CONFIG"
fi

log "DONE. Restart Claude Code on the DGX; '/mcp' should list lattice-memory."
