# Persistent User Avatar — automatic capture

The avatar is a cross-session, cross-project model of the user, stored as
slot-keyed memories in a dedicated `avatar` collection (one collection,
partitioned by `avatar_id`, default `"default"`; multiple avatars supported).

Enable it in `config.toml`:

```toml
[avatar]
enabled = true
allow_freeform = false          # curated slot vocabulary only (recommended)
auto_derive = true              # extract slots from each session at session_end
redact_patterns = []            # extra literal PII substrings to mask (email/long-numbers always masked)
allow_remote = false            # default: local-first (embedded DB only)
```

> **Privacy:** the avatar is **local-first by default**. Config validation
> **fails fast** if `[avatar].enabled` is set while `[database].mode = "remote"`
> *unless* you set `allow_remote = true` — so a persistent model of a real
> person never leaves the device unless you explicitly opt in. Set
> `allow_remote = true` to persist the avatar to a trusted remote LatticeDB
> (e.g. a shared LAN server), accepting that the user model then leaves the
> device. Values are PII-redacted before write (built-in email + long-digit
> masking, plus your `redact_patterns`) regardless of where they are stored.
> Use `forget` to delete a slot; a changed value records the previous one in
> `prior_value` for a one-step reversible audit.

## Why "automatic" has three layers

An MCP server only sees the tool calls the model makes over stdio — it
**cannot observe your messages directly**, and Claude Code hooks run shell
commands that **cannot call MCP tools**. So truly-zero-effort capture isn't
possible from the server alone. Instead, capture is reliable through three
cooperating layers, in increasing automation:

1. **Agent-nudge (built in, always on when enabled).** The server's
   instructions and the `avatar_store` / `avatar_search` tool descriptions
   tell the model to record durable user facts as it learns them and to read
   the avatar at task start. No setup required.

2. **Session-end auto-derive (built in, `auto_derive = true`).** At
   `session_end`, the server scans this session's episodic memories and
   extracts explicit, high-precision statements (e.g. "my preferred language
   is Rust", "timezone: America/New_York") into avatar slots — no LLM needed.
   This sharpens the avatar every session with zero per-turn effort.

3. **Settings.json hooks (this doc, opt-in).** A `UserPromptSubmit` hook
   injects a per-turn reminder so the model reliably captures durable facts
   and calls `session_end` at the end — turning layers 1–2 from "the model
   might" into "the model is prompted every turn".

## Ready-to-install hooks

Add to `~/.claude/settings.json` (or the project `.claude/settings.json`).
This injects a lightweight standing reminder on every prompt; the model still
performs the `avatar_store` call (hooks cannot call MCP tools themselves):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "printf '%s' 'If this message reveals a durable user preference, habit, working style, tool, or trait, call the lattice avatar_store tool to record it. At the start of a new task, call avatar_search to personalize. At conversation end, call session_end.'"
          }
        ]
      }
    ]
  }
}
```

The `command` writes the reminder to stdout; Claude Code injects
`UserPromptSubmit` stdout into the model's context. Adjust the wording freely.

> Tip: you can ask Claude Code to install this for you via the
> `update-config` skill ("add a UserPromptSubmit hook that …").

## Manual control

- `avatar_store { slot_key, value, avatar_id?, provenance? }` — record/reinforce.
- `avatar_search { query, avatar_id?, limit? }` — personalized recall.
- `avatar_get { avatar_id? }` — dump all slots (most-confident first).
- `forget { category: "avatar", memory_id }` — delete a slot.

Curated slots include: `preferred_language`, `coding_style`,
`communication_tone`, `tooling`, `timezone`, `expertise`, `domain_focus`,
`workflow_preference`, `naming_convention`, `testing_preference`, `editor`,
`os_platform`, `review_preference`, `goal`. Set `allow_freeform = true` to
permit others.
