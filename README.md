# Megaplan

A planning and execution harness that helps LLMs solve complex tasks through structured phases — plan, critique, gate, revise, finalize, execute, and review. Instead of one-shot attempts, Megaplan gives any model a rigorous process with independent critique and gating.

## Quick Start — Claude Code / Codex

Copy and give this to your agent:

```
Please install megaplan and set it up for this project:

pip install megaplan-harness
megaplan setup

Once you're done, ask me what I need megaplan for.
```

## Quick Start — Open Models via OpenRouter

Copy and give this to your agent:

```
Please install megaplan with the open-model backend and set it up:

pip install megaplan-harness hermes-agent

Then create ~/.hermes/.env with:
OPENROUTER_API_KEY=<my key>

Then run: megaplan setup

Once you're done, ask me what I need megaplan for.
```

Get an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys). Any model on OpenRouter works — Qwen, Llama, Mistral, DeepSeek, etc.

---

## How it works

```
plan → critique → gate → [revise → critique → gate]* → finalize → execute → review
```

Each phase can use a different model. The critique phase uses an independent model to review the plan and raise flags. The gate decides whether to proceed or iterate. This prevents models from rubber-stamping their own work. Planning now goes through a visible `prep` phase so repository investigation is observable instead of hidden inside `plan`.

## Running manually

```bash
megaplan init --project-dir . "Fix the authentication bug in login.py"
megaplan plan --plan <name>
megaplan critique --plan <name>
megaplan gate --plan <name>
megaplan finalize --plan <name>
megaplan execute --plan <name>
```

## Using different models per phase

Models with provider prefixes route to direct APIs. Models without a prefix go through OpenRouter:

```json
{
  "models": {
    "prep": "zhipu:glm-5.1",
    "plan": "zhipu:glm-5.1",
    "critique": "minimax:MiniMax-M2.7-highspeed",
    "execute": "zhipu:glm-5.1",
    "review": "minimax:MiniMax-M2.7-highspeed"
  }
}
```

Configure direct provider keys in `~/.hermes/.env`:

```bash
ZHIPU_API_KEY=...          # for zhipu: prefix
MINIMAX_API_KEY=...        # for minimax: prefix
GEMINI_API_KEY=...         # for google: prefix
```

## Robustness levels

- **light** — visible `prep` + one critique/revise pass, no gate or review
- **standard** — visible `prep` + 4 critique checks (default)
- **robust** — visible `prep` + 8 critique checks + parallel critique
- **superrobust** — same as robust + parallel review

## Observability

```bash
megaplan status --plan <name>
```

`status` is the single monitoring command. It exposes lifecycle fields such as `active_step`, `last_step`, notes, cost, execute progress, and next-step runtime guidance.
`watch` remains as a backward-compatible alias to `status`.

## Subagent mode (Claude Code / Codex)

Subagent mode delegates the full workflow to an autonomous agent, returning control only at defined breakpoints. It is the default orchestration mode for Claude Code and Codex. Cursor continues to run inline.

```bash
megaplan config set orchestration.mode subagent   # default
megaplan config set orchestration.mode inline      # switch back
```

## Configuration & Defaults

View all settings with `megaplan config show`. Override with `megaplan config set <key> <value>`. Reset with `megaplan config reset`.

| Key | Default | Description |
|-----|---------|-------------|
| `orchestration.mode` | `subagent` | `inline` or `subagent` (Claude Code and Codex) |
| `orchestration.max_critique_concurrency` | `2` | Max parallel critique checks |
| `execution.worker_timeout_seconds` | `7200` | Worker process timeout (seconds) |
| `execution.max_execute_no_progress` | `3` | No-progress execute attempts before escalation |
| `execution.max_review_rework_cycles` | `3` | Review→rework loops before force-proceeding |
| `agents.<step>` | varies | Agent for each phase (`claude`, `codex`, `hermes`) |

```bash
megaplan config set execution.worker_timeout_seconds 3600
megaplan config set agents.critique hermes
megaplan config reset
```

## SWE-bench Experiment

Megaplan is being tested live against Claude 4.5 Opus on SWE-bench Verified:

- **[Live dashboard](https://peteromallet.github.io/swe-bench-challenge/)** — watch the experiment in real time
- **[hermes-megaplan](https://github.com/peteromallet/hermes-megaplan)** — experiment orchestration code

## Code Health

<img src="scorecard.png" width="100%">

## License

MIT
