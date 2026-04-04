# Megaplan

A general-purpose planning and execution harness for LLMs. Megaplan helps any model solve complex tasks through structured phases — prep, plan, critique, gate, execute, and review.

Instead of attempting tasks in one shot, Megaplan gives models a rigorous process: plan the approach, critique it for issues, gate whether to proceed or revise, then execute with verification.

## Features

- **Structured phases**: prep → plan → critique → gate → finalize → execute → review
- **Critique with flags**: Parallel per-check critique that raises typed flags (blocking, significant, minor)
- **Gate enforcement**: LLM-driven gate decides proceed vs iterate, with structured flag resolutions
- **Provider routing**: Support for multiple LLM providers (OpenRouter, Zhipu/GLM, MiniMax, Google Gemini) with API key pooling and automatic failover
- **Robustness levels**: light (no structured critique), standard (4 checks), heavy (8 checks + prep research)
- **Model-agnostic**: Use different models for different phases (e.g. GLM for execution, MiniMax for critique)

## Quick Start

Megaplan uses [Hermes Agent](https://github.com/peteromallet/megaplan-autoimprover) as the execution backend — any model accessible via OpenRouter (or direct provider APIs) works out of the box.

### 1. Install

```bash
pip install megaplan-harness
pip install hermes-agent
```

### 2. Configure API keys

Add your keys to `~/.hermes/.env`:

```bash
# OpenRouter (works with any model)
OPENROUTER_API_KEY=sk-or-v1-...

# Or use direct provider APIs for better performance:
# ZHIPU_API_KEY=...          # for zhipu: prefix (GLM models)
# MINIMAX_API_KEY=...        # for minimax: prefix
# GEMINI_API_KEY=...         # for google: prefix
```

### 3. Run

```bash
megaplan init --project-dir . "Fix the authentication bug in login.py"
megaplan plan --plan <name>
megaplan critique --plan <name>
megaplan gate --plan <name>
megaplan finalize --plan <name>
megaplan execute --plan <name>
```

### Using different models per phase

You can specify models with provider prefixes. Models without a prefix route through OpenRouter:

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

Or use any OpenRouter model for everything:

```json
{
  "models": {
    "prep": "qwen/qwen3.5-27b",
    "plan": "qwen/qwen3.5-27b",
    "critique": "qwen/qwen3.5-27b",
    "execute": "qwen/qwen3.5-27b"
  }
}
```

## SWE-bench Experiment

Megaplan is being used in a live experiment to test whether open-source models can beat Claude Opus 4.5 on [SWE-bench Verified](https://www.swebench.com):

- **Live dashboard**: [peteromallet.github.io/swe-bench-challenge](https://peteromallet.github.io/swe-bench-challenge/)
- **Experiment code**: [megaplan-autoimprover](https://github.com/peteromallet/megaplan-autoimprover)

## License

MIT
