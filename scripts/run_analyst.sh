#!/usr/bin/env bash
# Run Claude (Opus or Fable) as the Weight-Space Programming analyst against a running lab.
#
#   python -m wsp.lab serve --model Qwen/Qwen3-4B --budget 50      # terminal 1
#   scripts/run_analyst.sh                                          # terminal 2
#   scripts/run_analyst.sh "Reduce looping on GSM8K practice" claude-fable-5-2
#
# The analyst gets only the wsp-lab MCP tools: no shell, no file access, no web.
# Env: WSP_LAB_URL (default http://127.0.0.1:8765), MAX_USD (optional spend cap).
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_OBJECTIVE="Improve accuracy on the practice math problems by editing the weights. Follow the method in your instructions."
OBJECTIVE="${1:-$DEFAULT_OBJECTIVE}"
MODEL="${2:-claude-opus-5-5}"
URL="${WSP_LAB_URL:-http://127.0.0.1:8765}"

if ! curl -sf -X POST "$URL/api/budget" -H 'content-type: application/json' -d '{}' >/dev/null; then
  echo "lab not reachable at $URL; start it with: python -m wsp.lab serve" >&2
  exit 1
fi

mkdir -p runs/analyst
LOG="runs/analyst/$(date +%Y%m%d-%H%M%S)-${MODEL}.jsonl"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
MCP_CFG="$TMP/lab.mcp.json"
sed "s#http://127.0.0.1:8765#${URL}#" lab.mcp.json > "$MCP_CFG"

CMD=(claude -p "$OBJECTIVE"
  --model "$MODEL"
  --mcp-config "$MCP_CFG" --strict-mcp-config
  --tools ""
  --allowedTools "mcp__wsp-lab__*"
  --append-system-prompt-file prompts/analyst.md
  --output-format stream-json --verbose)
if [[ -n "${MAX_USD:-}" ]]; then CMD+=(--max-budget-usd "$MAX_USD"); fi

echo "analyst: $MODEL  lab: $URL  log: $LOG" >&2
"${CMD[@]}" | tee "$LOG" | .venv/bin/python scripts/show_stream.py
