#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if rg -n 'https?://localhost:1313|livereload\.js' \
  --glob '*.html' \
  --glob '*.xml' \
  "$repo_root"; then
  echo "Published files must not contain Hugo development-server URLs or livereload scripts." >&2
  exit 1
fi

test -f "$repo_root/projects/agent-r1-architecture/index.html"
test -f "$repo_root/projects/agent-r1-architecture/style.css"
test -f "$repo_root/projects/agent-r1-architecture/assets/framework.png"
rg -q 'href="/projects/agent-r1-architecture/"' "$repo_root/index.html"
rg -q 'https://thefoolgy.github.io/projects/agent-r1-architecture/' "$repo_root/sitemap.xml"

test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/index.html"
test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/style.css"
test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/app.js"
test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/fit-source.py"
test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/fit-source.provenance.json"
test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/dynamic-helper-source.py"
test -f "$repo_root/projects/agent-r1-ray-trainer-walkthrough/dynamic-helper-source.provenance.json"
rg -q 'Dynamic Sampling overlay' "$repo_root/projects/agent-r1-ray-trainer-walkthrough/index.html"
rg -q 'data-jump-event="filter-1"' "$repo_root/projects/agent-r1-ray-trainer-walkthrough/index.html"
rg -q 'filter_informative_prompt_groups' "$repo_root/projects/agent-r1-ray-trainer-walkthrough/app.js"
rg -q 'https://thefoolgy.github.io/projects/agent-r1-ray-trainer-walkthrough/' "$repo_root/sitemap.xml"

test -f "$repo_root/projects/hotpotqa-hybrid-format-ablation/index.html"
test -f "$repo_root/projects/hotpotqa-hybrid-format-ablation/style.css"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-hybrid-format-ablation/' "$repo_root/sitemap.xml"

test -f "$repo_root/projects/hotpotqa-recovery-sft-pipeline/index.html"
test -f "$repo_root/projects/hotpotqa-recovery-sft-pipeline/style.css"
rg -q 'href="/projects/hotpotqa-recovery-sft-pipeline/"' "$repo_root/projects/hotpotqa-experiments/index.html"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-recovery-sft-pipeline/' "$repo_root/sitemap.xml"
rg -q 'build_recovery_sft.py:632-710' "$repo_root/projects/hotpotqa-recovery-sft-pipeline/index.html"

test -f "$repo_root/projects/hotpotqa-experiments/index.html"
test -f "$repo_root/projects/hotpotqa-experiments/style.css"
test -f "$repo_root/projects/hotpotqa-grpo-signal-gate/index.html"
test -f "$repo_root/projects/hotpotqa-grpo-signal-gate/style.css"
test -f "$repo_root/projects/hotpotqa-evidence-state-grpo-step100/index.html"
test -f "$repo_root/projects/hotpotqa-evidence-state-grpo-step100/style.css"
rg -q 'href="/projects/hotpotqa-experiments/"' "$repo_root/index.html"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-experiments/' "$repo_root/sitemap.xml"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-grpo-signal-gate/' "$repo_root/sitemap.xml"
rg -q 'href="/projects/hotpotqa-evidence-state-grpo-step100/"' "$repo_root/projects/hotpotqa-experiments/index.html"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-evidence-state-grpo-step100/' "$repo_root/sitemap.xml"
rg -q 'id="2026-08-03"' "$repo_root/projects/hotpotqa-evidence-agent/index.html"
rg -q 'href="/projects/hotpotqa-evidence-agent/#2026-08-03"' "$repo_root/projects/hotpotqa-experiments/index.html"
rg -q 'GRPO-DS' "$repo_root/projects/hotpotqa-evidence-agent/index.html"

hotpotqa_pages=(
  agent-r1-architecture
  agent-r1-ray-trainer-walkthrough
  agent-r1-grpo-guide
  hotpotqa-evidence-agent
  hotpotqa-experiments
  hotpotqa-format-valid-debug
  hotpotqa-reward-schema-debug
  hotpotqa-hybrid-format-ablation
  hotpotqa-recovery-sft-pipeline
  hotpotqa-bridge-trace-audit
  hotpotqa-gold-evidence-oracle
  hotpotqa-gold-sentence-oracle
  hotpotqa-persistent-failures
  hotpotqa-grpo-signal-gate
  hotpotqa-evidence-state-sft
  hotpotqa-evidence-state-grpo-step100
)

for page in "${hotpotqa_pages[@]}"; do
  file="$repo_root/projects/$page/index.html"
  test -f "$file"
  nav_html="$(sed -n '/<nav class="project-nav"/,/<\/nav>/p' "$file")"
  rg -q 'href="/projects/hotpotqa-experiments/"' <<<"$nav_html"
  rg -q 'href="/projects/agent-r1-ray-trainer-walkthrough/"' <<<"$nav_html"
  test "$(rg -c '<a ' <<<"$nav_html")" -eq 5
done

echo "Public links do not reference the Hugo development server."
