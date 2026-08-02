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

test -f "$repo_root/projects/hotpotqa-hybrid-format-ablation/index.html"
test -f "$repo_root/projects/hotpotqa-hybrid-format-ablation/style.css"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-hybrid-format-ablation/' "$repo_root/sitemap.xml"

test -f "$repo_root/projects/hotpotqa-experiments/index.html"
test -f "$repo_root/projects/hotpotqa-experiments/style.css"
test -f "$repo_root/projects/hotpotqa-grpo-signal-gate/index.html"
test -f "$repo_root/projects/hotpotqa-grpo-signal-gate/style.css"
rg -q 'href="/projects/hotpotqa-experiments/"' "$repo_root/index.html"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-experiments/' "$repo_root/sitemap.xml"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-grpo-signal-gate/' "$repo_root/sitemap.xml"

hotpotqa_pages=(
  agent-r1-architecture
  agent-r1-grpo-guide
  hotpotqa-evidence-agent
  hotpotqa-experiments
  hotpotqa-format-valid-debug
  hotpotqa-reward-schema-debug
  hotpotqa-hybrid-format-ablation
  hotpotqa-bridge-trace-audit
  hotpotqa-gold-evidence-oracle
  hotpotqa-gold-sentence-oracle
  hotpotqa-persistent-failures
  hotpotqa-grpo-signal-gate
)

for page in "${hotpotqa_pages[@]}"; do
  file="$repo_root/projects/$page/index.html"
  test -f "$file"
  nav_html="$(sed -n '/<nav class="project-nav"/,/<\/nav>/p' "$file")"
  rg -q 'href="/projects/hotpotqa-experiments/"' <<<"$nav_html"
  test "$(rg -c '<a ' <<<"$nav_html")" -eq 4
done

echo "Public links do not reference the Hugo development server."
