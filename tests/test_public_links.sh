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
rg -q 'href="/projects/hotpotqa-hybrid-format-ablation/"' "$repo_root/index.html"
rg -q 'https://thefoolgy.github.io/projects/hotpotqa-hybrid-format-ablation/' "$repo_root/sitemap.xml"

echo "Public links do not reference the Hugo development server."
