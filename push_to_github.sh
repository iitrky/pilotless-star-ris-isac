#!/usr/bin/env bash
# Creates https://github.com/iitrky/pilotless-star-ris-isac and pushes this folder.
# Needs the GitHub CLI logged in as iitrky:  gh auth login
set -e
git init -b main 2>/dev/null || true
git add .
git commit -m "Code for: Belief State Reinforcement Learning for Pilotless Intelligent Surface Control" || true
gh repo create iitrky/pilotless-star-ris-isac --public --source . --remote origin --push \
  --description "Pilotless STAR-RIS ISAC control with belief-state hierarchical-attention TD3 (HA-TD3)"
echo "Done: https://github.com/iitrky/pilotless-star-ris-isac"
