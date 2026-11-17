#!/usr/bin/env bash
# Pushes this folder to https://github.com/iitrky/pilotless-star-ris-isac (needs: gh auth login, or git credentials).
set -e
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/iitrky/pilotless-star-ris-isac.git
git push origin main
echo "Done: https://github.com/iitrky/pilotless-star-ris-isac"
