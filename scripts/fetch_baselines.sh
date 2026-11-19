#!/usr/bin/env bash
# Clone the official baseline code at the exact commits used in the paper (no modification).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
clone() {  # name url commit
  if [ ! -d "external/$1/.git" ]; then git clone "$2" "external/$1"; fi
  git -C "external/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "external/$1" checkout --quiet "$3"
  echo "$1 at $(git -C "external/$1" rev-parse HEAD)"
}
clone SOFTS https://github.com/Secilia-Cxy/SOFTS.git f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44
clone TimeFilter https://github.com/TROUBADOUR000/TimeFilter.git dffde87e4fff0fdeeebbacde03dc1e432e15b3a1
clone iTransformer https://github.com/thuml/iTransformer.git c2426e68ca13f74aaec08045c5c724d8ad328124
clone CRIB https://github.com/Muyiiiii/CRIB.git a457672c7b0152f74c929858dba2a9c886405519
python - <<'PY'
import sys; sys.path.insert(0, '.')
from otarf.baselines import verify_official_source
for name in ('SOFTS', 'TimeFilter', 'iTransformer', 'CRIB'):
    verify_official_source(f'external/{name}', name, 'configs/baselines.lock.json')
print('Official SOFTS / TimeFilter / iTransformer / CRIB sources match the recorded SHA-256 of every Python file.')
PY
