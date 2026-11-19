#!/usr/bin/env bash
# Download the three public datasets into data/raw and check their SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
get() {  # url, file, sha256
  [ -f "data/raw/$2" ] || curl -L --fail -o "data/raw/$2" "$1"
  echo "$3  data/raw/$2" | shasum -a 256 -c -
}
get "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip" beijing_multisite.zip \
    b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4
get "https://zenodo.org/records/4656719/files/kdd_cup_2018_dataset_with_missing_values.zip?download=1" kdd2018.zip \
    3d0d2dbec3621b20c7a8ae6e6f00a63bfaebb962dca34d9f60f0ddbbde03e9bd
get "https://archive.ics.uci.edu/static/public/394/pm2+5+data+of+five+chinese+cities.zip" five_chinese_cities.zip \
    3d1e830a3950fe8c92bd8d034644f91b6185280fe315ee4bc8814b88ef7c303b
# The UCI archive wraps a RAR file; the Shanghai CSV must be extracted from it (needs unrar, unar or 7z).
CSV=ShanghaiPM20100101_20151231.csv
if [ ! -f "data/raw/$CSV" ]; then
  tmp=$(mktemp -d); unzip -q data/raw/five_chinese_cities.zip -d "$tmp"
  rar=$(find "$tmp" -name '*.rar' | head -1)
  if command -v unrar >/dev/null; then unrar e -inul "$rar" "$tmp/out/"
  elif command -v unar >/dev/null; then unar -q -o "$tmp/out" "$rar"
  elif command -v 7z >/dev/null; then 7z e -y -o"$tmp/out" "$rar" >/dev/null
  else echo "Install unrar, unar or 7z, or extract $CSV from $rar manually into data/raw/"; exit 1; fi
  cp "$(find "$tmp/out" -name "$CSV" | head -1)" "data/raw/$CSV"; rm -rf "$tmp"
fi
echo "49258afcca08ca1a5c76379c31a439cc0179f53371842f14fd6ee6127ebffc14  data/raw/$CSV" | shasum -a 256 -c -
