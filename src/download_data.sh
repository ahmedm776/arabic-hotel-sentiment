#!/bin/sh
# Download the HARD balanced dataset from its canonical source and verify integrity.
# Source: https://github.com/elnagara/HARD-Arabic-Dataset (data/balanced-reviews.zip)
# Rights: the authors distribute the dataset for research use with citation
# (Elnagar, Khalifa & Einea 2018, doi:10.1007/978-3-319-67056-0_3). No explicit
# license file ships with the dataset, so it is NOT redistributed inside this
# project - download it from the canonical source instead.
set -e
cd "$(dirname "$0")/../data"
curl -sL -o balanced-reviews.zip https://raw.githubusercontent.com/elnagara/HARD-Arabic-Dataset/master/data/balanced-reviews.zip
echo "1939c1ca59ff50bd3887223b153c7fb5c9fd232b405320244c55791bc8b5d448  balanced-reviews.zip" | sha256sum -c - || { echo "checksum mismatch - source may have changed"; exit 1; }
unzip -o balanced-reviews.zip
