#!/usr/bin/env bash
set -euo pipefail

# Script to download, extract, organize, and clean BIRD dev dataset.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/data/bird"
TMP_ZIP="/tmp/bird-dev.zip"
DOWNLOAD_URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

echo "=========================================================="
echo "      BIRD dev Dataset Downloader and Setup Script        "
echo "=========================================================="

mkdir -p "${TARGET_DIR}"

# Check if data is already downloaded and setup
if [ -f "${TARGET_DIR}/dev.json" ] && [ -d "${TARGET_DIR}/dev_databases" ]; then
    echo "BIRD dev dataset already exists at ${TARGET_DIR}."
    echo "Verifying databases..."
    COUNT=$(find "${TARGET_DIR}/dev_databases" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Found ${COUNT} SQLite database directories."
    echo "Setup already complete!"
    exit 0
fi

echo "1. Downloading official development dataset (~2 GB)..."
curl -L --fail --progress-bar "${DOWNLOAD_URL}" -o "${TMP_ZIP}"

echo "2. Extracting archive to ${TARGET_DIR}..."
unzip -q -o "${TMP_ZIP}" -d "${TARGET_DIR}"

echo "3. Organizing extracted files..."
# Handle subfolder extraction if present (e.g. dev_20240627)
for SUBDIR in "${TARGET_DIR}"/dev_*; do
    if [ -d "${SUBDIR}" ] && [ "$(basename "${SUBDIR}")" != "dev_databases" ]; then
        echo "Moving files from ${SUBDIR} to ${TARGET_DIR}..."
        mv "${SUBDIR}"/* "${TARGET_DIR}/" || true
        rmdir "${SUBDIR}" || true
    fi
done

# Extract dev_databases.zip if present
if [ -f "${TARGET_DIR}/dev_databases.zip" ]; then
    echo "4. Extracting dev_databases.zip..."
    unzip -q -o "${TARGET_DIR}/dev_databases.zip" -d "${TARGET_DIR}"
fi

echo "5. Cleaning up temporary archives and OS metadata..."
rm -f "${TMP_ZIP}"
rm -f "${TARGET_DIR}/dev_databases.zip"
rm -rf "${TARGET_DIR}/__MACOSX"
find "${TARGET_DIR}" -name ".DS_Store" -type f -delete 2>/dev/null || true

echo "=========================================================="
echo "Verification:"
if [ -f "${TARGET_DIR}/dev.json" ] && [ -d "${TARGET_DIR}/dev_databases" ]; then
    echo " [OK] dev.json found."
    DB_COUNT=$(find "${TARGET_DIR}/dev_databases" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo " [OK] ${DB_COUNT} database directories available in dev_databases/."
    echo "BIRD dev setup successfully completed!"
else
    echo " [ERROR] Setup incomplete. Please check extraction paths."
    exit 1
fi
echo "=========================================================="
