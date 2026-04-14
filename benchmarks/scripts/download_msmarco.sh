#!/bin/sh
# Downloads MS MARCO passage corpus and converts to vecdb JSONL format
# Requires: wget, python3
# Output: benchmarks/data/msmarco_passages.jsonl
#
# Usage:
#   ./benchmarks/scripts/download_msmarco.sh           # full download
#   ./benchmarks/scripts/download_msmarco.sh --dry-run # print what would happen

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/benchmarks/data"
OUTPUT_FILE="$DATA_DIR/msmarco_passages.jsonl"
TARBALL_URL="https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz"
TARBALL_FILE="$DATA_DIR/collection.tar.gz"

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== DRY RUN — no files will be downloaded ==="
    echo ""
    echo "Would create directory : $DATA_DIR"
    echo "Would download         : $TARBALL_URL"
    echo "Would save to          : $TARBALL_FILE"
    echo "Would extract to       : $DATA_DIR/collection.tsv"
    echo "Would convert to JSONL : $OUTPUT_FILE"
    echo ""
    echo "Download size  : ~8.8 GB"
    echo "Extracted size : ~16 GB"
    echo "JSONL size     : ~4 GB (text only, no vectors)"
    echo ""
    echo "JSONL format per line:"
    echo '  {"id": "7154321", "text": "passage text here..."}'
    echo ""
    echo "NOTE: Vectors are NOT included. Embed passages separately before ingesting."
    exit 0
fi

echo "=== MS MARCO Passage Corpus Downloader ==="

# Check dependencies
for cmd in wget python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' not found. Please install it first."
        exit 1
    fi
done

mkdir -p "$DATA_DIR"

echo "Downloading MS MARCO collection (~8.8 GB)..."
echo "URL: $TARBALL_URL"
wget --continue --show-progress -O "$TARBALL_FILE" "$TARBALL_URL" || {
    echo "ERROR: download failed."
    exit 1
}

echo "Extracting archive..."
tar -xzf "$TARBALL_FILE" -C "$DATA_DIR" || {
    echo "ERROR: extraction failed."
    exit 1
}

TSV_FILE="$DATA_DIR/collection.tsv"
if [ ! -f "$TSV_FILE" ]; then
    echo "ERROR: extraction did not produce $TSV_FILE"
    exit 1
fi

echo "Converting TSV to JSONL..."
python3 << PYEOF
import json, sys, os

tsv_path = "$TSV_FILE"
out_path = "$OUTPUT_FILE"
count = 0

with open(tsv_path, "r", encoding="utf-8") as fin, \\
     open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.rstrip("\\n").split("\\t", 1)
        if len(parts) != 2:
            continue
        pid, passage = parts
        fout.write(json.dumps({"id": pid, "text": passage}, ensure_ascii=False) + "\\n")
        count += 1
        if count % 500000 == 0:
            print(f"  {count:,} passages processed...", file=sys.stderr)

print(f"Done: {count:,} passages written to {out_path}")
PYEOF

echo ""
echo "=== Download complete ==="
echo "Output: $OUTPUT_FILE"
echo ""
echo "NEXT STEPS:"
echo "  1. Embed passages with your model of choice (sentence-transformers, OpenAI, etc.)"
echo "     Add a 'vector' field to each JSONL record."
echo "  2. Ingest into vecdb:"
echo "     vecdb ingest --collection msmarco --file benchmarks/data/msmarco_embedded.jsonl"
echo "  3. Run benchmark:"
echo "     ./target/release/vecdb-bench --server http://localhost:8080 \\"
echo "         --collection msmarco --num-queries 1000 --k 10"
