#!/bin/sh
# Downloads a BEIR benchmark dataset
# Usage: ./download_beir.sh [dataset_name]
# Available datasets: scifact, fiqa, arguana, trec-covid, nfcorpus
# Requires: wget, unzip
#
# Examples:
#   ./benchmarks/scripts/download_beir.sh              # downloads scifact (default)
#   ./benchmarks/scripts/download_beir.sh fiqa         # downloads fiqa
#   ./benchmarks/scripts/download_beir.sh trec-covid   # downloads trec-covid
#
# Output: benchmarks/data/beir/{dataset_name}/
#
# Dataset sizes:
#   scifact   ~5,183 docs   (recommended for first run)
#   nfcorpus  ~3,633 docs
#   arguana   ~8,674 docs
#   fiqa      ~57,638 docs
#   trec-covid ~171,332 docs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/benchmarks/data/beir"
BASE_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DATASET="${1:-scifact}"

# Validate dataset name
case "$DATASET" in
    scifact|fiqa|arguana|trec-covid|nfcorpus)
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'."
        echo ""
        echo "Available datasets:"
        echo "  scifact    (~5k docs,   recommended for testing)"
        echo "  nfcorpus   (~3.6k docs, medical)"
        echo "  arguana    (~8.7k docs, argumentation)"
        echo "  fiqa       (~57k docs,  finance Q&A)"
        echo "  trec-covid (~171k docs, biomedical)"
        echo ""
        echo "Usage: $0 [dataset_name]"
        exit 1
        ;;
esac

echo "=== BEIR Dataset Downloader ==="
echo "Dataset : $DATASET"

# Check dependencies
for cmd in wget unzip; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' not found. Please install it first."
        exit 1
    fi
done

mkdir -p "$DATA_DIR"

ZIP_URL="$BASE_URL/$DATASET.zip"
ZIP_FILE="$DATA_DIR/$DATASET.zip"
EXTRACT_DIR="$DATA_DIR/$DATASET"

echo "Downloading $ZIP_URL ..."
wget --continue --show-progress -O "$ZIP_FILE" "$ZIP_URL" || {
    echo "ERROR: download failed for $ZIP_URL"
    exit 1
}

echo "Extracting to $EXTRACT_DIR ..."
mkdir -p "$EXTRACT_DIR"
unzip -q -o "$ZIP_FILE" -d "$DATA_DIR" || {
    echo "ERROR: extraction failed."
    exit 1
}

echo ""
echo "=== Download complete ==="
echo "Path: $EXTRACT_DIR"

# Count documents if corpus file exists
CORPUS_FILE="$EXTRACT_DIR/corpus.jsonl"
if [ -f "$CORPUS_FILE" ]; then
    if command -v wc >/dev/null 2>&1; then
        count=$(wc -l < "$CORPUS_FILE")
        echo "Documents: $count"
    fi
else
    echo "Note: corpus.jsonl not found at expected path."
    echo "Contents of $EXTRACT_DIR:"
    ls "$EXTRACT_DIR" 2>/dev/null || echo "  (empty or not found)"
fi

echo ""
echo "NEXT STEPS:"
echo "  1. Embed corpus documents with your model of choice."
echo "     corpus.jsonl format: {\"_id\": \"...\", \"title\": \"...\", \"text\": \"...\"}"
echo "     Add a 'vector' field and rename '_id' to 'id' for vecdb."
echo "  2. Ingest into vecdb:"
echo "     vecdb ingest --collection beir-$DATASET --file $EXTRACT_DIR/corpus_embedded.jsonl"
echo "  3. Run benchmark:"
echo "     ./target/release/vecdb-bench --server http://localhost:8080 \\"
echo "         --collection beir-$DATASET --num-queries 100 --k 10"
