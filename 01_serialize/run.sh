#!/usr/bin/env bash
# =============================================================================
# 01_serialize/run.sh -- Serialize EHRs into naivetext format.
#
# Usage:
#   bash 01_serialize/run.sh
#
# Required environment variables:
#   EHRSHOT_DB     -- path to FEMR extract directory (e.g. /path/to/EHRSHOT_ASSETS/femr/extract)
#   EHRSHOT_LABELS -- path to benchmark labels directory (e.g. /path/to/EHRSHOT_ASSETS/benchmark)
#   EHRSHOT_SPLITS -- path to person_id_map.csv (e.g. /path/to/EHRSHOT_ASSETS/splits/person_id_map.csv)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_PATH="${EHRSHOT_DB:?Set EHRSHOT_DB to your FEMR extract directory (e.g. /path/to/EHRSHOT_ASSETS/femr/extract)}"
LABELS_DIR="${EHRSHOT_LABELS:?Set EHRSHOT_LABELS to your benchmark labels directory (e.g. /path/to/EHRSHOT_ASSETS/benchmark)}"
SPLITS_FILE="${EHRSHOT_SPLITS:?Set EHRSHOT_SPLITS to your person_id_map.csv (e.g. /path/to/EHRSHOT_ASSETS/splits/person_id_map.csv)}"
DATA_DIR="${SCRIPT_DIR}/../data/serialized"

echo "=== Generating naivetext serializations ==="
python "$SCRIPT_DIR/serialize.py" \
    --path_to_database "$DB_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --path_to_splits "$SPLITS_FILE" \
    --output_dir "$DATA_DIR/naivetext" \
    --mode naivetext \
    --force \
    --skip_existing

echo ""
echo "=== Generating unsubsampled naivetext serializations ==="
python "$SCRIPT_DIR/serialize.py" \
    --path_to_database "$DB_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --path_to_splits "$SPLITS_FILE" \
    --output_dir "$DATA_DIR/naivetext_all" \
    --mode naivetext \
    --no_balance \
    --force \
    --skip_existing

echo ""
echo "=== Done. Outputs in $DATA_DIR ==="
