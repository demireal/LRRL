#!/bin/bash
# Shell script to generate count-GBM features for all tasks
# (acute MI, hyperlipidemia, hypertension, pancreatic cancer)

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values (using EHRSHOT_ASSETS pattern from other scripts)
PATH_TO_DATABASE="${PATH_TO_DATABASE:-${PROJECT_ROOT}/EHRSHOT_ASSETS/femr/extract}"
PATH_TO_LABELS_DIR="${PATH_TO_LABELS_DIR:-${PROJECT_ROOT}/EHRSHOT_ASSETS/benchmark}"
PATH_TO_FEATURES_DIR="${PATH_TO_FEATURES_DIR:-${PROJECT_ROOT}/EHRSHOT_ASSETS/features}"
OUTPUT_FILENAME="count_gbm_features.pkl"
IS_FORCE_REFRESH=false
NUM_THREADS="${NUM_THREADS:-1}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path_to_database)
            PATH_TO_DATABASE="$2"
            shift 2
            ;;
        --path_to_labels_dir)
            PATH_TO_LABELS_DIR="$2"
            shift 2
            ;;
        --path_to_features_dir)
            PATH_TO_FEATURES_DIR="$2"
            shift 2
            ;;
        --output_filename)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --num_threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --is_force_refresh)
            IS_FORCE_REFRESH=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --path_to_database PATH      Path to FEMR patient database (default: \$PROJECT_ROOT/EHRSHOT_ASSETS/femr/extract)"
            echo "  --path_to_labels_dir PATH    Path to directory containing labeled_patients.csv files (default: \$PROJECT_ROOT/EHRSHOT_ASSETS/benchmark)"
            echo "  --path_to_features_dir PATH Path to directory where features will be saved (default: \$PROJECT_ROOT/EHRSHOT_ASSETS/features)"
            echo "  --output_filename NAME       Output filename (default: count_gbm_features.pkl)"
            echo "  --num_threads INT            Number of threads to use (default: 1)"
            echo "  --is_force_refresh           Overwrite existing output file"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments (now with defaults, but check if paths exist)
if [ ! -d "$PATH_TO_DATABASE" ]; then
    echo "ERROR: Database directory not found: $PATH_TO_DATABASE"
    echo "Please provide --path_to_database or set PATH_TO_DATABASE environment variable"
    exit 1
fi

if [ ! -d "$PATH_TO_LABELS_DIR" ]; then
    echo "ERROR: Labels directory not found: $PATH_TO_LABELS_DIR"
    echo "Please provide --path_to_labels_dir or set PATH_TO_LABELS_DIR environment variable"
    exit 1
fi

# Create features directory if it doesn't exist
mkdir -p "$PATH_TO_FEATURES_DIR"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Count-GBM Feature Generation"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Database: $PATH_TO_DATABASE"
echo "Labels directory: $PATH_TO_LABELS_DIR"
echo "Features directory: $PATH_TO_FEATURES_DIR"
echo "Output filename: $OUTPUT_FILENAME"
echo "Num threads: $NUM_THREADS"
echo "Force refresh: $IS_FORCE_REFRESH"
echo "=========================================="
echo ""

# Build command
CMD_ARGS=(
    "python"
    "$SCRIPT_DIR/generate_count_features.py"
    "--path_to_database" "$PATH_TO_DATABASE"
    "--path_to_labels_dir" "$PATH_TO_LABELS_DIR"
    "--path_to_features_dir" "$PATH_TO_FEATURES_DIR"
    "--output_filename" "$OUTPUT_FILENAME"
    "--num_threads" "$NUM_THREADS"
)

if [ "$IS_FORCE_REFRESH" = true ]; then
    CMD_ARGS+=("--is_force_refresh")
fi

# Run command
echo "Running feature generation..."
echo ""
"${CMD_ARGS[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Feature generation completed successfully!"
    echo "=========================================="
    echo "Output saved to: $PATH_TO_FEATURES_DIR/$OUTPUT_FILENAME"
    echo ""
else
    echo ""
    echo "ERROR: Feature generation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
