#!/bin/bash

KAGGLE_DATASET="matthewjansen/ucf101-action-recognition"
TARGET_DIR="data"

pip install kaggle unzip

mkdir -p "$TARGET_DIR"
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR"

ZIP_FILE=$(ls "$TARGET_DIR"/*.zip | head -n 1)
if [ -f "$ZIP_FILE" ]; then
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"
else
    echo "No zip file found in $TARGET_DIR."
fi
