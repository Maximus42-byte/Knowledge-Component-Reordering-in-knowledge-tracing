#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
SRC_DIR="/home/maximus/Desktop/KT"    # parent directory containing "data"
DEST_DIR="/home/maximus/Desktop/KT/pykt-toolkit/data/assist2009"
METRICS_CSV="./metrics.csv"           # metrics file stays outside DEST_DIR

# ─── INIT METRICS CSV ───────────────────────────────────────────────────────────
if [[ ! -e "$METRICS_CSV" ]]; then
  echo "Embedding model,number of clusters,AUC,ACC" > "$METRICS_CSV"
  echo "Created metrics file: $METRICS_CSV"
fi

# ─── PREPARE DESTINATION ────────────────────────────────────────────────────────
mkdir -p "$DEST_DIR"

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────────
for subdir in "$SRC_DIR/data"/*; do
  [[ -d "$subdir" ]] || continue
  embedding_model=$(basename "$subdir")

  # Process each CSV within this subdirectory
  for src_file in "$subdir"/*.csv; do
    [[ -f "$src_file" ]] || continue
    filename=$(basename "$src_file")
    # Extract number of clusters from filename (digits before .csv)
    cluster_count=$(basename "$src_file" .csv | grep -oP '[0-9]+' || echo "$filename")

    echo "🔄 Processing model='$embedding_model', clusters='$cluster_count'"

    # ─── CLEANUP DESTINATION ─────────────────────────────────────────────────────
    echo "🗑  Clearing out $DEST_DIR"
    rm -rf "$DEST_DIR"/*

    # ─── COPY & RENAME ───────────────────────────────────────────────────────────
    cp "$src_file" "$DEST_DIR/skill_builder_data_corrected_collapsed.csv"
    echo "  ↳ Copied → skill_builder_data_corrected_collapsed.csv"

    # ─── STEP 1: DATA PREPROCESSING ─────────────────────────────────────────────
    echo "🏗  [1/4] Preprocessing (assist2009)…"
    python data_preprocess.py --dataset_name=assist2009

    # ─── STEP 2: MODEL TRAINING ─────────────────────────────────────────────────
    echo "🤖  [2/4] Training (assist2009)…"
    python wandb_dkt_train.py --dataset_name=assist2009 --use_wandb=0 --add_uuid=0

    # ─── DETECT SAVED MODEL ──────────────────────────────────────────────────────
    # find the most recently created/modified folder under saved_model
    model_dir=$(ls -td saved_model/*/ | head -n1)
    echo "📁 Using trained model directory: $model_dir"

    # ─── STEP 3: PREDICTION ───────────────────────────────────────────────────────
    echo "📈  [3/4] Predicting…"
    predict_output=$(python wandb_predict.py \
      --save_dir="$model_dir" \
      --use_wandb=0)
    echo "$predict_output"

    # # ─── PARSE & APPEND METRICS ──────────────────────────────────────────────────
    # auc=$(echo "$predict_output" | grep -oP 'testauc:\s*\K[0-9]+(\.[0-9]+)?' || echo "NA")
    # acc=$(echo "$predict_output" | grep -oP 'testacc:\s*\K[0-9]+(\.[0-9]+)?' || echo "NA")

 # ─── STEP 4: PARSE & APPEND METRICS ──────────────────────────────────────────
    # extract only the final dictionary line and parse testauc/testacc
    dict_line=$(echo "$predict_output" | grep -oP '^\{.*\}$' | tail -n1)
    auc=$(echo "$dict_line" | grep -oP "'testauc':\s*\K[0-9]+(?:\.[0-9]+)?" || echo "NA")
    acc=$(echo "$dict_line" | grep -oP "'testacc':\s*\K[0-9]+(?:\.[0-9]+)?" || echo "NA")

    echo "  ➤ testauc=$auc, testacc=$acc"
    echo "$embedding_model,$cluster_count,$auc,$acc" >> "$METRICS_CSV"
    echo "  ↳ Appended to $METRICS_CSV"

    echo "✅ Done with $filename"
    echo "###################################################################################"
    echo "###################################################################################"
    echo "###################################################################################"
    echo

  done
done

echo "🎉 All files processed! Metrics written to $METRICS_CSV"
