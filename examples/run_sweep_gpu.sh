#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─── INPUT PARAMETERS ───────────────────────────────────────────────────────────
method="$2"
dims="$3"
cluster_count="$4"
PROJECT="kt_assist2009_dkt_${method}_${dims}_${cluster_count}_gpu"

echo "🆕  Sweep will be under project:  $PROJECT"

# ─── CLEANUP ────────────────────────────────────────────────────────────────────
echo "🧹  Removing best_ckpt/, models/, wandb/, all_wandbs/"
rm -rf best_ckpt/ models/ wandb/ all_wandbs/

# ─── GENERATE SWEEP YAML ────────────────────────────────────────────────────────
echo "🛠️   Generating sweep YAML..."
python generate_wandb.py \
  --dataset_names assist2009 \
  --model_names dkt \
  --project_name "$PROJECT" \


# ─── EXTRACT FIRST YAML PATH (ONLY USE ONE FOLD FOR SWEEP) ──────────────────────
template_yaml=$(ls all_wandbs/assist2009_dkt_qid_0.yaml)

echo "  ↳ using template $template_yaml"

# ─── REGISTER SWEEP AND EXTRACT SWEEP ID ────────────────────────────────────────
echo "📡  Creating sweep from: $template_yaml"


# Create the sweep and extract ID using awk
sweep_output=$(wandb sweep "$template_yaml" 2>&1 | tee sweep_log.txt)

sweep_id=$(echo "$sweep_output" | awk '/Run sweep agent with:/ {print $NF}' | awk -F/ '{print $NF}')

if [[ -z "$sweep_id" ]]; then
  echo "❌ Failed to extract sweep ID. Check sweep_log.txt for full output."
  exit 1
fi

echo "🚀 Launching wandb agent for sweep ID: $sweep_id"
wandb agent "knowledgetracing42-ucla/Knowledge-Component-Reordering-in-knowledge-tracing-examples/$sweep_id" --count 10



# ─── EXTRACT BEST RUN AND CHECKPOINT ────────────────────────────────────────────
echo "🔍  Finding best run…"
best_run=$(python - "$sweep_id" <<PY
import wandb, sys

sweep_id = sys.argv[1]
api = wandb.Api()
sweep = api.sweep(f"knowledgetracing42-ucla/Knowledge-Component-Reordering-in-knowledge-tracing-examples/{sweep_id}")
best = max(sweep.runs, key=lambda r: r.summary.get("best_valid_auc", 0))
print(best.id)
PY
)

echo "🏆  Best run ID: $best_run"
# # Get best run config
# echo "🔍  Finding best run…"
# best_run=$(python - <<'PY'
# import wandb
# api = wandb.Api()
# sweep = api.sweep("knowledgetracing42-ucla/Knowledge-Component-Reordering-in-knowledge-tracing-examples/$sweep_id")
# best = max(sweep.runs, key=lambda r: r.summary.get("best_valid_auc", 0))
# print(best.id)
# PY
# )

# echo "🏆  Best run ID: $best_run"

# Look up run config to extract save_dir and other parameters
python - <<PY
import wandb, pathlib, sys, re, json

run = wandb.Api().run("knowledgetracing42-ucla/Knowledge-Component-Reordering-in-knowledge-tracing-examples/$best_run")
cfg = run.config
base = pathlib.Path("models/dkt_tiaocan_assist2009/assist2009_dkt_qid_models")

if not base.exists():
    sys.exit(f"❌  Base path {base} does not exist")

# Build a matching pattern from key hyperparams
parts = [
    str(cfg.get("seed")),
    str(cfg.get("fold", 0)),
    str(cfg.get("dropout")),
    str(cfg.get("emb_size")),
    str(cfg.get("learning_rate")),
]
pattern = ".*" + ".*".join(re.escape(p) for p in parts) + ".*"

matches = [p for p in base.iterdir() if p.is_dir() and re.match(pattern, p.name)]
if not matches:
    sys.exit("❌  No matching local folder found for best run config")

best_folder = max(matches, key=lambda p: p.stat().st_mtime)
print("📂  Best checkpoint folder:", best_folder)
ckpt = best_folder / "qid_model.ckpt"
cfgf = best_folder / "config.json"

if ckpt.exists() and cfgf.exists():
    import shutil
    dest = pathlib.Path("best_ckpt")
    dest.mkdir(exist_ok=True)
    shutil.copy(ckpt, dest / ckpt.name)
    shutil.copy(cfgf, dest / cfgf.name)
    print("✅  Copied checkpoint and config to best_ckpt/")
else:
    print("❌  Missing checkpoint or config in folder:", best_folder)
PY