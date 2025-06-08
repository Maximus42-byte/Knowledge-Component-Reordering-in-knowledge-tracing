export BEST_RUN="knowledgetracing42-ucla/Knowledge-Component-Reordering-in-knowledge-tracing-examples/qq9etlqn"  # ← your best run id

python - <<'PY'
import os, wandb, pathlib, json, shutil, sys
run   = wandb.Api().run(os.environ["BEST_RUN"])
cfg   = run.config
base  = pathlib.Path(cfg["save_dir"]).expanduser().resolve()

def matches(local_cfg):
    keys = ("seed","dropout","emb_size","learning_rate")
    return all(str(local_cfg.get(k)) == str(cfg.get(k)) for k in keys)

folder = None
for p in base.rglob("config.json"):        # **recursive** search
    try:
        if matches(json.load(p.open())):
            folder = p.parent
            break
    except Exception:
        pass

if folder is None:
    sys.exit("❌  still couldn’t find the folder ― double-check that the run actually finished.")

print("📂  checkpoint folder:", folder)

dst = pathlib.Path("best_ckpt"); dst.mkdir(exist_ok=True)
for f in folder.glob("best*.*") | folder.glob("qid_model.ckpt"):
    shutil.copy2(f, dst / f.name)
shutil.copy2(folder / "config.json", dst / "config.json")
print("✅  copied to", dst)
PY
