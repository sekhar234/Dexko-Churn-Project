from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.synthetic.generator import load_config, generate, write_bundle
from src.synthetic.qa import run_qa, write_qa


def main():
    ap=argparse.ArgumentParser(description="Generate deterministic synthetic EDW data for the DexKo churn local replica.")
    ap.add_argument("--config",default=str(ROOT/"config"/"synthetic.yml"))
    ap.add_argument("--scale",choices=["small","medium","large"])
    ap.add_argument("--seed",type=int)
    ap.add_argument("--format",choices=["parquet","csv"])
    args=ap.parse_args()
    cfg=load_config(args.config)
    if args.scale: cfg["scale"]=args.scale
    if args.seed is not None: cfg["seed"]=args.seed
    if args.format: cfg["output_format"]=args.format
    print(f"Generating synthetic churn source data: scale={cfg['scale']} seed={cfg['seed']}")
    bundle=generate(cfg)
    files=write_bundle(bundle,ROOT,cfg.get("output_format","parquet"))
    qa=run_qa(bundle); write_qa(qa,ROOT)
    print("\nCreated:")
    for k,v in files.items(): print(f"  {k:28s} {v}")
    print("\nQA checks:")
    for k,v in qa["checks"].items(): print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
    if not all(qa["checks"].values()):
        raise SystemExit("Synthetic QA failed. Review data/qa/synthetic_qa.md")
    print("\nSynthetic source dataset is ready.")

if __name__=="__main__": main()
