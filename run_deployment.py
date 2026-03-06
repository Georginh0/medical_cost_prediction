import argparse, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
from pipelines.deployment_pipeline import deployment_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--min-r2", type=float, default=0.80)
args = parser.parse_args()

print("\n╔══════════════════════════════════════════════════╗")
print("║   Medical Cost Prediction — Deployment Pipeline  ║")
print(f"║   Min R² threshold: {args.min_r2:<28.2f}║")
print("╚══════════════════════════════════════════════════╝\n")
deployment_pipeline(min_r2=args.min_r2)
print("\n✅  Deployment pipeline complete")
print("    Check: extracted_data/deployment_manifest.json")
