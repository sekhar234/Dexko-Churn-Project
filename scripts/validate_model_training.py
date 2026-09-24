from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

metadata_path = ROOT/"artifacts"/"models"/"category_churn_h14"/"metadata.json"
metrics_path = ROOT/"artifacts"/"models"/"category_churn_h14"/"metrics.json"
model_path = ROOT/"artifacts"/"models"/"category_churn_h14"/"model.json"
encoder_path = ROOT/"artifacts"/"models"/"category_churn_h14"/"encoder.joblib"
pred_path = ROOT/"artifacts"/"models"/"category_churn_h14"/"test_predictions.parquet"
mlflow_db = ROOT/"mlflow.db"

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics = payload["metrics"]
promotion = payload["promotion"]

checks = {
    "model_exists": model_path.exists(),
    "encoder_exists": encoder_path.exists(),
    "test_predictions_exist": pred_path.exists(),
    "mlflow_db_exists": mlflow_db.exists(),
    "raw_feature_count_44": metadata["raw_feature_count"] == 44,
    "encoded_features_ge_raw": metadata["encoded_feature_count"] >= 44,
    "two_categorical_features": metadata["categorical_feature_count"] == 2,
    "train_before_valid": metadata["split_boundaries"]["train_max"] < metadata["split_boundaries"]["valid_min"],
    "valid_before_test": metadata["split_boundaries"]["valid_max"] < metadata["split_boundaries"]["test_min"],
    "valid_auc_finite": math.isfinite(metrics["valid"]["auc"]),
    "valid_ap_finite": math.isfinite(metrics["valid"]["ap"]),
    "test_auc_finite": math.isfinite(metrics["test"]["auc"]),
    "test_ap_finite": math.isfinite(metrics["test"]["ap"]),
    "valid_auc_gate": metrics["valid"]["auc"] >= 0.70,
    "valid_ap_gate": metrics["valid"]["ap"] >= 0.25,
    "auc_overfit_gap_gate": promotion["auc_overfit_gap"] <= 0.20,
}

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "metrics": metrics,
    "promotion": promotion,
    "best_iteration": metadata["best_iteration"],
    "split_boundaries": metadata["split_boundaries"],
    "features": {
        "raw": metadata["raw_feature_count"],
        "encoded": metadata["encoded_feature_count"],
    },
}

out = ROOT/"data"/"qa"
out.mkdir(parents=True, exist_ok=True)
text = json.dumps(result, indent=2)
(out/"model_training_qa.json").write_text(text, encoding="utf-8")
print(text)

if result["status"] != "PASS":
    raise SystemExit(1)
